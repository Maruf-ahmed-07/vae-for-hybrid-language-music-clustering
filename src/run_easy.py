from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from tqdm import tqdm

from .audio_features import MFCCConfig, extract_mfcc_stats
from .clustering import KMeansConfig, pca_features, run_kmeans, tsne_2d
from .dataset import load_audio_paths
from .evaluation import clustering_metrics
from .vae import VAEConfig, encode_latents, train_vae


def _pick_best_by_silhouette(rows: list[dict]) -> dict:
    def key(r: dict) -> tuple[float, float]:
        return (float(r.get("silhouette", float("-inf"))), -float(r.get("davies_bouldin", float("inf"))))

    return max(rows, key=key)


def _int_or_none(value: str) -> int | None:
    v = value.strip().lower()
    if v in {"none", "null", "all"}:
        return None
    return int(value)

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Easy task: audio MFCC -> (PCA/VAE) -> KMeans clustering")
    parser.add_argument(
        "--max-tracks",
        type=_int_or_none,
        default=1000,
        help="Max number of tracks to use (default: 1000). Use 'none' for all.",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="audio",
        help="Subfolder under results/easy/ to write outputs (default: audio).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "results" / "easy" / args.out_subdir
    vis_dir = out_dir / "latent_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Use a manageable subset for runtime. Set to None to use all tracks.
    meta_df = load_audio_paths(repo_root, max_rows=args.max_tracks)

    y_true_all = meta_df["genre_label"].astype(str).to_numpy() if "genre_label" in meta_df.columns else None

    cfg = MFCCConfig()
    feats: list[np.ndarray] = []
    keep_rows: list[int] = []
    errors = 0

    for i, row in tqdm(list(meta_df.iterrows()), desc="Extracting MFCC", total=len(meta_df)):
        rel = row["audio_path"]
        p = repo_root / str(rel)
        try:
            f = extract_mfcc_stats(p, cfg)
        except Exception:
            errors += 1
            continue
        feats.append(f)
        keep_rows.append(i)

    if not feats:
        raise RuntimeError(
            "No audio features extracted. If MP3 decoding failed, install ffmpeg and ensure it's on PATH, "
            "then re-run."
        )

    used_df = meta_df.loc[keep_rows].reset_index(drop=True)
    y_true = y_true_all[keep_rows] if y_true_all is not None else None
    X = np.stack(feats, axis=0)

    # Standardize then L2-normalize for cosine-ish geometry
    X = StandardScaler().fit_transform(X)
    X = normalize(X, norm="l2")

    # ---- Baseline: PCA(16) + KMeans (k sweep) ----
    X_pca16 = pca_features(X, n_components=16)
    X_pca16_n = normalize(X_pca16, norm="l2")
    k_list = [6, 8, 10, 12, 15]

    pca_rows: list[dict] = []
    for k in k_list:
        labels = run_kmeans(X_pca16_n, KMeansConfig(n_clusters=k))
        m = clustering_metrics(X_pca16_n, labels, labels_true=y_true)
        pca_rows.append({"task": "easy", "repr": "mfcc->pca16_l2", "k": k, "method": f"mfcc->pca16_l2+kmeans{k}", **m})
    best_pca = _pick_best_by_silhouette(pca_rows)
    labels_pca = run_kmeans(X_pca16_n, KMeansConfig(n_clusters=int(best_pca["k"])))

    # ---- VAE: MFCC -> latent -> KMeans (k sweep) ----
    # Small hyperparameter sweep to find a good representation on the full dataset.
    # (Keeps the project "Easy"—still MFCC stats + MLP VAE + KMeans—but avoids one fixed config.)
    vae_candidates = [
        VAEConfig(input_dim=X.shape[1], latent_dim=16, hidden_dim=128, epochs=60, beta=0.1, seed=42),
        VAEConfig(input_dim=X.shape[1], latent_dim=16, hidden_dim=256, epochs=120, beta=0.01, seed=42),
        VAEConfig(input_dim=X.shape[1], latent_dim=32, hidden_dim=256, epochs=120, beta=0.01, seed=42),
        VAEConfig(input_dim=X.shape[1], latent_dim=16, hidden_dim=256, epochs=120, beta=0.001, seed=42),
    ]

    vae_rows: list[dict] = []
    best_vae: dict | None = None
    best_Z_n: np.ndarray | None = None
    best_labels_vae: np.ndarray | None = None

    for cfg_i, vae_cfg in enumerate(vae_candidates, start=1):
        vae = train_vae(X, vae_cfg)
        Z = encode_latents(vae, X)
        Z_n = normalize(Z, norm="l2")

        tag = f"ld{vae_cfg.latent_dim}_hd{vae_cfg.hidden_dim}_b{vae_cfg.beta}_e{vae_cfg.epochs}"
        repr_name = f"mfcc->vae_{tag}_l2"

        rows_this_cfg: list[dict] = []
        for k in k_list:
            labels = run_kmeans(Z_n, KMeansConfig(n_clusters=k))
            m = clustering_metrics(
                Z_n,
                labels,
                labels_true=y_true,
            )
            rows_this_cfg.append(
                {
                    "task": "easy",
                    "repr": repr_name,
                    "k": k,
                    "method": f"{repr_name}+kmeans{k}",
                    "vae_latent_dim": vae_cfg.latent_dim,
                    "vae_hidden_dim": vae_cfg.hidden_dim,
                    "vae_beta": vae_cfg.beta,
                    "vae_epochs": vae_cfg.epochs,
                    "vae_seed": vae_cfg.seed,
                    **m,
                }
            )

        vae_rows.extend(rows_this_cfg)
        best_this = _pick_best_by_silhouette(rows_this_cfg)

        if (best_vae is None) or (
            float(best_this.get("silhouette", float("-inf"))) > float(best_vae.get("silhouette", float("-inf")))
        ):
            best_vae = best_this
            best_Z_n = Z_n
            best_labels_vae = run_kmeans(Z_n, KMeansConfig(n_clusters=int(best_this["k"])))

    if best_vae is None or best_Z_n is None or best_labels_vae is None:
        raise RuntimeError("VAE tuning failed: no VAE rows were produced.")

    # ---- Visualize (t-SNE) ----
    tsne_max_points = 2000
    rng = np.random.default_rng(42)
    if best_Z_n.shape[0] > tsne_max_points:
        idx = rng.choice(best_Z_n.shape[0], size=tsne_max_points, replace=False)
    else:
        idx = np.arange(best_Z_n.shape[0])

    Z2 = tsne_2d(best_Z_n[idx])
    plt.figure(figsize=(7, 5))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=best_labels_vae[idx], s=10, cmap="tab10")
    title_suffix = "" if idx.shape[0] == best_Z_n.shape[0] else f" (sample={idx.shape[0]})"
    plt.title(
        f"Easy(audio): MFCC->VAE(best,l2) + KMeans (k={int(best_vae['k'])}) (t-SNE){title_suffix}"
    )
    plt.tight_layout()
    plt.savefig(vis_dir / "vae_kmeans_tsne.png", dpi=200)
    plt.close()

    Xp2 = tsne_2d(X_pca16_n[idx])
    plt.figure(figsize=(7, 5))
    plt.scatter(Xp2[:, 0], Xp2[:, 1], c=labels_pca[idx], s=10, cmap="tab10")
    plt.title(
        f"Easy(audio): MFCC->PCA(16,l2) + KMeans (k={int(best_pca['k'])}) (t-SNE){title_suffix}"
    )
    plt.tight_layout()
    plt.savefig(vis_dir / "pca_kmeans_tsne.png", dpi=200)
    plt.close()

    sweep_df = pd.DataFrame(pca_rows + vae_rows)
    sweep_path = out_dir / "clustering_metrics_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)

    best_df = pd.DataFrame([
        {k: v for k, v in best_pca.items() if k != "repr"},
        {k: v for k, v in best_vae.items() if k != "repr"},
    ])
    best_path = out_dir / "clustering_metrics.csv"
    best_df.to_csv(best_path, index=False)

    used_df.to_csv(out_dir / "used_tracks.csv", index=False)

    print("Saved:")
    print(f"- {best_path}")
    print(f"- {sweep_path}")
    print(f"- {out_dir / 'used_tracks.csv'}")
    print(f"- {vis_dir / 'vae_kmeans_tsne.png'}")
    print(f"- {vis_dir / 'pca_kmeans_tsne.png'}")
    if errors:
        print(f"Warning: {errors} audio files failed to decode/featurize and were skipped.")


if __name__ == "__main__":
    main()
