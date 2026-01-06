from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from tqdm import tqdm

from .audio_features import MFCCConfig, extract_logmel_sequence
from .autoencoder import AEConfig, encode_ae_latents, train_autoencoder
from .dataset import load_audio_lyrics_pairs
from .evaluation import clustering_metrics_safe
from .lyrics_features import LyricsEmbeddingConfig, compute_lyrics_embeddings, load_lyrics_text
from .multimodal_spec_cvae import (
    MultiModalSpecCVAEConfig,
    encode_multimodal_spec_cvae_latents,
    reconstruct_examples,
    train_multimodal_spec_cvae,
)


def _int_or_none(value: str) -> int | None:
    v = value.strip().lower()
    if v in {"none", "null", "all"}:
        return None
    return int(value)


def _parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for part in value.split(","):
        p = part.strip()
        if not p:
            continue
        items.append(int(p))
    if not items:
        raise argparse.ArgumentTypeError("k-list must contain at least one integer")
    return items


def _one_hot_from_labels(labels: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    labels = np.asarray(labels).astype(str)
    uniq = sorted(set(labels.tolist()))
    mapping = {g: i for i, g in enumerate(uniq)}
    out = np.zeros((labels.shape[0], len(uniq)), dtype=np.float32)
    for i, g in enumerate(labels.tolist()):
        out[i, mapping[g]] = 1.0
    return out, mapping


def _plot_latent_2d(Z2: np.ndarray, y: np.ndarray, title: str, out_path: Path, cmap: str = "tab20") -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=y, s=10, cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_cluster_genre_distribution(labels: np.ndarray, genres: np.ndarray, out_path: Path) -> None:
    labels = np.asarray(labels)
    genres = np.asarray(genres).astype(str)

    clusters = np.unique(labels)
    uniq_genres = sorted(set(genres.tolist()))

    mat = np.zeros((clusters.size, len(uniq_genres)), dtype=np.int64)
    for i, c in enumerate(clusters.tolist()):
        idx = labels == c
        gvals = genres[idx]
        for j, g in enumerate(uniq_genres):
            mat[i, j] = int(np.sum(gvals == g))

    plt.figure(figsize=(max(8, len(uniq_genres) * 0.5), max(5, clusters.size * 0.35)))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="count")
    plt.xticks(ticks=np.arange(len(uniq_genres)), labels=uniq_genres, rotation=90)
    plt.yticks(ticks=np.arange(clusters.size), labels=[str(c) for c in clusters.tolist()])
    plt.xlabel("Genre")
    plt.ylabel("Cluster")
    plt.title("Cluster distribution over genres")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_recon_grid(orig: np.ndarray, recon: np.ndarray, out_path: Path, n_show: int = 6) -> None:
    # orig/recon: (B, 1, n_mels, n_frames)
    B = int(min(n_show, orig.shape[0]))
    if B <= 0:
        return

    plt.figure(figsize=(12, 2.2 * B))
    for i in range(B):
        o = orig[i, 0]
        r = recon[i, 0]

        ax1 = plt.subplot(B, 2, 2 * i + 1)
        ax1.imshow(o, aspect="auto", origin="lower")
        ax1.set_title(f"Original {i}")
        ax1.axis("off")

        ax2 = plt.subplot(B, 2, 2 * i + 2)
        ax2.imshow(r, aspect="auto", origin="lower")
        ax2.set_title(f"Reconstruction {i}")
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hard task: CVAE conditioned on genre + multimodal clustering")

    parser.add_argument("--max-tracks", type=_int_or_none, default=None)
    parser.add_argument("--out-subdir", type=str, default="hard_cvae_spec_v1")

    parser.add_argument("--spec-frames", type=int, default=256)
    parser.add_argument("--spec-mels", type=int, default=64)

    parser.add_argument("--lyrics-svd-dim", type=int, default=256)
    parser.add_argument("--lyrics-max-features", type=int, default=20000)

    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--lyrics-hidden-dim", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--audio-recon-weight", type=float, default=1.0)
    parser.add_argument("--lyrics-recon-weight", type=float, default=1.0)

    parser.add_argument("--k-list", type=_parse_int_list, default=_parse_int_list("6,8,10"))

    parser.add_argument("--viz-max-points", type=int, default=2000)
    parser.add_argument("--recon-examples", type=int, default=6)

    args = parser.parse_args(argv)

    if int(args.spec_frames) <= 0 or int(args.spec_mels) <= 0:
        raise SystemExit("--spec-frames and --spec-mels must be positive")
    if (int(args.spec_frames) % 8) != 0 or (int(args.spec_mels) % 8) != 0:
        raise SystemExit("--spec-frames and --spec-mels must be divisible by 8")

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "results" / "hard" / args.out_subdir
    vis_dir = out_dir / "visualizations"
    rec_dir = out_dir / "reconstructions"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    df = load_audio_lyrics_pairs(repo_root, max_rows=args.max_tracks)
    if df.empty:
        raise RuntimeError("No rows found with BOTH audio and lyrics files.")

    if "genre_label" not in df.columns:
        raise RuntimeError("Hard requires genre_label column for conditioning.")

    genres_all = df["genre_label"].astype(str).to_numpy()

    # ---- Extract spectrograms + audio stats ----
    mfcc_cfg = MFCCConfig(duration_sec=30.0, n_mfcc=40)
    spec_list: list[np.ndarray] = []
    audio_stats: list[np.ndarray] = []
    keep_rows: list[int] = []
    errors = 0

    for i, row in tqdm(list(df.iterrows()), desc="Extracting log-mel spectrograms", total=len(df)):
        ap = repo_root / str(row["audio_path"])
        try:
            spec = extract_logmel_sequence(ap, mfcc_cfg, n_frames=int(args.spec_frames), n_mels=int(args.spec_mels))
        except Exception:
            errors += 1
            continue

        spec_list.append(spec)
        mean = spec.mean(axis=1)
        std = spec.std(axis=1)
        audio_stats.append(np.concatenate([mean, std], axis=0).astype(np.float32))
        keep_rows.append(i)

    if not spec_list:
        raise RuntimeError("No spectrograms extracted.")

    used_df = df.loc[keep_rows].reset_index(drop=True)
    genres = genres_all[keep_rows]

    X_spec = np.stack(spec_list, axis=0).astype(np.float32)  # (N, n_mels, n_frames)
    X_audio = X_spec[:, None, :, :]

    # Per-sample standardization
    mean = X_audio.mean(axis=(2, 3), keepdims=True)
    std = X_audio.std(axis=(2, 3), keepdims=True) + 1e-6
    X_audio = (X_audio - mean) / std

    X_audio_stats = np.stack(audio_stats, axis=0).astype(np.float32)

    # ---- Lyrics embeddings ----
    texts: list[str] = []
    for _, row in used_df.iterrows():
        lp = repo_root / str(row["lyrics_path"])
        texts.append(load_lyrics_text(lp))

    lyr_cfg = LyricsEmbeddingConfig(
        max_features=int(args.lyrics_max_features),
        svd_dim=int(args.lyrics_svd_dim),
        min_df=2,
        ngram_range=(2, 5),
    )
    X_lyrics = compute_lyrics_embeddings(texts, lyr_cfg)

    # Standardize lyrics
    lyrics_scaler = StandardScaler()
    X_lyrics_s = lyrics_scaler.fit_transform(X_lyrics).astype(np.float32)

    # ---- Condition (genre one-hot) ----
    C, genre_to_id = _one_hot_from_labels(genres)

    # ---- Train CVAE ----
    cfg = MultiModalSpecCVAEConfig(
        n_mels=int(args.spec_mels),
        n_frames=int(args.spec_frames),
        lyrics_dim=int(X_lyrics_s.shape[1]),
        cond_dim=int(C.shape[1]),
        latent_dim=int(args.latent_dim),
        hidden_channels=int(args.hidden_channels),
        lyrics_hidden_dim=int(args.lyrics_hidden_dim),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        beta=float(args.beta),
        audio_recon_weight=float(args.audio_recon_weight),
        lyrics_recon_weight=float(args.lyrics_recon_weight),
        seed=42,
    )

    model = train_multimodal_spec_cvae(X_audio, X_lyrics_s, C, cfg, verbose=True)

    Z_cvae = encode_multimodal_spec_cvae_latents(
        model,
        X_audio,
        X_lyrics_s,
        C,
        batch_size=int(cfg.batch_size),
    )
    Z_cvae_n = normalize(Z_cvae, norm="l2")

    # ---- Baselines ----
    # PCA + KMeans on concatenated stats+lyrics
    stats_s = StandardScaler().fit_transform(X_audio_stats)
    stats_s = normalize(stats_s, norm="l2")

    X_concat = np.concatenate([stats_s, X_lyrics], axis=1)
    X_concat = normalize(X_concat, norm="l2")
    X_pca32 = PCA(n_components=32, random_state=42).fit_transform(X_concat)
    X_pca32 = normalize(X_pca32, norm="l2")

    # Autoencoder + KMeans on concatenated features (same input as PCA baseline)
    ae_cfg = AEConfig(input_dim=int(X_concat.shape[1]), latent_dim=32, hidden_dim=256, epochs=60, batch_size=64)
    ae = train_autoencoder(X_concat.astype(np.float32), ae_cfg, verbose=True)
    Z_ae = encode_ae_latents(ae, X_concat.astype(np.float32), batch_size=512)
    Z_ae_n = normalize(Z_ae, norm="l2")

    # Direct spectral clustering: kmeans on standardized audio stats (no PCA)
    X_direct_spec = stats_s

    # ---- Clustering + metrics ----
    k_list = list(dict.fromkeys(int(k) for k in args.k_list))
    rows: list[dict] = []

    reps: list[tuple[str, np.ndarray]] = [
        ("cvae_latent", Z_cvae_n),
        ("pca_concat", X_pca32),
        ("ae_concat", Z_ae_n),
        ("direct_spectral", X_direct_spec),
    ]

    y_true = genres.astype(str)

    for repr_name, Xrepr in reps:
        for k in k_list:
            labels = KMeans(n_clusters=int(k), random_state=42, n_init=10).fit_predict(Xrepr)
            m = clustering_metrics_safe(Xrepr, labels, labels_true=y_true)
            rows.append(
                {
                    "task": "hard",
                    "repr": repr_name,
                    "algo": f"kmeans{k}",
                    "k": k,
                    "silhouette": m.get("silhouette", np.nan),
                    "davies_bouldin": m.get("davies_bouldin", np.nan),
                    "calinski_harabasz": m.get("calinski_harabasz", np.nan),
                    "ari": m.get("ari", np.nan),
                    "nmi": m.get("nmi", np.nan),
                    "purity": m.get("purity", np.nan),
                    "latent_dim": cfg.latent_dim if repr_name == "cvae_latent" else np.nan,
                    "beta": cfg.beta if repr_name == "cvae_latent" else np.nan,
                    "spec_mels": int(args.spec_mels),
                    "spec_frames": int(args.spec_frames),
                    "lyrics_svd_dim": lyr_cfg.svd_dim,
                    "lyrics_max_features": lyr_cfg.max_features,
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "clustering_metrics.csv", index=False)

    # Best by ARI (Hard is label-aware)
    tmp = metrics_df.copy()
    tmp["_ari"] = pd.to_numeric(tmp["ari"], errors="coerce").fillna(float("-inf"))
    best_by_repr = []
    for repr_name, _ in reps:
        sub = tmp[tmp["repr"] == repr_name]
        if sub.empty:
            continue
        best_by_repr.append(sub.sort_values(by=["_ari"], ascending=[False]).iloc[0].drop(labels=["_ari"]).to_dict())
    best_df = pd.DataFrame(best_by_repr)
    best_df.to_csv(out_dir / "best_by_ari.csv", index=False)

    # ---- Visualizations ----
    rng = np.random.default_rng(42)
    max_points = int(args.viz_max_points)
    if Z_cvae_n.shape[0] > max_points:
        idx = rng.choice(Z_cvae_n.shape[0], size=max_points, replace=False)
    else:
        idx = np.arange(Z_cvae_n.shape[0])

    # Latent 2D (PCA) colored by genre id and by cluster id
    Z2 = PCA(n_components=2, random_state=42).fit_transform(Z_cvae_n[idx])

    genre_ids = np.array([genre_to_id[g] for g in y_true[idx].tolist()], dtype=int)
    _plot_latent_2d(
        Z2,
        genre_ids,
        "Hard CVAE latent (PCA-2D) colored by genre",
        vis_dir / "latent_by_genre.png",
    )

    # Best k for CVAE latent by ARI
    cvae_rows = tmp[tmp["repr"] == "cvae_latent"].copy()
    best_cvae = cvae_rows.sort_values(by=["_ari"], ascending=[False]).iloc[0]
    best_k = int(best_cvae["k"])
    best_labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(Z_cvae_n)

    _plot_latent_2d(
        Z2,
        best_labels[idx],
        f"Hard CVAE latent (PCA-2D) colored by KMeans{best_k} clusters",
        vis_dir / "latent_by_cluster.png",
    )

    _plot_cluster_genre_distribution(best_labels, y_true, vis_dir / "cluster_genre_distribution.png")

    # ---- Reconstructions ----
    n_recon = int(args.recon_examples)
    if n_recon > 0:
        if Z_cvae_n.shape[0] >= n_recon:
            sel = rng.choice(Z_cvae_n.shape[0], size=n_recon, replace=False)
        else:
            sel = np.arange(Z_cvae_n.shape[0])
        recon_a, _recon_l = reconstruct_examples(model, X_audio, X_lyrics_s, C, sel)
        _save_recon_grid(X_audio[sel], recon_a, rec_dir / "spectrogram_recon_grid.png", n_show=n_recon)

    # ---- Save config + mappings ----
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "max_tracks": args.max_tracks,
                "spec_frames": args.spec_frames,
                "spec_mels": args.spec_mels,
                "lyrics_svd_dim": args.lyrics_svd_dim,
                "lyrics_max_features": args.lyrics_max_features,
                "latent_dim": args.latent_dim,
                "hidden_channels": args.hidden_channels,
                "lyrics_hidden_dim": args.lyrics_hidden_dim,
                "epochs": args.epochs,
                "beta": args.beta,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "audio_recon_weight": args.audio_recon_weight,
                "lyrics_recon_weight": args.lyrics_recon_weight,
                "k_list": k_list,
                "viz_max_points": args.viz_max_points,
                "recon_examples": args.recon_examples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (out_dir / "genre_mapping.json").write_text(json.dumps(genre_to_id, indent=2), encoding="utf-8")

    used_df.to_csv(out_dir / "used_tracks.csv", index=False)

    def _md_table(df_: pd.DataFrame, cols: list[str]) -> str:
        cols = [c for c in cols if c in df_.columns]
        if not cols or df_.empty:
            return ""

        def esc(v: object) -> str:
            s = "" if v is None else str(v)
            return s.replace("\n", " ")

        header = "| " + " | ".join(cols) + " |\n"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
        rows_md: list[str] = []
        for _, r in df_[cols].iterrows():
            rows_md.append("| " + " | ".join(esc(r[c]) for c in cols) + " |\n")
        return header + sep + "".join(rows_md)

    report = []
    report.append("# Hard results report\n\n")
    report.append("Hard model: Conditional VAE (CVAE) conditioned on genre.\n\n")
    report.append("Modalities: log-mel spectrogram (Conv2D) + lyrics embedding (TF-IDF+SVD).\n\n")
    report.append("Evaluation metrics: Silhouette, Davies–Bouldin, ARI, NMI, Purity.\n\n")
    report.append("## Best by ARI (per representation)\n\n")
    report.append(_md_table(best_df, ["repr", "algo", "k", "silhouette", "davies_bouldin", "ari", "nmi", "purity"]))
    report.append("\n\n## Notes\n")
    report.append("- CVAE uses genre as conditioning variable c during encoding/decoding.\n")
    report.append("- Language distribution plots are omitted (no reliable language labels).\n")

    (out_dir / "analysis_report.md").write_text("".join(report), encoding="utf-8")

    print("Saved:")
    for p in [
        out_dir / "clustering_metrics.csv",
        out_dir / "best_by_ari.csv",
        out_dir / "analysis_report.md",
        out_dir / "used_tracks.csv",
        vis_dir,
        rec_dir,
    ]:
        print(f"- {p}")

    if errors:
        print(f"Warning: {errors} audio files failed to decode/featurize and were skipped.")


if __name__ == "__main__":
    main()
