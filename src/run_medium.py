from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, normalize
from tqdm import tqdm

from .audio_features import MFCCConfig, extract_logmel_sequence
from .clustering import KMeansConfig, pca_features, run_kmeans, tsne_2d
from .dataset import load_audio_lyrics_pairs
from .evaluation import clustering_metrics_safe
from .lyrics_features import LyricsEmbeddingConfig, compute_lyrics_embeddings, load_lyrics_text
from .multimodal_spec_vae import (
    MultiModalSpecVAEConfig,
    encode_multimodal_spec_latents,
    train_multimodal_spec_vae,
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


def _pick_best_subset(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("Cannot pick best from empty dataframe subset")

    tmp = df.copy()
    tmp["_sil"] = pd.to_numeric(tmp.get("silhouette"), errors="coerce")
    tmp["_db"] = pd.to_numeric(tmp.get("davies_bouldin"), errors="coerce")
    tmp["_sil"] = tmp["_sil"].replace([np.inf, -np.inf], np.nan).fillna(float("-inf"))
    tmp["_db"] = tmp["_db"].replace([np.inf, -np.inf], np.nan).fillna(float("inf"))
    best = tmp.sort_values(by=["_sil", "_db"], ascending=[False, True]).iloc[0].to_dict()
    best.pop("_sil", None)
    best.pop("_db", None)
    return best


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Medium task: CNN-VAE on log-mel spectrograms + lyrics embeddings -> shared latent -> clustering"
        )
    )
    parser.add_argument(
        "--max-tracks",
        type=_int_or_none,
        default=None,
        help="Max number of tracks to use (default: all). Use an integer or 'none'.",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="medium_spec_full_v1",
        help="Subfolder under results/medium/ to write outputs (default: medium_spec_full_v1).",
    )
    parser.add_argument(
        "--spec-frames",
        type=int,
        default=512,
        help="Fixed frame length for spectrogram inputs (default: 512). Must be divisible by 8.",
    )
    parser.add_argument(
        "--spec-mels",
        type=int,
        default=64,
        help="Number of mel bins for log-mel spectrograms (default: 64). Must be divisible by 8.",
    )
    parser.add_argument(
        "--lyrics-svd-dim",
        type=int,
        default=256,
        help="Lyrics SVD embedding dimension (default: 256).",
    )
    parser.add_argument(
        "--lyrics-max-features",
        type=int,
        default=20000,
        help="Lyrics TF-IDF max features (default: 20000).",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=16,
        help="Shared latent dimension (default: 16).",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=128,
        help="CNN base channels for spectrogram encoder/decoder (default: 128).",
    )
    parser.add_argument(
        "--lyrics-hidden-dim",
        type=int,
        default=256,
        help="Lyrics MLP hidden dim (default: 256).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Training epochs for multimodal CNN-VAE (default: 200).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.001,
        help="KL weight beta (default: 0.001).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--audio-recon-weight",
        type=float,
        default=1.0,
        help="Reconstruction weight for audio spectrogram (default: 1.0).",
    )
    parser.add_argument(
        "--lyrics-recon-weight",
        type=float,
        default=1.0,
        help="Reconstruction weight for lyrics embedding (default: 1.0).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run a small hyperparameter sweep for the CNN-VAE and select by silhouette.",
    )
    parser.add_argument(
        "--tune-sample",
        type=int,
        default=2000,
        help="Max tracks used during tuning/selection (default: 2000).",
    )
    parser.add_argument(
        "--k-list",
        type=_parse_int_list,
        default=_parse_int_list("6,8,10"),
        help="Comma-separated K values for KMeans/Agglo sweeps (default: 6,8,10).",
    )
    parser.add_argument(
        "--viz-method",
        choices=["pca", "tsne"],
        default="pca",
        help="2D visualization embedding method (default: pca).",
    )
    parser.add_argument(
        "--viz-max-points",
        type=int,
        default=2000,
        help="Max points to plot per visualization (default: 2000).",
    )

    args = parser.parse_args(argv)

    if int(args.spec_frames) <= 0 or int(args.spec_mels) <= 0:
        raise SystemExit("--spec-frames and --spec-mels must be positive integers")
    if (int(args.spec_frames) % 8) != 0 or (int(args.spec_mels) % 8) != 0:
        raise SystemExit("--spec-frames and --spec-mels must be divisible by 8")

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "results" / "medium" / args.out_subdir
    vis_dir = out_dir / "latent_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    df = load_audio_lyrics_pairs(repo_root, max_rows=args.max_tracks)
    if df.empty:
        raise RuntimeError("No rows found with BOTH audio and lyrics files.")

    y_true_all = df["genre_label"].astype(str).to_numpy() if "genre_label" in df.columns else None

    # ---- Extract spectrograms (log-mel) ----
    mfcc_cfg = MFCCConfig(duration_sec=30.0, n_mfcc=40)
    spec_list: list[np.ndarray] = []
    audio_stats: list[np.ndarray] = []
    keep_rows: list[int] = []
    errors = 0

    for i, row in tqdm(list(df.iterrows()), desc="Extracting log-mel spectrograms", total=len(df)):
        ap = repo_root / str(row["audio_path"])
        try:
            spec = extract_logmel_sequence(
                ap,
                mfcc_cfg,
                n_frames=int(args.spec_frames),
                n_mels=int(args.spec_mels),
            )  # (n_mels, n_frames)
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
    y_true = y_true_all[keep_rows] if y_true_all is not None else None

    X_spec = np.stack(spec_list, axis=0).astype(np.float32)  # (N, n_mels, n_frames)
    X_stats = np.stack(audio_stats, axis=0).astype(np.float32)  # (N, 2*n_mels)

    # CNN expects (N, 1, n_mels, n_frames)
    X_audio = X_spec[:, None, :, :]

    # Per-sample standardization for stability
    mean = X_audio.mean(axis=(2, 3), keepdims=True)
    std = X_audio.std(axis=(2, 3), keepdims=True) + 1e-6
    X_audio = (X_audio - mean) / std

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
    Z_lyrics = compute_lyrics_embeddings(texts, lyr_cfg)

    lyrics_scaler = StandardScaler()
    Z_lyrics_train = lyrics_scaler.fit_transform(Z_lyrics).astype(np.float32)

    # ---- Baselines (non-VAE) ----
    X_stats_n = StandardScaler().fit_transform(X_stats)
    X_stats_n = normalize(X_stats_n, norm="l2")
    X_base_audio_pca = normalize(pca_features(X_stats_n, n_components=32), norm="l2")

    X_base_lyrics = Z_lyrics

    X_base_hybrid = np.concatenate([X_stats_n, Z_lyrics], axis=1)
    X_base_hybrid = normalize(X_base_hybrid, norm="l2")
    X_base_hybrid_pca = normalize(pca_features(X_base_hybrid, n_components=32), norm="l2")

    # ---- Select/train multimodal CNN-VAE (spectrogram + lyrics) ----
    k_list = list(dict.fromkeys(int(k) for k in args.k_list))

    rng = np.random.default_rng(42)
    if args.tune and (args.tune_sample is not None) and (X_audio.shape[0] > int(args.tune_sample)):
        tune_idx = rng.choice(X_audio.shape[0], size=int(args.tune_sample), replace=False)
    else:
        tune_idx = np.arange(X_audio.shape[0])

    def build_cfg(*, latent_dim: int, hidden_channels: int, beta: float, epochs: int) -> MultiModalSpecVAEConfig:
        return MultiModalSpecVAEConfig(
            n_mels=int(args.spec_mels),
            n_frames=int(args.spec_frames),
            lyrics_dim=int(Z_lyrics_train.shape[1]),
            latent_dim=int(latent_dim),
            hidden_channels=int(hidden_channels),
            lyrics_hidden_dim=int(args.lyrics_hidden_dim),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            epochs=int(epochs),
            beta=float(beta),
            audio_recon_weight=float(args.audio_recon_weight),
            lyrics_recon_weight=float(args.lyrics_recon_weight),
            seed=42,
        )

    if args.tune:
        base = build_cfg(
            latent_dim=int(args.latent_dim),
            hidden_channels=int(args.hidden_channels),
            beta=float(args.beta),
            epochs=min(int(args.epochs), 60),
        )
        candidates = [
            base,
            replace(base, latent_dim=32),
            replace(base, latent_dim=64),
            replace(base, hidden_channels=max(64, int(args.hidden_channels) * 2), latent_dim=32),
            replace(base, beta=0.005),
            replace(base, beta=0.0005),
        ]
    else:
        candidates = [
            build_cfg(
                latent_dim=int(args.latent_dim),
                hidden_channels=int(args.hidden_channels),
                beta=float(args.beta),
                epochs=int(args.epochs),
            )
        ]

    select_rows: list[dict] = []
    best_cfg: MultiModalSpecVAEConfig | None = None
    best_score = float("-inf")

    for idx_cfg, cfg in enumerate(candidates, start=1):
        print(
            f"[mmSpecVAE] Training candidate {idx_cfg}/{len(candidates)}: "
            f"latent={cfg.latent_dim}, hidden={cfg.hidden_channels}, beta={cfg.beta}, epochs={cfg.epochs}"
        )
        model = train_multimodal_spec_vae(X_audio[tune_idx], Z_lyrics_train[tune_idx], cfg, verbose=True)
        Z = encode_multimodal_spec_latents(
            model,
            X_audio[tune_idx],
            Z_lyrics_train[tune_idx],
            batch_size=int(cfg.batch_size),
        )
        Z = normalize(Z, norm="l2")

        local_best = float("-inf")
        for k in k_list:
            labels = run_kmeans(Z, KMeansConfig(n_clusters=k))
            m = clustering_metrics_safe(Z, labels, labels_true=y_true[tune_idx] if y_true is not None else None)
            sil = float(m.get("silhouette", float("nan")))
            if np.isfinite(sil) and sil > local_best:
                local_best = sil
            select_rows.append(
                {
                    "task": "medium",
                    "stage": "select",
                    "repr": "hybrid_spec_mmvae",
                    "algo": f"kmeans{k}",
                    "k": k,
                    "latent_dim": cfg.latent_dim,
                    "hidden_channels": cfg.hidden_channels,
                    "beta": cfg.beta,
                    "epochs": cfg.epochs,
                    "batch_size": cfg.batch_size,
                    "lr": cfg.lr,
                    "audio_recon_weight": cfg.audio_recon_weight,
                    "lyrics_recon_weight": cfg.lyrics_recon_weight,
                    "spec_mels": cfg.n_mels,
                    "spec_frames": cfg.n_frames,
                    "lyrics_svd_dim": lyr_cfg.svd_dim,
                    "lyrics_max_features": lyr_cfg.max_features,
                    **m,
                }
            )

        if local_best > best_score:
            best_score = local_best
            best_cfg = cfg

    if best_cfg is None:
        raise RuntimeError("Failed to select a CNN-VAE configuration.")

    # Retrain best config for full epochs
    if args.tune:
        best_cfg = replace(best_cfg, epochs=int(args.epochs))

    print(
        f"[mmSpecVAE] Retraining best config: latent={best_cfg.latent_dim}, hidden={best_cfg.hidden_channels}, "
        f"beta={best_cfg.beta}, epochs={best_cfg.epochs}"
    )
    best_model = train_multimodal_spec_vae(X_audio, Z_lyrics_train, best_cfg, verbose=True)
    X_latent = encode_multimodal_spec_latents(
        best_model,
        X_audio,
        Z_lyrics_train,
        batch_size=int(best_cfg.batch_size),
    )
    X_latent = normalize(X_latent, norm="l2")

    # ---- Clustering suite ----
    feature_sets: list[tuple[str, np.ndarray]] = [
        ("hybrid_spec_mmvae", X_latent),
        ("audio_pca_baseline", X_base_audio_pca),
        ("lyrics_baseline", X_base_lyrics),
        ("hybrid_pca_baseline", X_base_hybrid_pca),
    ]

    rows: list[dict] = []

    def add_row(repr_name: str, algo: str, labels: np.ndarray, Xrepr: np.ndarray, extra: dict) -> None:
        m = clustering_metrics_safe(Xrepr, labels, labels_true=y_true)
        rows.append(
            {
                "task": "medium",
                "stage": "final",
                "repr": repr_name,
                "algo": algo,
                "k": extra.get("k", np.nan),
                "eps": extra.get("eps", np.nan),
                "min_samples": extra.get("min_samples", np.nan),
                "silhouette": m.get("silhouette", np.nan),
                "davies_bouldin": m.get("davies_bouldin", np.nan),
                "calinski_harabasz": m.get("calinski_harabasz", np.nan),
                "ari": m.get("ari", np.nan),
                "nmi": m.get("nmi", np.nan),
                "purity": m.get("purity", np.nan),
                "latent_dim": best_cfg.latent_dim if repr_name == "hybrid_spec_mmvae" else np.nan,
                "hidden_channels": best_cfg.hidden_channels if repr_name == "hybrid_spec_mmvae" else np.nan,
                "beta": best_cfg.beta if repr_name == "hybrid_spec_mmvae" else np.nan,
                "epochs": best_cfg.epochs if repr_name == "hybrid_spec_mmvae" else np.nan,
                "batch_size": best_cfg.batch_size if repr_name == "hybrid_spec_mmvae" else np.nan,
                "lr": best_cfg.lr if repr_name == "hybrid_spec_mmvae" else np.nan,
                "audio_recon_weight": best_cfg.audio_recon_weight if repr_name == "hybrid_spec_mmvae" else np.nan,
                "lyrics_recon_weight": best_cfg.lyrics_recon_weight if repr_name == "hybrid_spec_mmvae" else np.nan,
                "spec_mels": int(args.spec_mels),
                "spec_frames": int(args.spec_frames),
                "lyrics_svd_dim": lyr_cfg.svd_dim,
                "lyrics_max_features": lyr_cfg.max_features,
            }
        )

    # KMeans + Agglo sweeps
    for k in k_list:
        for repr_name, Xrepr in feature_sets:
            add_row(repr_name, f"kmeans{k}", run_kmeans(Xrepr, KMeansConfig(n_clusters=k)), Xrepr, {"k": k})
            labels_a = AgglomerativeClustering(n_clusters=k).fit_predict(Xrepr)
            add_row(repr_name, f"agglo{k}", labels_a, Xrepr, {"k": k})

    # DBSCAN grid
    eps_grid = [0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3]
    min_samples_grid = [5, 10, 20]
    for eps in eps_grid:
        for min_s in min_samples_grid:
            for repr_name, Xrepr in feature_sets:
                labels_d = DBSCAN(eps=float(eps), min_samples=int(min_s)).fit_predict(Xrepr)
                add_row(repr_name, "dbscan", labels_d, Xrepr, {"eps": eps, "min_samples": min_s})

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "clustering_metrics.csv", index=False)

    # Best per representation
    best_rows = []
    for repr_name, _ in feature_sets:
        sub = results[results["repr"] == repr_name]
        best_rows.append(_pick_best_subset(sub))
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(out_dir / "best_overall_by_repr.csv", index=False)

    # Comparison table
    compare_cols = [
        "repr",
        "algo",
        "k",
        "eps",
        "min_samples",
        "silhouette",
        "davies_bouldin",
        "ari",
        "latent_dim",
        "hidden_channels",
        "beta",
        "epochs",
        "batch_size",
        "lr",
        "audio_recon_weight",
        "lyrics_recon_weight",
        "spec_mels",
        "spec_frames",
        "lyrics_svd_dim",
        "lyrics_max_features",
    ]
    results.sort_values(by=["repr", "algo", "k"], inplace=True, kind="mergesort")
    results[compare_cols].to_csv(out_dir / "comparison_table.csv", index=False)

    # Save run config
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
                "tune": bool(args.tune),
                "tune_sample": args.tune_sample,
                "k_list": k_list,
                "viz_method": args.viz_method,
                "viz_max_points": args.viz_max_points,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.tune:
        pd.DataFrame(select_rows).to_csv(out_dir / "mmvae_selection.csv", index=False)

    # ---- Visualization (PCA-2D default, t-SNE optional) ----
    max_points = int(args.viz_max_points)
    if X_latent.shape[0] > max_points:
        idx = rng.choice(X_latent.shape[0], size=max_points, replace=False)
    else:
        idx = np.arange(X_latent.shape[0])

    cluster_count_rows: list[dict] = []

    for repr_name, Xrepr in feature_sets:
        sub = results[results["repr"] == repr_name]
        if sub.empty:
            continue

        families = {
            "kmeans": sub[sub["algo"].astype(str).str.startswith("kmeans")],
            "agglo": sub[sub["algo"].astype(str).str.startswith("agglo")],
            "dbscan": sub[sub["algo"].astype(str) == "dbscan"],
        }

        for family_name, fam_df in families.items():
            if fam_df.empty:
                continue

            best = _pick_best_subset(fam_df)
            algo = str(best["algo"])

            if algo.startswith("kmeans"):
                k = int(best.get("k", 8))
                labels = run_kmeans(Xrepr, KMeansConfig(n_clusters=k))
                label_desc = f"kmeans{k}"
            elif algo.startswith("agglo"):
                k = int(best.get("k", 8))
                labels = AgglomerativeClustering(n_clusters=k).fit_predict(Xrepr)
                label_desc = f"agglo{k}"
            else:
                labels = DBSCAN(
                    eps=float(best.get("eps", 0.5)),
                    min_samples=int(best.get("min_samples", 5)),
                ).fit_predict(Xrepr)
                label_desc = f"dbscan_eps{float(best.get('eps', 0.5))}_ms{int(best.get('min_samples', 5))}"

            uniq, cnt = np.unique(labels, return_counts=True)
            for cid, c in zip(uniq.tolist(), cnt.tolist()):
                cluster_count_rows.append(
                    {"repr": repr_name, "algo": label_desc, "cluster": int(cid), "count": int(c)}
                )

            X_vis = Xrepr[idx]
            if not np.all(np.isfinite(X_vis)) or X_vis.shape[0] < 5 or float(np.var(X_vis)) < 1e-12:
                continue

            try:
                if args.viz_method == "tsne":
                    Z2 = tsne_2d(X_vis)
                    method_label = "t-SNE"
                else:
                    Z2 = pca_features(X_vis, n_components=2)
                    method_label = "PCA-2D"
                if not np.all(np.isfinite(Z2)):
                    raise ValueError("non-finite 2D embedding")
            except Exception:
                continue

            plt.figure(figsize=(7, 5))
            plt.scatter(Z2[:, 0], Z2[:, 1], c=labels[idx], s=10, cmap="tab20")
            plt.title(f"Medium: {repr_name} labels={label_desc} ({method_label})")
            plt.tight_layout()
            plt.savefig(vis_dir / f"viz_{repr_name}_{family_name}.png", dpi=200)
            plt.close()

    if cluster_count_rows:
        pd.DataFrame(cluster_count_rows).sort_values(by=["repr", "algo", "count"], ascending=[True, True, False]).to_csv(
            out_dir / "cluster_counts.csv", index=False
        )

    # ---- Analysis report ----
    def _fmt(x: object) -> str:
        try:
            v = float(x)
        except Exception:
            return ""
        if np.isnan(v) or np.isinf(v):
            return ""
        return f"{v:.4f}"

    def _md_table(df_: pd.DataFrame, cols: list[str]) -> str:
        cols = [c for c in cols if c in df_.columns]
        if not cols:
            return ""

        def esc(s: object) -> str:
            v = "" if s is None else str(s)
            return v.replace("\n", " ")

        header = "| " + " | ".join(cols) + " |\n"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
        rows_md: list[str] = []
        for _, r in df_[cols].iterrows():
            rows_md.append("| " + " | ".join(esc(r[c]) for c in cols) + " |\n")
        return header + sep + "".join(rows_md)

    report_lines: list[str] = []
    report_lines.append("# Medium results report\n")
    report_lines.append("\nThis Medium run implements TWO enhancements:\n")
    report_lines.append("1) CNN-VAE on log-mel spectrograms (audio branch)\n")
    report_lines.append("2) Hybrid audio+lyrics via a shared latent representation\n")
    report_lines.append("\nClustering: KMeans, Agglomerative, DBSCAN.\n")

    if y_true is not None:
        report_lines.append("\nLabels were available, so ARI is included.\n")
    else:
        report_lines.append("\nNo labels available; ARI is omitted.\n")

    report_lines.append("\n## Best configuration per representation (by silhouette, tie-breaker: lowest DB)\n")
    show_cols = ["repr", "algo", "k", "eps", "min_samples", "silhouette", "davies_bouldin", "ari"]
    tbl = best_df[show_cols].copy()
    for col in ["silhouette", "davies_bouldin", "ari"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].map(_fmt)
    report_lines.append(_md_table(tbl, show_cols))

    report_lines.append("\n## Why VAE may be better/worse than baselines\n")
    report_lines.append(
        "- Better: the CNN encoder can learn non-linear time-frequency structure and denoise nuisance variation, making the latent more clusterable than simple PCA baselines.\n"
    )
    report_lines.append(
        "- Worse: VAE training optimizes reconstruction + KL, which may not align with cluster separation; the latent can be over-smoothed or collapse toward a small number of modes.\n"
    )
    report_lines.append(
        "- Hybrid-specific: if lyrics embeddings carry weak/orthogonal signal or are imbalanced vs audio, the shared latent can under-utilize lyrics unless weighting/architecture is tuned.\n"
    )

    (out_dir / "analysis_report.md").write_text("".join(report_lines), encoding="utf-8")

    used_df.to_csv(out_dir / "used_tracks.csv", index=False)

    print("Saved:")
    print(f"- {out_dir / 'clustering_metrics.csv'}")
    print(f"- {out_dir / 'best_overall_by_repr.csv'}")
    print(f"- {out_dir / 'comparison_table.csv'}")
    print(f"- {out_dir / 'cluster_counts.csv'}")
    print(f"- {out_dir / 'analysis_report.md'}")
    print(f"- {out_dir / 'used_tracks.csv'}")
    print(f"- {vis_dir}")
    if args.tune:
        print(f"- {out_dir / 'mmvae_selection.csv'}")
    if errors:
        print(f"Warning: {errors} audio files failed to decode/featurize and were skipped.")


if __name__ == "__main__":
    main()
