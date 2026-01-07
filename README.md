# VAE Music Clustering (Audio + Lyrics)

This is my Neural Networks course project: learn latent features with VAE-style models and then cluster music tracks.

## Dataset + report links
- Dataset folder (audio/lyrics + metadata): https://drive.google.com/drive/folders/1wqAJRniPUHkouTTBVkcoPgaqXAwrt2s3?usp=sharing
- Project report (PDF): https://drive.google.com/file/d/1uWtDcjoa_txpwcqO1ZD-bBRDAwhvS4US/view?usp=sharing

Note: the GitHub repo does not include raw audio/lyrics files; see `data/README.md` for how the dataset is structured.

## Repository layout
- `data/`: metadata CSVs + pointers to audio/lyrics files
- `notebooks/`: quick exploration / sanity checks
- `src/`: all pipelines + models
- `results/`: outputs (metrics + plots)

Quick summaries:
- `results/clustering_metrics.csv`: 1-row-per-task summary from the final runs
- `results/latent_visualization/`: a few representative final figures

## Setup
1. Create/activate a Python environment.
2. Install deps:
   - `pip install -r requirements.txt`

## Easy (audio-only)
Audio MFCC stats → (PCA baseline vs MLP-VAE latent) → KMeans → t-SNE.

Run:
- `python -m src.run_easy`

Notes:
- This is intentionally **audio-only** (no lyrics).
- The script does a small VAE sweep and picks the best config by silhouette (fully unsupervised).

Outputs are written to `results/easy/audio_full/`.

## Medium (CNN-VAE on spectrograms + lyrics)
Log-mel spectrograms (Conv2D) + lyrics embeddings (TF-IDF+SVD) → shared latent → clustering (KMeans / Agglo / DBSCAN).

Run:
- `python -m src.run_medium`

Outputs are written to `results/medium/medium_spec_fast_v1/`.

## Hard (CVAE conditioned on genre)
Audio + lyrics + **genre conditioning** (CVAE) → latent → KMeans.
Also compares against PCA+KMeans, AE+KMeans, and direct spectral clustering.

Run:
- `python -m src.run_hard`

Outputs are written to `results/hard/hard_full_v1/`.

## Notebook
If you just want to poke around (no full training runs):
- `notebooks/exploratory.ipynb`
