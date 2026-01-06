
# Data (how the dataset was built)

This project uses a *hybrid* dataset: **audio + lyrics + metadata**.

Because audio and lyrics files are large, the GitHub repo only includes the **metadata CSVs** needed to reproduce the pipelines (assuming you have the raw files locally).

The three CSV files in this folder (`audio_metadata.csv`, `lyrics_metadata.csv`, `metadata_master.csv`) represent the **final combined dataset after merging MERGE + FMA**. Since MERGE and FMA/Kaggle did not share the same schema, the metadata fields were **aligned into a common structure**, and lyrics were materialized from a lyrics CSV into per-track `.txt` files before producing the final merged tables.

## Sources used (and how many tracks came from each)

This project’s final dataset is the combination of two subsets:

1) **MERGE dataset (~2100+ tracks)**
- MERGE already contains **both audio tracks and lyrics** for these songs.
- We included ~2100+ tracks directly from MERGE.

2) **FMA + Kaggle “5 Million Lyrics” (~900+ matched tracks)**
- We started from a larger pool of **~25,000 FMA tracks**.
- We matched those FMA tracks to entries in the Kaggle **5 Million Lyrics** dataset.
- Only **~900+ tracks** had reliable matches (so they were included).

3) **Master merged metadata**
- We built `metadata_master.csv` by concatenating:
	- MERGE rows (audio+lyrics already available), and
	- FMA+Kaggle matched rows (audio from FMA + lyrics from Kaggle)

Each final row points to:
- an audio file path (local)
- a lyrics text path (local)
- a genre label (from metadata)

## Matching / merging approach (high level)

For the **FMA + Kaggle** subset, tracks were matched using **artist + title**.

Typical steps:
- Normalize strings (lowercase, strip punctuation, collapse whitespace)
- Handle common formatting issues (e.g., “feat.” / “ft.”, extra parentheses, remix markers)
- Join or fuzzy-match on normalized `(artist, title)`
- Keep only matches that pass matching rules (to avoid incorrect lyrics assignments)

The output of this process is saved as the master metadata file:
- `metadata_master.csv`

## Standardizing structure (what was changed)

Because the source datasets used different column names and formats, we first standardized them into a shared schema (consistent identifiers and fields such as artist/title, local `audio_path`, local `lyrics_path`, and any usable labels like `genre_label`).

For lyrics specifically (Kaggle lyrics), lyrics were originally stored in a CSV; they were exported into individual text files under `data/lyrics/` (one file per track, e.g. `fma_000010.txt`) so the pipelines can load lyrics consistently from disk.

## Source links (official pages)

- MERGE audio dataset (Zenodo):
	- https://zenodo.org/records/13939205
- Kaggle “5 Million Song Lyrics Dataset”:
	- https://www.kaggle.com/datasets/nikhilnayak123/5-million-song-lyrics-dataset
- Kaggle FMA (Free Music Archive) small/medium:
	- https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium

## Files in this folder

- `metadata_master.csv`
	- The main index used by the training scripts.
	- Contains (at minimum): `audio_path`, `lyrics_path`, `genre_label`.
- `audio_metadata.csv`
	- Audio metadata table in the **standardized (post-merge) structure**.
	- Contains rows from both MERGE and FMA after alignment.
- `lyrics_metadata.csv`
	- Lyrics metadata table in the **standardized (post-merge) structure**.
	- Contains rows from both MERGE and FMA/Kaggle after alignment.

## Raw data (not pushed to GitHub)

These folders are expected to exist locally, but are excluded from GitHub via `.gitignore`:
- `data/audio/` (audio files)
- `data/lyrics/` (lyrics text files)

If you want to download the prepared dataset files directly, use this folder:
- https://drive.google.com/drive/folders/1wqAJRniPUHkouTTBVkcoPgaqXAwrt2s3?usp=sharing

If you clone the repo from GitHub, you need to place your local audio and lyrics files in these folders (or update paths in `metadata_master.csv`).

