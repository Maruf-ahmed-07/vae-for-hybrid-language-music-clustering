
# Data (how the dataset was built)

This project uses a *hybrid* dataset: **audio + lyrics + metadata**.

Because audio and lyrics files are large, the GitHub repo only includes the **metadata CSVs** needed to reproduce the pipelines (assuming you have the raw files locally).

## Sources used

1) **FMA (Free Music Archive)**
- Used as the main source of audio files + track metadata (artist/title/track IDs).

2) **Kaggle “5 Million Lyrics” dataset**
- Used as the large pool of lyrics text.

3) **Merged / master metadata**
- We created a merged table by combining the two sources and matching tracks so each row points to:
	- an audio file (from FMA)
	- a lyrics file (from Kaggle lyrics, saved locally as text)
	- a genre label (from metadata)

## Matching / merging approach (high level)

Tracks were matched across datasets using **artist + title**.

Typical steps:
- Normalize strings (lowercase, strip punctuation, collapse whitespace)
- Handle common formatting issues (e.g., “feat.” / “ft.”, extra parentheses, remix markers)
- Join or fuzzy-match on normalized `(artist, title)`
- Keep only matches that pass matching rules (to avoid incorrect lyrics assignments)

The output of this process is saved as the master metadata file:
- `metadata_master.csv`

## Files in this folder

- `metadata_master.csv`
	- The main index used by the training scripts.
	- Contains (at minimum): `audio_path`, `lyrics_path`, `genre_label`.
- `audio_metadata.csv`
	- Audio-side metadata extracted from FMA.
- `lyrics_metadata.csv`
	- Lyrics-side metadata extracted from the Kaggle lyrics dataset.

## Raw data (not pushed to GitHub)

These folders are expected to exist locally, but are excluded from GitHub via `.gitignore`:
- `data/audio/` (audio files)
- `data/lyrics/` (lyrics text files)

If you clone the repo from GitHub, you need to place your local audio and lyrics files in these folders (or update paths in `metadata_master.csv`).

