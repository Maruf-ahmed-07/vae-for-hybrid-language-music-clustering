from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_metadata(repo_root: str | Path) -> pd.DataFrame:
    repo_root = Path(repo_root)
    df = pd.read_csv(repo_root / "data" / "metadata_master.csv")

    required = {"uid", "lyrics_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata_master.csv missing columns: {sorted(missing)}")

    return df


def load_audio_paths(
    repo_root: str | Path,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Return metadata rows that have an existing `audio_path` file."""

    repo_root = Path(repo_root)
    df = load_metadata(repo_root)

    if "audio_path" not in df.columns:
        raise ValueError("metadata_master.csv missing column: audio_path")

    if max_rows is not None:
        df = df.head(int(max_rows)).copy()

    keep_rows: list[int] = []
    for idx, row in df.iterrows():
        rel_path = row["audio_path"]
        if not isinstance(rel_path, str) or not rel_path:
            continue
        p = repo_root / rel_path
        if not p.exists():
            continue
        keep_rows.append(idx)

    return df.loc[keep_rows].reset_index(drop=True)


def load_audio_lyrics_pairs(
    repo_root: str | Path,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Return metadata rows that have existing audio_path and lyrics_path files."""

    repo_root = Path(repo_root)
    df = load_metadata(repo_root)

    if "audio_path" not in df.columns:
        raise ValueError("metadata_master.csv missing column: audio_path")
    if "lyrics_path" not in df.columns:
        raise ValueError("metadata_master.csv missing column: lyrics_path")

    if max_rows is not None:
        df = df.head(int(max_rows)).copy()

    keep_rows: list[int] = []
    for idx, row in df.iterrows():
        a_rel = row["audio_path"]
        l_rel = row["lyrics_path"]
        if not isinstance(a_rel, str) or not a_rel:
            continue
        if not isinstance(l_rel, str) or not l_rel:
            continue

        a_p = repo_root / a_rel
        l_p = repo_root / l_rel
        if not a_p.exists() or not l_p.exists():
            continue
        keep_rows.append(idx)

    return df.loc[keep_rows].reset_index(drop=True)
