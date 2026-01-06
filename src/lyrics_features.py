from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LyricsEmbeddingConfig:
    max_features: int = 20000
    svd_dim: int = 256
    min_df: int = 2
    ngram_range: tuple[int, int] = (2, 5)


def load_lyrics_text(lyrics_path: str | Path) -> str:
    p = Path(lyrics_path)
    # Be tolerant of encoding issues (lyrics may contain mixed scripts)
    return p.read_text(encoding="utf-8", errors="ignore")


def compute_lyrics_embeddings(texts: list[str], cfg: LyricsEmbeddingConfig) -> np.ndarray:
    """Compute dense lyrics embeddings.

    Uses character n-gram TF-IDF (language-agnostic) then TruncatedSVD to a dense vector.
    Returns shape (N, svd_dim).
    """

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        min_df=cfg.min_df,
    )
    X = vectorizer.fit_transform(texts)

    # TruncatedSVD works on sparse matrices
    svd = TruncatedSVD(n_components=cfg.svd_dim, random_state=42)
    Z = svd.fit_transform(X).astype(np.float32)

    # L2 normalize for cosine-ish geometry
    Z = normalize(Z, norm="l2")
    return Z
