from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass(frozen=True)
class KMeansConfig:
    n_clusters: int = 10
    seed: int = 42


def run_kmeans(X: np.ndarray, cfg: KMeansConfig) -> np.ndarray:
    km = KMeans(n_clusters=cfg.n_clusters, n_init="auto", random_state=cfg.seed)
    return km.fit_predict(X)


def pca_features(X: np.ndarray, n_components: int = 16, seed: int = 42) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(X)


def tsne_2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=seed)
    return tsne.fit_transform(X)
