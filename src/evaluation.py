from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def clustering_metrics(
    X: np.ndarray,
    labels_pred: np.ndarray,
    labels_true: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute clustering metrics.

    - Always computes intrinsic metrics: Silhouette, CH, DB.
    - Computes ARI/NMI/Purity only if `labels_true` is provided.
    """

    metrics: dict[str, float] = {}

    metrics["silhouette"] = float(silhouette_score(X, labels_pred))
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels_pred))
    metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels_pred))

    if labels_true is not None:
        metrics["ari"] = float(adjusted_rand_score(labels_true, labels_pred))
        metrics["nmi"] = float(normalized_mutual_info_score(labels_true, labels_pred))
        metrics["purity"] = float(cluster_purity(labels_true, labels_pred))

    return metrics


def clustering_metrics_safe(
    X: np.ndarray,
    labels_pred: np.ndarray,
    labels_true: np.ndarray | None = None,
) -> dict[str, float]:
    """Like clustering_metrics, but returns NaNs when metrics are undefined.

    Some clusterers (notably DBSCAN) may return:
    - a single cluster, or
    - all points as noise (-1), or
    - clusters of size 1

    In those cases, Silhouette/CH/DB can be undefined or error.
    """

    labels_pred = np.asarray(labels_pred)
    unique = np.unique(labels_pred)

    # If all points in one cluster, intrinsic metrics are not defined.
    if unique.size < 2:
        out: dict[str, float] = {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
        }
        if labels_true is not None:
            out["ari"] = float(adjusted_rand_score(labels_true, labels_pred))
            out["nmi"] = float(normalized_mutual_info_score(labels_true, labels_pred))
            out["purity"] = float(cluster_purity(labels_true, labels_pred))
        return out

    try:
        return clustering_metrics(X, labels_pred, labels_true=labels_true)
    except Exception:
        out = {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
        }
        if labels_true is not None:
            out["ari"] = float(adjusted_rand_score(labels_true, labels_pred))
            out["nmi"] = float(normalized_mutual_info_score(labels_true, labels_pred))
            out["purity"] = float(cluster_purity(labels_true, labels_pred))
        return out


def cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("labels_true and labels_pred must have same length")

    total = labels_true.shape[0]
    purity_sum = 0

    for cluster_id in np.unique(labels_pred):
        idx = labels_pred == cluster_id
        true_labels_in_cluster = labels_true[idx]
        if true_labels_in_cluster.size == 0:
            continue

        values, counts = np.unique(true_labels_in_cluster, return_counts=True)
        purity_sum += int(counts.max())

    return purity_sum / float(total)
