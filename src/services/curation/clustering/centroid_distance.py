"""Cosine distance of each item to its cluster's member-mean centroid.

The fallback for a clustering result without centroid distances (IVF's
small-pool single bucket; label-only AHC/HDBSCAN), so every run writes a
``cluster_distance`` the outlier sorts and the core-member cut line can
use. Same semantics as IVF's own distances: ``1 - cos(x, c)`` with ``c``
the unit-normalized centroid, in ``[0, 2]``.
"""

from __future__ import annotations

import numpy as np


def member_centroid_distances(embeddings: np.ndarray, labels: np.ndarray) -> list[float | None]:
    """Per-row distance to the mean of its label's rows; ``None`` for noise
    (label ``-1``)."""
    x = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.where(norms == 0, 1.0, norms)
    labels = np.asarray(labels, dtype=np.int64)
    out: list[float | None] = [None] * len(labels)
    for label in np.unique(labels):
        if label < 0:
            continue
        rows = np.flatnonzero(labels == label)
        centroid = x[rows].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0.0:
            continue
        dists = np.clip(1.0 - x[rows] @ (centroid / norm), 0.0, 2.0)
        for row, dist in zip(rows.tolist(), dists.tolist(), strict=True):
            out[row] = float(dist)
    return out


__all__ = ['member_centroid_distances']
