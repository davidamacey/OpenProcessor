"""Cheap per-face quality signals, the reject curve, and clustering with a quality gate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable


def pose_offset(landmarks: np.ndarray) -> np.ndarray:
    """Yaw proxy from the 5 landmarks (eye, eye, nose, mouth, mouth): nose offset over eye distance."""
    left, right, nose = landmarks[:, 0:2], landmarks[:, 2:4], landmarks[:, 4:6]
    eye_dist = np.hypot(*(right - left).T)
    offset: np.ndarray = np.abs(nose[:, 0] - (left[:, 0] + right[:, 0]) / 2) / np.maximum(
        eye_dist, 1e-6
    )
    return offset


def error_reject_area(quality: np.ndarray, error: np.ndarray) -> float:
    """Mean error rate of what remains as the lowest-quality items are rejected one by one.

    Lower is better; the area of a signal unrelated to the errors equals the overall error rate.
    """
    err = error[np.argsort(quality, kind='stable')].astype(np.float64)
    n = len(err)
    remaining = err.sum() - np.concatenate([[0.0], np.cumsum(err)[:-1]])
    return float((remaining / (n - np.arange(n))).mean())


def gated(
    x: np.ndarray,
    quality: np.ndarray,
    cut: float,
    method: Callable[[np.ndarray], np.ndarray],
    join_sim: float,
) -> np.ndarray:
    """Cluster only faces at or above ``cut``; weaker faces join the nearest centroid at ``join_sim`` or stay -1.

    A weak face never founds a cluster and never links two, because the structure is fixed before it joins.
    """
    strong = quality >= cut
    label = np.full(len(x), -1, dtype=np.int64)
    label[strong] = method(x[strong])
    ids = np.unique(label[label >= 0])
    if len(ids) and (~strong).any():
        cent = np.stack([x[label == i].sum(axis=0) for i in ids])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        sim = x[~strong] @ cent.T
        best = sim.argmax(axis=1)
        joined = sim[np.arange(len(best)), best] >= join_sim
        weak = np.flatnonzero(~strong)
        label[weak[joined]] = ids[best[joined]]
    return label
