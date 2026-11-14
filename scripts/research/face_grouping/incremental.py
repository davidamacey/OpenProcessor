"""Add faces in batches: nearest centroid within a threshold, the leftovers clustered among themselves."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def incremental_steps(
    x: np.ndarray,
    batches: list[np.ndarray],
    method: Callable[[np.ndarray], np.ndarray],
    join_sim: float,
) -> Iterator[np.ndarray]:
    """Yield the label array (-1 = not yet added or unassigned) after each batch."""
    label = np.full(len(x), -1, dtype=np.int64)
    sums: dict[int, np.ndarray] = {}
    next_id = 0

    def adopt(rows: np.ndarray, local: np.ndarray) -> None:
        nonlocal next_id
        for g in np.unique(local[local >= 0]):
            members = rows[local == g]
            label[members] = next_id
            sums[next_id] = x[members].sum(axis=0)
            next_id += 1

    for step, rows in enumerate(batches):
        left = rows
        if step and sums:
            ids = np.array(list(sums))
            cent = np.stack([sums[i] for i in ids])
            cent /= np.linalg.norm(cent, axis=1, keepdims=True)
            sim = x[rows] @ cent.T
            best = sim.argmax(axis=1)
            joined = sim[np.arange(len(rows)), best] >= join_sim
            label[rows[joined]] = ids[best[joined]]
            left = rows[~joined]
        if len(left):
            adopt(left, method(x[left]))
        yield label.copy()


def incremental(
    x: np.ndarray,
    batches: list[np.ndarray],
    method: Callable[[np.ndarray], np.ndarray],
    join_sim: float,
) -> np.ndarray:
    *_, last = incremental_steps(x, batches, method, join_sim)
    return last
