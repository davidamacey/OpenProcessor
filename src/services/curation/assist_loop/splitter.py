"""Split a node of the loop's partition tree into children.

Within :data:`~src.services.curation.clustering.refine.MAX_REFINE_MEMBERS` this is the
refine core (complete-linkage cosine AHC cut into exactly ``branching`` groups) with a
writer that only captures the labels; above it, whose pairwise matrix would not fit, a
seeded k-means on L2-normalized vectors. Neither writes to the datastore: the tree lives
in the job directory. Members are fitted in sorted id order, so the result depends on the
ids, the vectors, ``seed`` and ``path`` only, never on input order.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.clients.curation_opensearch.crops import mget_crops
from src.domain.loop_tree import seed_int
from src.services.curation.clustering.refine import (
    MAX_REFINE_MEMBERS,
    MIN_REFINE_MEMBERS,
    refine_members,
)
from src.services.curation.clustering.residual_fetch import RESIDUAL_EMBEDDING_FIELD


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from opensearchpy import AsyncOpenSearch

KMEANS_N_INIT = 4


@dataclass(frozen=True)
class SplitResult:
    children: list[list[str]]
    """Sorted member ids per child, ordered by each child's smallest id (this fixes the
    child paths). Empty when the node cannot be split into at least two children."""
    unplaced: list[str]
    """Members with no embedding: in no child."""
    method: Literal['ahc', 'kmeans']


async def fetch_embeddings(opensearch: AsyncOpenSearch, ids: Sequence[str]) -> dict[str, Any]:
    """``{crop_id: embedding}`` for the ids that have one."""
    docs = await mget_crops(opensearch, list(ids), source_includes=[RESIDUAL_EMBEDDING_FIELD])
    out: dict[str, Any] = {}
    for crop_id, doc in docs.items():
        vector = (doc.get('_source') or {}).get(RESIDUAL_EMBEDDING_FIELD)
        if vector is not None:
            out[crop_id] = vector
    return out


def _kmeans_labels(matrix: np.ndarray, *, k: int, random_state: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.where(norms == 0, 1.0, norms)
    fitted = KMeans(n_clusters=k, random_state=random_state, n_init=KMEANS_N_INIT).fit_predict(unit)
    return np.asarray(fitted)


async def _kmeans_groups(
    placed: list[str], vectors: list[Any], *, branching: int, seed: int, path: str
) -> dict[Any, list[str]]:
    matrix = np.asarray(vectors, dtype=np.float32)
    distinct = len(np.unique(matrix, axis=0))
    if distinct < 2:
        return {}
    labels = await asyncio.to_thread(
        _kmeans_labels, matrix, k=min(branching, distinct), random_state=seed_int(seed, path)
    )
    groups: dict[Any, list[str]] = {}
    for crop_id, label in zip(placed, labels, strict=True):
        groups.setdefault(int(label), []).append(crop_id)
    return groups


async def _ahc_groups(
    placed: list[str], vectors: list[Any], *, branching: int, max_members: int
) -> dict[Any, list[str]]:
    members = [
        {'_id': crop_id, 'embedding': vector, 'class_name': None}
        for crop_id, vector in zip(placed, vectors, strict=True)
    ]
    captured: list[tuple[Any, str]] = []

    async def count() -> int:
        return len(members)

    async def fetch() -> list[dict[str, Any]]:
        return members

    async def capture(pairs: list[tuple[Any, str]]) -> int:
        captured.extend(pairs)
        return len(pairs)

    summary = await refine_members(
        0,
        count_members=count,
        fetch_members=fetch,
        write_subids=capture,
        unit='items',
        max_members=max_members,
        n_clusters=branching,
    )
    if summary['action'] != 'refined':
        return {}
    groups: dict[Any, list[str]] = {}
    for crop_id, subid in captured:
        groups.setdefault(subid, []).append(crop_id)
    return groups


async def split_node(
    ids: Sequence[str],
    embeddings: Mapping[str, Any],
    *,
    branching: int,
    seed: int,
    path: str,
    max_members: int = MAX_REFINE_MEMBERS,
) -> SplitResult:
    """Split ``ids`` into up to ``branching`` children by embedding similarity."""
    unique = sorted(set(ids))
    placed = [crop_id for crop_id in unique if crop_id in embeddings]
    unplaced = [crop_id for crop_id in unique if crop_id not in embeddings]
    vectors = [embeddings[crop_id] for crop_id in placed]
    method: Literal['ahc', 'kmeans'] = 'kmeans' if len(placed) > max_members else 'ahc'
    groups: dict[Any, list[str]] = {}
    if len(placed) >= max(MIN_REFINE_MEMBERS, branching):
        if method == 'kmeans':
            groups = await _kmeans_groups(
                placed, vectors, branching=branching, seed=seed, path=path
            )
        else:
            groups = await _ahc_groups(
                placed, vectors, branching=branching, max_members=max_members
            )
    children = sorted((sorted(group) for group in groups.values() if group), key=lambda c: c[0])
    return SplitResult(
        children=children if len(children) >= 2 else [], unplaced=unplaced, method=method
    )


__all__ = ['SplitResult', 'fetch_embeddings', 'split_node']
