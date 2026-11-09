"""#372: the residual pool must not depend on the order the datastore returns documents in.

The refresh worker reads the pool through a sliced point-in-time search, so the row order is
the shard and ingest order. k-means picks its initial centroids and its training sample from
the rows by position, so the same documents in another order gave another partition (adjusted
Rand index 0.306 between two orders of the same 2,950 vectors). The fetch now returns the pool
sorted by document id, which makes every fit that follows a function of the document set.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pytest

from src.services.curation.clustering.residual_fetch import (
    fetch_residual_embeddings as _fetch_scroll,
    fetch_residual_embeddings_parallel as _fetch_parallel,
)


N_DOCS = 400
DIM = 16
FIELD = 'item_vec'
fetch_residual_embeddings = partial(_fetch_scroll, field=FIELD)
fetch_residual_embeddings_parallel = partial(_fetch_parallel, field=FIELD)


def _corpus() -> list[tuple[str, list[float]]]:
    """Documents ``crop-000`` .. with vectors drawn around eight centres (fixed seed)."""
    rng = np.random.default_rng(3)
    centres = rng.standard_normal((8, DIM))
    rows = centres[rng.integers(0, 8, N_DOCS)] + 0.6 * rng.standard_normal((N_DOCS, DIM))
    return [(f'crop-{i:03d}', row.tolist()) for i, row in enumerate(rows)]


class _Client:
    """A PIT- and scroll-capable fake whose documents sit in the given (shard) order."""

    def __init__(self, docs: list[tuple[str, list[float]]]) -> None:
        self.docs = docs

    def _hit(self, i: int) -> dict[str, Any]:
        doc_id, vec = self.docs[i]
        return {'_id': doc_id, '_source': {FIELD: vec}, 'sort': [i]}

    async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        return {'count': len(self.docs)}

    async def create_pit(self, *, index: str, keep_alive: str) -> dict[str, Any]:  # noqa: ARG002
        return {'pit_id': 'pit'}

    async def delete_pit(self, *, body: dict[str, Any]) -> None:  # noqa: ARG002
        return None

    async def search(
        self,
        *,
        body: dict[str, Any],
        index: str | None = None,  # noqa: ARG002
        scroll: str | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        if 'pit' not in body:  # the scroll variant
            return {
                '_scroll_id': 'sc',
                'hits': {'hits': [self._hit(i) for i in range(len(self.docs))]},
            }
        sl = body.get('slice')
        after = body.get('search_after')
        idx = [i for i in range(len(self.docs)) if sl is None or i % sl['max'] == sl['id']]
        if after is not None:
            idx = [i for i in idx if i > after[0]]
        return {'hits': {'hits': [self._hit(i) for i in idx[: body['size']]]}}

    async def scroll(self, *, scroll_id: str, scroll: str) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, *, scroll_id: str) -> None:  # noqa: ARG002
        return None


def _permuted(docs: list[tuple[str, list[float]]], seed: int) -> list[tuple[str, list[float]]]:
    order = np.random.default_rng(seed).permutation(len(docs))
    return [docs[i] for i in order]


@pytest.mark.asyncio
@pytest.mark.parametrize('fetch', [fetch_residual_embeddings, fetch_residual_embeddings_parallel])
async def test_fetch_returns_the_pool_sorted_by_document_id(fetch: Any) -> None:
    docs = _corpus()
    ids, matrix = await fetch(_Client(_permuted(docs, 1)))
    assert ids == sorted(d for d, _ in docs)
    by_id = dict(docs)
    expected = np.asarray([by_id[i] for i in ids], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(matrix, expected, atol=1e-6)


@pytest.mark.asyncio
@pytest.mark.parametrize('fetch', [fetch_residual_embeddings, fetch_residual_embeddings_parallel])
async def test_two_shard_orders_fetch_identical_pools(fetch: Any) -> None:
    docs = _corpus()
    ids_a, mat_a = await fetch(_Client(_permuted(docs, 1)))
    ids_b, mat_b = await fetch(_Client(_permuted(docs, 2)))
    assert ids_a == ids_b
    assert np.array_equal(mat_a, mat_b)
