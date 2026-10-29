"""#372: the residual pool must not depend on the order the datastore returns documents in.

The refresh worker reads the pool through a sliced point-in-time search, so the row order is
the shard and ingest order. k-means picks its initial centroids and its training sample from
the rows by position, so the same documents in another order gave another partition (adjusted
Rand index 0.306 between two orders of the same 2,950 vectors). The fetch now returns the pool
sorted by document id, which makes every fit that follows a function of the document set.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest import mock

import numpy as np
import pytest

from scripts.bench.parity_metrics import adjusted_rand_index
from src.services.curation.clustering.methods import ivf_store
from src.services.curation.clustering.methods.ivf import IVFMethod
from src.services.curation.clustering.residual_fetch import (
    RESIDUAL_EMBEDDING_FIELD,
    fetch_residual_embeddings,
    fetch_residual_embeddings_parallel,
)


if TYPE_CHECKING:
    from pathlib import Path

N_DOCS = 400
DIM = 16


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
        return {'_id': doc_id, '_source': {RESIDUAL_EMBEDDING_FIELD: vec}, 'sort': [i]}

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


async def _ivf_labels_by_id(ids: list[str], matrix: np.ndarray, state: Path) -> dict[str, int]:
    cfg = SimpleNamespace(project_state_dir=state)
    with mock.patch.object(ivf_store, 'get_curation_config', return_value=cfg):
        result = await IVFMethod().fit_predict(
            matrix,
            backend_info=SimpleNamespace(name='cpu'),  # type: ignore[arg-type]
        )
    return dict(zip(ids, result.labels.tolist(), strict=True))


@pytest.mark.asyncio
async def test_ivf_partition_is_the_same_for_any_shard_order(tmp_path: Path) -> None:
    docs = _corpus()
    fetched = [
        await fetch_residual_embeddings_parallel(_Client(_permuted(docs, seed)))
        for seed in (11, 12)
    ]
    # the fixture would not catch the bug if k-means did not care about row order
    unsorted_a = np.asarray([v for _, v in _permuted(docs, 11)], dtype=np.float32)
    unsorted_b = np.asarray([v for _, v in _permuted(docs, 12)], dtype=np.float32)
    raw_a = await _ivf_labels_by_id(
        [d for d, _ in _permuted(docs, 11)], unsorted_a, tmp_path / 'ra'
    )
    raw_b = await _ivf_labels_by_id(
        [d for d, _ in _permuted(docs, 12)], unsorted_b, tmp_path / 'rb'
    )
    keys = sorted(raw_a)
    sensitive = adjusted_rand_index(
        np.array([raw_a[k] for k in keys]), np.array([raw_b[k] for k in keys])
    )
    assert sensitive < 0.95, 'k-means no longer depends on row order; this test proves nothing'

    parts = [
        await _ivf_labels_by_id(ids, mat, tmp_path / f'f{n}')
        for n, (ids, mat) in enumerate(fetched)
    ]
    assert parts[0] == parts[1]
