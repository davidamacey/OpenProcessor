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

from src.services.curation.clustering.residual_fetch import fetch_residual_embeddings as _fetch
from tests.curation.vector_os_fake import VectorOS


N_DOCS = 400
DIM = 16
FIELD = 'item_vec'
fetch_residual_embeddings = partial(_fetch, field=FIELD)


def _corpus() -> list[tuple[str, list[float]]]:
    """Documents ``crop-000`` .. with vectors drawn around eight centres (fixed seed)."""
    rng = np.random.default_rng(3)
    centres = rng.standard_normal((8, DIM))
    rows = centres[rng.integers(0, 8, N_DOCS)] + 0.6 * rng.standard_normal((N_DOCS, DIM))
    return [(f'crop-{i:03d}', row.tolist()) for i, row in enumerate(rows)]


def _client(docs: list[tuple[str, list[float]]]) -> Any:
    return VectorOS([(i, {FIELD: np.asarray(v, dtype=np.float32)}) for i, v in docs], FIELD)


def _permuted(docs: list[tuple[str, list[float]]], seed: int) -> list[tuple[str, list[float]]]:
    order = np.random.default_rng(seed).permutation(len(docs))
    return [docs[i] for i in order]


@pytest.mark.asyncio
async def test_fetch_returns_the_pool_sorted_by_document_id() -> None:
    docs = _corpus()
    ids, matrix = await fetch_residual_embeddings(_client(_permuted(docs, 1)))
    assert ids == sorted(d for d, _ in docs)
    by_id = dict(docs)
    expected = np.asarray([by_id[i] for i in ids], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(matrix, expected, atol=1e-6)


@pytest.mark.asyncio
async def test_two_shard_orders_fetch_identical_pools() -> None:
    docs = _corpus()
    ids_a, mat_a = await fetch_residual_embeddings(_client(_permuted(docs, 1)))
    ids_b, mat_b = await fetch_residual_embeddings(_client(_permuted(docs, 2)))
    assert ids_a == ids_b
    assert np.array_equal(mat_a, mat_b)
