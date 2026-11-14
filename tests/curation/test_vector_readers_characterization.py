"""What each vector reader returns, pinned from the documents alone (not from any reader's code).

The expectations are computed here from the seeded float32 vectors, so the same tests hold for
the reader that asked for ``_source`` and for the one that asks for ``docvalue_fields``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from src.services.curation.clustering.outliers import compute_centroid_distances
from src.services.curation.clustering.pool_snapshot import snapshot_items
from src.services.curation.selection.pool_fetch import fetch_pool_embeddings
from src.services.near_duplicates.vectors import scan_items_of_images, scan_members
from tests.curation.vector_os_fake import VectorOS, random_docs


FIELD = 'pe_embedding'


async def _zero_count(**_kw: Any) -> dict[str, int]:
    return {'count': 0}


def _with_vector(docs: list[tuple[str, dict[str, Any]]]) -> dict[str, np.ndarray]:
    return {i: f[FIELD] for i, f in docs if FIELD in f}


@pytest.mark.asyncio
async def test_snapshot_items_is_sorted_by_id_with_the_exact_float32_vectors() -> None:
    docs = random_docs(53, seed=1)
    ids, matrix = await snapshot_items(cast('Any', VectorOS(docs)), FIELD)
    want = _with_vector(docs)
    assert ids == sorted(want)
    assert matrix.dtype == np.float32
    assert np.array_equal(matrix, np.stack([want[i] for i in ids]))


@pytest.mark.asyncio
async def test_snapshot_items_of_nothing_is_empty() -> None:
    ids, matrix = await snapshot_items(cast('Any', VectorOS([])), FIELD)
    assert ids == []
    assert matrix.shape == (0, 0)


@pytest.mark.asyncio
async def test_pool_fetch_returns_every_vector_and_truncates_past_the_cap() -> None:
    docs = random_docs(40, seed=2)
    want = _with_vector(docs)
    ids, matrix, truncated = await fetch_pool_embeddings(
        cast('Any', VectorOS(docs)), 'i', {'match_all': {}}, cap=1000, embedding_field=FIELD
    )
    assert not truncated
    assert sorted(ids) == sorted(want)
    by_id = dict(zip(ids, matrix, strict=True))
    assert all(np.array_equal(by_id[i], want[i]) for i in want)

    # the count says the pool fits, the read finds more: it stops with the first `cap` ids
    liar = VectorOS(docs)
    liar.count = _zero_count  # type: ignore[method-assign]
    ids, matrix, truncated = await fetch_pool_embeddings(
        cast('Any', liar), 'i', {'match_all': {}}, cap=5, embedding_field=FIELD
    )
    assert truncated
    assert len(ids) == 5
    assert matrix.shape == (0, 0)


@pytest.mark.asyncio
async def test_outlier_distances_match_the_centroid_distance_of_the_stored_vectors() -> None:
    docs = random_docs(30, seed=3)
    want = _with_vector(docs)
    got = await compute_centroid_distances(
        cast('Any', VectorOS(docs)),
        'i',
        {'match_all': {}},
        embedding_field=FIELD,
        current_count=None,
    )
    assert got is not None
    x = np.stack([want[i] for i in sorted(want)])
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    centroid = xn.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    expected = dict(zip(sorted(want), (1.0 - xn @ centroid).tolist(), strict=True))
    assert sorted(got) == sorted(expected)
    for doc_id, dist in expected.items():
        assert got[doc_id] == pytest.approx(dist, abs=1e-6)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else np.zeros_like(v)


@pytest.mark.asyncio
async def test_scan_members_are_unit_rows_with_item_fields() -> None:
    docs = random_docs(25, seed=4)
    want = _with_vector(docs)
    seen = {}
    async for page in scan_members(cast('Any', VectorOS(docs)), index='i', field=FIELD, item=True):
        seen.update({m.member_id: m for m in page})
    assert sorted(seen) == sorted(want)
    for doc_id, vec in want.items():
        assert np.allclose(seen[doc_id].vector, _unit(vec), atol=1e-7)
        assert seen[doc_id].vector.dtype == np.float32
        assert seen[doc_id].image_id is not None
        assert seen[doc_id].detector_class == 'car'


@pytest.mark.asyncio
async def test_scan_items_of_images_returns_the_members_of_those_images() -> None:
    docs = random_docs(25, seed=5)
    want = _with_vector(docs)
    found = await scan_items_of_images(
        cast('Any', VectorOS(docs)), index='i', field=FIELD, image_ids=['img-0']
    )
    assert sorted(m.member_id for m in found) == sorted(want)
