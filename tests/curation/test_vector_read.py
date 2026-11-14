"""The one vector reader: what it asks OpenSearch for, how it covers the documents, how it decodes."""

from __future__ import annotations

import base64
from typing import Any, cast

import numpy as np
import pytest

from src.services.curation.clustering import (
    page_rows,
    pool_snapshot,
    region_box_rows,
    residual_fetch,
)
from src.services.curation.clustering.pool_snapshot import snapshot_items
from tests.curation.vector_os_fake import VectorOS, random_docs


FIELD = 'pe_embedding'


@pytest.mark.asyncio
async def test_the_vector_is_asked_through_docvalue_fields_not_source() -> None:
    client = VectorOS(random_docs(30, seed=1))
    await snapshot_items(cast('Any', client), FIELD)
    assert client.searches
    for body in client.searches:
        assert body['docvalue_fields'] == [FIELD]
        assert body['_source'] is False


@pytest.mark.asyncio
async def test_slices_cover_every_document_exactly_once() -> None:
    docs = random_docs(53, seed=2)
    client = VectorOS(docs)
    ids, _ = await snapshot_items(cast('Any', client), FIELD)
    slices = {b['slice']['id'] for b in client.searches}
    assert slices == set(range(8))
    assert {b['slice']['max'] for b in client.searches} == {8}
    assert len(ids) == len(set(ids)) == sum(1 for _, f in docs if FIELD in f)
    assert client.pits_deleted == 1


@pytest.mark.asyncio
async def test_the_pool_filter_is_the_one_the_fit_always_used() -> None:
    client = VectorOS(random_docs(5, seed=3))
    await snapshot_items(cast('Any', client), FIELD, [{'term': {'extra': 1}}])
    query = client.searches[0]['query']['bool']
    assert query['filter'][1] == {'term': {'extra': 1}}
    assert {'term': {'class_validated': True}} in query['must_not']
    assert {'term': {'class_excluded': True}} in query['must_not']


def test_a_page_decodes_in_one_step_to_the_stored_float32() -> None:
    vectors = np.random.default_rng(0).standard_normal((4, 6)).astype(np.float32)
    hits = [
        {'_id': f'd{i}', 'fields': {FIELD: [base64.b64encode(v.astype('<f4').tobytes()).decode()]}}
        for i, v in enumerate(vectors)
    ]
    hits.insert(2, {'_id': 'none'})
    ids, matrix = page_rows.hits_to_rows(hits, FIELD)
    assert ids == ['d0', 'd1', 'd2', 'd3']
    assert matrix is not None
    assert matrix.dtype == np.float32
    assert np.array_equal(matrix, vectors)


def test_a_vector_that_is_not_base64_float32_is_refused() -> None:
    with pytest.raises(TypeError):
        page_rows.hits_to_rows([{'_id': 'a', 'fields': {FIELD: [[1.0, 2.0]]}}], FIELD)
    ragged = [
        {'_id': 'a', 'fields': {FIELD: [base64.b64encode(np.zeros(2, '<f4').tobytes()).decode()]}},
        {'_id': 'b', 'fields': {FIELD: [base64.b64encode(np.zeros(3, '<f4').tobytes()).decode()]}},
    ]
    with pytest.raises(ValueError, match='different'):
        page_rows.hits_to_rows(ragged, FIELD)


def test_the_superseded_readers_are_gone() -> None:
    assert not hasattr(residual_fetch, 'fetch_residual_embeddings_parallel')
    assert not hasattr(residual_fetch, '_fetch_one_slice')
    assert not hasattr(pool_snapshot, 'PAGE')
    assert not hasattr(region_box_rows, '_SCROLL_PAGE')
    assert not hasattr(page_rows, 'stack_pages')
