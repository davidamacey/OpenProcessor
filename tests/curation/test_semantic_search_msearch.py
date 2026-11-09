"""Semantic search sends the kNN page and the unembedded count as one ``_msearch`` (ES-C4).

OpenSearch multi-search: https://docs.opensearch.org/latest/api-reference/multi-search/
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.schemas.curation.search_target import SearchTarget
from src.services.curation import semantic_search


def _encoder() -> MagicMock:
    enc = MagicMock()
    enc.text_model = 'pe_text_encoder'
    enc.encode_text = MagicMock(return_value=np.ones((1, 4), dtype=np.float32))
    return enc


def _opensearch(hit_count: int = 0, unembedded: int = 7) -> AsyncMock:
    """A client that only answers ``msearch``: a call to ``search`` or ``count`` is a failure."""
    fake = AsyncMock()
    knn = {
        'took': 3,
        'hits': {
            'hits': [{'_id': f'c{i}', '_score': 0.9, '_source': {}} for i in range(hit_count)],
            'total': {'value': hit_count},
        },
    }
    count = {'took': 1, 'hits': {'hits': [], 'total': {'value': unembedded}}}
    fake.msearch = AsyncMock(return_value={'responses': [knn, count]})
    fake.search = AsyncMock(side_effect=AssertionError('search must go through msearch'))
    fake.count = AsyncMock(side_effect=AssertionError('count must go through msearch'))
    return fake


async def _text(fake: AsyncMock, **kwargs) -> dict:
    return await semantic_search.semantic_text_search(
        opensearch=fake,
        pe_encoder=_encoder(),
        executor=None,
        query='q',
        page=1,
        page_size=10,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_the_knn_page_and_the_count_are_one_client_call() -> None:
    fake = _opensearch(hit_count=2, unembedded=7)
    result = await _text(fake)
    assert fake.msearch.await_count == 1
    assert (fake.search.await_count, fake.count.await_count) == (0, 0)
    assert result['unembedded_in_scope'] == 7
    assert [i['crop_id'] for i in result['items']] == ['c0', 'c1']


@pytest.mark.asyncio
async def test_the_request_pairs_a_header_per_body_and_counts_with_a_size_zero_search() -> None:
    fake = _opensearch()
    await _text(fake)
    payload = fake.msearch.await_args.kwargs['body']
    headers, bodies = payload[::2], payload[1::2]
    assert [h['index'] for h in headers] == [semantic_search.get_curation_config().items_index] * 2
    knn, count = bodies
    assert list(knn['query']) == ['knn']
    assert knn['size'] == 10
    assert count['size'] == 0
    assert count['track_total_hits'] is True
    clauses = count['query']['bool']['filter']
    assert clauses[-1] == {'bool': {'must_not': [{'exists': {'field': 'pe_embedding'}}]}}
    # the count scopes to the kNN filter plus the unsearchable clause and nothing else
    assert clauses[:-1] == knn['query']['knn']['pe_embedding']['filter']['bool']['filter']


@pytest.mark.asyncio
async def test_a_box_search_counts_items_without_a_box_vector_in_the_same_call() -> None:
    fake = _opensearch()
    await _text(fake, target=SearchTarget.BOX)
    assert fake.msearch.await_count == 1
    count = fake.msearch.await_args.kwargs['body'][3]
    (clause,) = count['query']['bool']['filter'][-1:]
    assert clause['bool']['must_not'][0]['nested']['path'] == 'region_box_embeddings'


@pytest.mark.asyncio
async def test_a_failed_entry_fails_the_search_instead_of_reading_as_no_matches() -> None:
    fake = _opensearch()
    fake.msearch = AsyncMock(
        return_value={
            'responses': [
                {'error': {'type': 'search_phase_execution_exception'}, 'status': 500},
                {},
            ]
        }
    )
    with pytest.raises(RuntimeError, match='_msearch entry failed'):
        await _text(fake)


@pytest.mark.asyncio
async def test_a_short_answer_fails_the_search() -> None:
    fake = _opensearch()
    fake.msearch = AsyncMock(return_value={'responses': []})
    with pytest.raises(RuntimeError, match='answered 0 of 2'):
        await _text(fake)
