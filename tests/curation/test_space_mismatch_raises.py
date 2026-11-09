"""A kNN query names the space it searches and the space its vector was made in (ES-C2).

A mismatch is a programming error and raises ``EmbeddingSpaceMismatch``; a matching query
searches the field the registry names for the space, whatever that field is called.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from _embedding_registry_fake import default_space, new_space
from _msearch_fake import install_msearch

from src.domain.embedding_spaces import EmbeddingSpaceMismatch, Target
from src.schemas.curation.search_target import SearchTarget
from src.services.curation import semantic_search, semantic_search_run
from src.services.curation.semantic_search_box import build_box_knn_query
from src.services.curation.semantic_search_run import QueryVector


ITEM = default_space(Target.ITEM)
BOX = default_space(Target.BOX)
OTHER_ITEM = new_space(Target.ITEM, 'mobileclip2_s2', text_model='pe_text_encoder')
OTHER_BOX = new_space(Target.BOX, 'mobileclip2_s2', text_model='pe_text_encoder')


def test_an_item_query_of_another_space_raises() -> None:
    with pytest.raises(EmbeddingSpaceMismatch, match='mobileclip2_s2_v1'):
        semantic_search_run.build_knn_query(
            [0.1], vector_space_key=OTHER_ITEM.space_key, space=ITEM, k=5, filter_clause=[]
        )


def test_a_box_query_of_another_space_raises() -> None:
    with pytest.raises(EmbeddingSpaceMismatch):
        build_box_knn_query(
            [0.1], vector_space_key=OTHER_BOX.space_key, space=BOX, k=5, filter_clause=[]
        )


def test_a_matching_item_query_searches_the_field_of_the_space() -> None:
    query = semantic_search_run.build_knn_query(
        [0.1],
        vector_space_key=OTHER_ITEM.space_key,
        space=OTHER_ITEM,
        k=5,
        filter_clause=[{'term': {'class_id': 1}}],
    )
    assert list(query['knn']) == ['vec_mobileclip2_s2_v1']
    assert query['knn']['vec_mobileclip2_s2_v1']['filter'] == {
        'bool': {'filter': [{'term': {'class_id': 1}}]}
    }


def test_a_matching_box_query_searches_the_nested_field_of_the_space() -> None:
    query = build_box_knn_query(
        [0.1], vector_space_key=OTHER_BOX.space_key, space=OTHER_BOX, k=5, filter_clause=[]
    )
    (nested,) = query['bool']['must']
    assert nested['nested']['path'] == 'boxvec_mobileclip2_s2_v1'
    assert list(nested['nested']['query']['knn']) == ['boxvec_mobileclip2_s2_v1.embedding']


def test_the_default_space_keeps_the_zero_six_field_names() -> None:
    query = semantic_search_run.build_knn_query(
        [0.1], vector_space_key=ITEM.space_key, space=ITEM, k=5, filter_clause=[]
    )
    assert list(query['knn']) == ['pe_embedding']
    (nested,) = build_box_knn_query(
        [0.1], vector_space_key=BOX.space_key, space=BOX, k=5, filter_clause=[]
    )['bool']['must']
    assert list(nested['nested']['query']['knn']) == ['region_box_embeddings.embedding']


def _opensearch() -> AsyncMock:
    fake = install_msearch(AsyncMock())
    fake.search = AsyncMock(return_value={'hits': {'hits': [], 'total': {'value': 0}}})
    fake.count = AsyncMock(return_value={'count': 0})
    return fake


def _encoder() -> MagicMock:
    enc = MagicMock()
    enc.text_model = 'pe_text_encoder'
    enc.encode_text = MagicMock(return_value=np.ones((1, 4), dtype=np.float32))
    return enc


@pytest.mark.asyncio
async def test_a_search_over_a_flipped_primary_space_queries_its_field(embedding_registry) -> None:
    embedding_registry.add(OTHER_ITEM, primary=True)
    fake = _opensearch()
    await semantic_search.semantic_text_search(
        opensearch=fake,
        pe_encoder=_encoder(),
        executor=None,
        query='q',
        page=1,
        page_size=10,
        target=SearchTarget.ITEM,
    )
    body = fake.search.await_args.kwargs['body']
    assert list(body['query']['knn']) == ['vec_mobileclip2_s2_v1']
    count = fake.count.await_args.kwargs['body']
    assert count['query']['bool']['filter'][-1] == {
        'bool': {'must_not': [{'exists': {'field': 'vec_mobileclip2_s2_v1'}}]}
    }


@pytest.mark.asyncio
async def test_a_vector_of_the_wrong_space_reaching_the_search_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def wrong_space(_space: object) -> QueryVector:
        return QueryVector([0.1, 0.2], OTHER_ITEM.space_key)

    monkeypatch.setattr(semantic_search, '_text_vector_for', lambda *_a: wrong_space)
    with pytest.raises(EmbeddingSpaceMismatch):
        await semantic_search.semantic_text_search(
            opensearch=_opensearch(),
            pe_encoder=_encoder(),
            executor=None,
            query='q',
            page=1,
            page_size=10,
        )
