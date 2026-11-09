"""The search wire names the space it searched and can fuse several (ES-C3).

``space`` picks a live space of the target (default: its primary) and is echoed; ``fuse`` adds
spaces whose rankings are merged by reciprocal rank fusion; each item says which space its
``semantic_score`` was measured in (``score_space``).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from _embedding_registry_fake import default_space, new_space
from fastapi import FastAPI

from src.domain.embedding_spaces import Target
from src.schemas.curation.search_target import SearchTarget
from src.services.curation import semantic_search, semantic_search_run


OTHER = new_space(Target.ITEM, 'mobileclip2_s2', text_model='pe_text_encoder')
DEFAULT_KEY = default_space(Target.ITEM).space_key


def _hit(doc_id: str, score: float) -> dict:
    return {'_id': doc_id, '_score': score, '_source': {'crop_id': doc_id}}


def _msearch_answer(*lists: list[dict], unembedded: int = 0) -> dict:
    knn = [{'took': 2, 'hits': {'hits': h, 'total': {'value': len(h)}}} for h in lists]
    count = {'hits': {'hits': [], 'total': {'value': unembedded}}}
    return {'responses': [*knn, count]}


def _encoder() -> MagicMock:
    enc = MagicMock()
    enc.text_model = 'pe_text_encoder'
    enc.ensure_text_ready = MagicMock(return_value=True)
    enc.encode_text = MagicMock(return_value=np.ones((1, 4), dtype=np.float32))
    return enc


def _opensearch(*lists: list[dict], unembedded: int = 0) -> AsyncMock:
    fake = AsyncMock()
    fake.msearch = AsyncMock(return_value=_msearch_answer(*lists, unembedded=unembedded))
    return fake


@pytest.mark.asyncio
async def test_the_default_answer_echoes_the_primary_space_and_marks_each_item() -> None:
    fake = _opensearch([_hit('a', 0.9)])
    result = await semantic_search.semantic_text_search(
        opensearch=fake, pe_encoder=_encoder(), executor=None, query='q', page=1, page_size=10
    )
    assert (result['space'], result['fuse']) == (DEFAULT_KEY, [])
    assert [(i['crop_id'], i['semantic_score'], i['score_space']) for i in result['items']] == [
        ('a', 0.9, DEFAULT_KEY)
    ]


@pytest.mark.asyncio
async def test_a_named_space_is_searched_and_echoed(embedding_registry) -> None:
    embedding_registry.add(OTHER)
    fake = _opensearch([_hit('a', 0.8)])
    result = await semantic_search.semantic_text_search(
        opensearch=fake,
        pe_encoder=_encoder(),
        executor=None,
        query='q',
        page=1,
        page_size=10,
        space_key=OTHER.space_key,
    )
    knn = fake.msearch.await_args.kwargs['body'][1]
    assert list(knn['query']['knn']) == [OTHER.vector_field]
    assert (result['space'], result['items'][0]['score_space']) == (OTHER.space_key,) * 2


@pytest.mark.asyncio
async def test_an_unknown_space_is_a_404_not_an_empty_page() -> None:
    from src.domain.embedding_spaces import EmbeddingSpaceError

    with pytest.raises(EmbeddingSpaceError) as excinfo:
        await semantic_search.semantic_text_search(
            opensearch=_opensearch(),
            pe_encoder=_encoder(),
            executor=None,
            query='q',
            page=1,
            page_size=10,
            space_key='nope_v9',
        )
    assert (excinfo.value.status, excinfo.value.code) == (404, 'embedding_space_not_found')


@pytest.mark.asyncio
async def test_fusing_two_spaces_ranks_by_rrf_in_one_round_trip(embedding_registry) -> None:
    embedding_registry.add(OTHER)
    # primary ranks a, b, c; the other space ranks b, c, d: b leads (ranks 2 and 1)
    fake = _opensearch(
        [_hit('a', 0.9), _hit('b', 0.8), _hit('c', 0.7)],
        [_hit('b', 0.6), _hit('c', 0.5), _hit('d', 0.4)],
        unembedded=3,
    )
    result = await semantic_search.semantic_text_search(
        opensearch=fake,
        pe_encoder=_encoder(),
        executor=None,
        query='q',
        page=1,
        page_size=10,
        fuse=[OTHER.space_key],
    )
    assert fake.msearch.await_count == 1
    assert len(fake.msearch.await_args.kwargs['body']) == 6  # two kNN pages and the count
    assert [i['crop_id'] for i in result['items']] == ['b', 'c', 'a', 'd']
    # an item keeps the cosine of the first space that found it
    scores = {i['crop_id']: (i['semantic_score'], i['score_space']) for i in result['items']}
    assert scores['b'] == (0.8, DEFAULT_KEY)
    assert scores['d'] == (0.4, OTHER.space_key)
    assert (result['space'], result['fuse'], result['total']) == (
        DEFAULT_KEY,
        [OTHER.space_key],
        4,
    )
    assert result['unembedded_in_scope'] == 3


@pytest.mark.asyncio
async def test_a_fused_page_slices_the_merged_ranking(embedding_registry) -> None:
    embedding_registry.add(OTHER)
    fake = _opensearch([_hit('a', 0.9), _hit('b', 0.8)], [_hit('c', 0.7), _hit('d', 0.6)])
    result = await semantic_search.semantic_text_search(
        opensearch=fake,
        pe_encoder=_encoder(),
        executor=None,
        query='q',
        page=2,
        page_size=2,
        fuse=[OTHER.space_key],
    )
    # both lists have two hits at ranks 1 and 2: a, c tie on rank 1 (id order), b, d on rank 2
    assert [i['crop_id'] for i in result['items']] == ['b', 'd']


def test_rrf_merge_scores_by_reciprocal_rank() -> None:
    first = default_space(Target.ITEM)
    ranked = [(first, [_hit('x', 1.0), _hit('y', 0.5)]), (OTHER, [_hit('y', 0.9)])]
    merged = semantic_search_run.rrf_merge(ranked)
    k = semantic_search_run.RRF_RANK_CONSTANT
    assert [(h['_id'], s) for h, _space, s in merged] == [
        ('y', pytest.approx(1 / (k + 2) + 1 / (k + 1))),
        ('x', pytest.approx(1 / (k + 1))),
    ]


@pytest.mark.asyncio
async def test_a_space_this_api_cannot_encode_is_refused_not_searched(embedding_registry) -> None:
    from src.domain.embedding_spaces import EmbeddingSpaceError

    other = new_space(Target.ITEM, 'siglip2_b', text_model='siglip_text')
    embedding_registry.add(other)
    fake = _opensearch()
    with pytest.raises(EmbeddingSpaceError) as excinfo:
        await semantic_search.semantic_text_search(
            opensearch=fake,
            pe_encoder=_encoder(),
            executor=None,
            query='q',
            page=1,
            page_size=10,
            space_key=other.space_key,
        )
    assert (excinfo.value.status, excinfo.value.code) == (422, 'embedding_model_unavailable')
    assert fake.msearch.await_count == 0


def test_the_three_routes_document_space_and_fuse_and_the_echo() -> None:
    from _curation_app import mount_curation_routers

    from src.routers.curation import router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    schema = app.openapi()
    for path in ('search/text', 'search/similar'):
        op = schema['paths'][f'/curation/projects/{{project}}/{path}']['get']
        names = {p['name'] for p in op['parameters']}
        assert {'space', 'fuse'} <= names, path
    image = schema['paths']['/curation/projects/{project}/search/image']['post']
    assert {'space', 'fuse'} <= {p['name'] for p in image['parameters']}
    response = schema['components']['schemas']['SearchTextResponse']
    assert {'space', 'fuse'} <= set(response['properties'])
    assert 'score_space' in schema['components']['schemas']['SearchItemDoc']['properties']


@pytest.mark.asyncio
async def test_box_target_fusion_matches_each_hit_in_its_own_space(embedding_registry) -> None:
    other_box = new_space(Target.BOX, 'mobileclip2_s2', text_model='pe_text_encoder')
    embedding_registry.add(other_box)
    embedding_registry.add(OTHER)
    fake = _opensearch([_hit('a', 0.9)], [_hit('b', 0.8)])
    result = await semantic_search.semantic_text_search(
        opensearch=fake,
        pe_encoder=_encoder(),
        executor=None,
        query='q',
        page=1,
        page_size=10,
        target=SearchTarget.BOX,
        fuse=[other_box.space_key],
    )
    bodies = fake.msearch.await_args.kwargs['body'][1::2]
    paths = [b['query']['bool']['must'][0]['nested']['path'] for b in bodies[:2]]
    assert paths == ['region_box_embeddings', 'boxvec_mobileclip2_s2_v1']
    assert result['fuse'] == [other_box.space_key]
