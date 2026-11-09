"""``target=box`` semantic search: nested kNN over ``region_box_embeddings`` (#402)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from _embedding_registry_fake import default_space
from _msearch_fake import install_msearch

from src.config.region_fields import get_region_fields
from src.core.exceptions import RequestRejectedError
from src.domain.embedding_spaces import Target
from src.schemas.curation.search_target import SearchTarget
from src.services.curation import semantic_search
from src.services.curation.semantic_search_box import no_box_vectors_clause


F = get_region_fields()
PATH = F.box_embeddings
VEC = [0.25, 0.5, 0.75, 1.0]


def _box(box_id: str, bbox=(0.1, 0.1, 0.5, 0.5), state='accepted') -> dict:
    return {'box_id': box_id, 'bbox_norm': list(bbox), 'state': state}


def _entry(box_id: str, bbox=(0.1, 0.1, 0.5, 0.5)) -> dict:
    return {'box_id': box_id, 'bbox_norm': list(bbox)}


def _hit(crop_id: str, offsets: list[int], *, score=0.8, boxes=None, entries=None) -> dict:
    boxes = boxes if boxes is not None else [_box('b1'), _box('b2')]
    entries = entries if entries is not None else [_entry('b1'), _entry('b2')]
    return {
        '_id': crop_id,
        '_score': score,
        '_source': {
            'crop_id': crop_id,
            'image_path': '/data/a.jpg',
            F.boxes: boxes,
            PATH: entries,
        },
        'inner_hits': {
            'matched_box': {'hits': {'hits': [{'_nested': {'offset': o}} for o in offsets]}}
        },
    }


def _fake_os(hits: list[dict], *, source_hits: list[dict] | None = None, unembedded=0):
    fake = install_msearch(AsyncMock())

    async def _search(*, index: str, body: dict) -> dict:
        if 'ids' in body.get('query', {}):
            sh = source_hits or []
            return {'hits': {'hits': sh, 'total': {'value': len(sh)}}}
        return {'hits': {'hits': hits, 'total': {'value': len(hits)}}}

    fake.search = AsyncMock(side_effect=_search)
    fake.count = AsyncMock(return_value={'count': unembedded})
    return fake


def _enc() -> MagicMock:
    enc = MagicMock()
    enc.text_model = 'pe_text_encoder'
    enc.encode_text = MagicMock(return_value=np.asarray([VEC], dtype=np.float32))
    return enc


def _knn_body(fake: AsyncMock) -> dict:
    bodies = [c.kwargs['body'] for c in fake.search.call_args_list]
    return next(b for b in bodies if 'ids' not in b['query'])


async def _text(fake, **kw):
    return await semantic_search.semantic_text_search(
        opensearch=fake,
        pe_encoder=_enc(),
        executor=None,
        query='wheel',
        page=kw.pop('page', 1),
        page_size=kw.pop('page_size', 10),
        **kw,
    )


@pytest.mark.asyncio
async def test_box_target_builds_a_nested_knn_with_the_parent_filter_outside():
    fake = _fake_os([])
    await _text(fake, target=SearchTarget.BOX, cluster_uid='6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11')
    query = _knn_body(fake)['query']
    nested = query['bool']['must'][0]['nested']
    assert nested['path'] == PATH
    assert nested['score_mode'] == 'max'
    knn = nested['query']['knn'][f'{PATH}.embedding']
    assert knn['vector'] == pytest.approx(VEC)
    assert 'inner_hits' in nested
    # the parent filter is outside the nested clause, and never inside it
    assert {'term': {'cluster_uid': '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11'}} in query['bool'][
        'filter'
    ]
    assert 'filter' not in knn
    assert 'cluster_uid' not in str(nested)


@pytest.mark.asyncio
async def test_box_target_over_fetches_candidates_but_stays_capped():
    fake = _fake_os([])
    await _text(fake, target=SearchTarget.BOX, page=2, page_size=10)
    k = _knn_body(fake)['query']['bool']['must'][0]['nested']['query']['knn'][f'{PATH}.embedding'][
        'k'
    ]
    assert 20 < k <= 2000
    fake = _fake_os([])
    await _text(fake, target=SearchTarget.BOX, page=10, page_size=200)
    k = _knn_body(fake)['query']['bool']['must'][0]['nested']['query']['knn'][f'{PATH}.embedding'][
        'k'
    ]
    assert k == 2000


@pytest.mark.asyncio
async def test_item_target_is_unchanged():
    fake = _fake_os([])
    await _text(fake)
    assert 'knn' in _knn_body(fake)['query']
    out = await _text(_fake_os([_hit('c1', [0])]))
    assert 'matched_box_id' not in out['items'][0]


@pytest.mark.asyncio
async def test_matched_box_id_comes_from_the_inner_hit_offset():
    out = await _text(_fake_os([_hit('c1', [1])]), target=SearchTarget.BOX)
    item = out['items'][0]
    assert item['matched_box_id'] == 'b2'
    assert item['matched_box_stale'] is False
    assert item['semantic_score'] == 0.8
    assert [i['crop_id'] for i in out['items']] == ['c1']


@pytest.mark.asyncio
async def test_missing_inner_hits_gives_no_match_not_a_crash():
    hit = _hit('c1', [0])
    del hit['inner_hits']
    out = await _text(_fake_os([hit]), target=SearchTarget.BOX)
    assert out['items'][0]['matched_box_id'] is None
    assert out['items'][0]['matched_box_stale'] is None


@pytest.mark.asyncio
async def test_stale_box_vector_is_flagged_not_hidden():
    moved = [_box('b1', bbox=(0.2, 0.2, 0.6, 0.6)), _box('b2')]
    out = await _text(_fake_os([_hit('c1', [0], boxes=moved)]), target=SearchTarget.BOX)
    assert out['items'][0]['matched_box_id'] == 'b1'
    assert out['items'][0]['matched_box_stale'] is True


@pytest.mark.asyncio
async def test_box_vectors_are_read_back_without_the_vector():
    fake = _fake_os([])
    await _text(fake, target=SearchTarget.BOX)
    excludes = _knn_body(fake)['_source']['excludes']
    assert PATH not in excludes
    assert f'{PATH}.embedding' in excludes


@pytest.mark.asyncio
async def test_box_target_counts_items_without_any_box_vector():
    fake = _fake_os([], unembedded=3)
    out = await _text(fake, target=SearchTarget.BOX)
    assert out['unembedded_in_scope'] == 3
    clauses = fake.count.call_args.kwargs['body']['query']['bool']['filter']
    assert no_box_vectors_clause(default_space(Target.BOX)) in clauses


# --- /search/similar with box_id ---------------------------------------------------------------

SRC_ENTRIES = [
    {'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.5, 0.5], 'embedding': VEC},
]


def _src(entries=None, boxes=None) -> list[dict]:
    return [
        {
            '_id': 'src',
            '_source': {
                PATH: SRC_ENTRIES if entries is None else entries,
                F.boxes: [_box('b1'), _box('b2')] if boxes is None else boxes,
            },
        }
    ]


@pytest.mark.asyncio
async def test_similar_box_id_uses_the_stored_box_vector_as_the_query():
    fake = _fake_os([], source_hits=_src())
    await semantic_search.semantic_similar_search(
        opensearch=fake, crop_id='src', box_id='b1', target=SearchTarget.BOX, page=1, page_size=10
    )
    nested = _knn_body(fake)['query']['bool']['must'][0]['nested']
    assert nested['query']['knn'][f'{PATH}.embedding']['vector'] == VEC


@pytest.mark.asyncio
async def test_similar_box_id_with_item_target_searches_items_by_the_box_vector():
    fake = _fake_os([], source_hits=_src())
    await semantic_search.semantic_similar_search(
        opensearch=fake, crop_id='src', box_id='b1', page=1, page_size=10
    )
    assert _knn_body(fake)['query']['knn']['pe_embedding']['vector'] == VEC


@pytest.mark.asyncio
async def test_similar_unknown_box_is_404_box_not_found():
    fake = _fake_os([], source_hits=_src())
    with pytest.raises(RequestRejectedError) as ei:
        await semantic_search.semantic_similar_search(
            opensearch=fake, crop_id='src', box_id='b9', page=1, page_size=10
        )
    assert (ei.value.status_code, ei.value.detail['error']) == (404, 'box_not_found')


@pytest.mark.asyncio
async def test_similar_box_without_vector_is_409_box_not_embedded():
    fake = _fake_os([], source_hits=_src(entries=[]))
    with pytest.raises(RequestRejectedError) as ei:
        await semantic_search.semantic_similar_search(
            opensearch=fake, crop_id='src', box_id='b2', page=1, page_size=10
        )
    assert (ei.value.status_code, ei.value.detail['error']) == (409, 'box_not_embedded')


@pytest.mark.asyncio
async def test_exclude_self_with_a_box_drops_that_box_only():
    only_self = _hit('src', [0])  # the source item matched only through b1 (the query box)
    other_item = _hit('c2', [0])
    self_and_other = _hit('src', [0, 1])  # b1 (self) then b2: the item stays, matched by b2
    fake = _fake_os([only_self, other_item], source_hits=_src())
    out = await semantic_search.semantic_similar_search(
        opensearch=fake, crop_id='src', box_id='b1', target=SearchTarget.BOX, page=1, page_size=10
    )
    assert [i['crop_id'] for i in out['items']] == ['c2']
    assert out['total'] == 1
    assert _knn_body(fake)['query']['bool']['must'][0]['nested']['inner_hits']['size'] == 2

    fake = _fake_os([self_and_other], source_hits=_src())
    out = await semantic_search.semantic_similar_search(
        opensearch=fake, crop_id='src', box_id='b1', target=SearchTarget.BOX, page=1, page_size=10
    )
    assert out['items'][0]['matched_box_id'] == 'b2'

    fake = _fake_os([only_self], source_hits=_src())
    out = await semantic_search.semantic_similar_search(
        opensearch=fake,
        crop_id='src',
        box_id='b1',
        exclude_self=False,
        target=SearchTarget.BOX,
        page=1,
        page_size=10,
    )
    assert [i['crop_id'] for i in out['items']] == ['src']


@pytest.mark.asyncio
async def test_box_target_without_box_id_excludes_the_source_item():
    fake = _fake_os([], source_hits=[{'_id': 'src', '_source': {'pe_embedding': VEC}}])
    await semantic_search.semantic_similar_search(
        opensearch=fake, crop_id='src', target=SearchTarget.BOX, page=1, page_size=10
    )
    flt = _knn_body(fake)['query']['bool']['filter']
    assert {'bool': {'must_not': [{'ids': {'values': ['src']}}]}} in flt
