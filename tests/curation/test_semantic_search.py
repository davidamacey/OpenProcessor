"""Unit tests for :mod:`src.services.curation.semantic_search`.

No OpenSearch/torch involved — pure filter-composition + kNN-query-shape
+ hydration/pagination checks against a stubbed opensearch client and a
stubbed PEEncoder, same stubbing style ``test_pe_encoder.py`` uses.
"""

from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.config import CurationConfig
from src.core.exceptions import RequestRejectedError
from src.services.curation import semantic_search
from src.services.curation.item_filter import ItemFilter


PE_DIM = 1024


def _fake_encoder() -> MagicMock:
    enc = MagicMock()
    vec = np.zeros((1, PE_DIM), dtype=np.float32)
    vec[0, 0] = 1.0
    enc.encode_text = MagicMock(return_value=vec)
    return enc


def _fake_os(hits: list[dict]) -> AsyncMock:
    fake = AsyncMock()
    fake.search = AsyncMock(return_value={'hits': {'hits': hits, 'total': {'value': len(hits)}}})
    fake.count = AsyncMock(return_value={'count': 0})
    return fake


def _fake_os_paged(candidates: list[dict]) -> AsyncMock:
    """Simulates OpenSearch's real from/size/min_score slicing —
    ``candidates`` is the fixed, already score-ordered ANN result set the
    ``knn`` clause would produce server-side; the fake applies min_score
    then from/size the same way OpenSearch would, so tests exercise the
    real request shape instead of asserting against a stub that ignores
    the body."""
    fake = AsyncMock()

    async def _search(*, index: str, body: dict) -> dict:
        pool = candidates
        min_score = body.get('min_score')
        if min_score is not None:
            pool = [h for h in pool if h.get('_score', 0.0) >= min_score]
        total = len(pool)
        frm = body.get('from', 0)
        size = body.get('size', len(pool))
        return {'hits': {'hits': pool[frm : frm + size], 'total': {'value': total}}}

    fake.search = AsyncMock(side_effect=_search)
    fake.count = AsyncMock(return_value={'count': 0})
    return fake


# =============================================================================
# _build_filter
# =============================================================================


def test_build_filter_default_excludes_validated_and_dismissed_and_holdout():
    filt = semantic_search._build_filter(
        tab=None,
        cluster_uid=None,
        date_from=None,
        date_to=None,
        max_rank=None,
        min_blur_ratio=None,
        hide_near_duplicates=False,
        include_test=False,
    )
    must_not_clause = next(f for f in filt if 'bool' in f and 'must_not' in f['bool'])
    must_not = must_not_clause['bool']['must_not']
    assert {'term': {'class_validated': True}} in must_not
    assert {'exists': {'field': 'review_dismissed_at'}} in must_not
    assert {'term': {'test_holdout': True}} in must_not
    # An excluded item must never surface in semantic search either.
    assert {'term': {'class_excluded': True}} in must_not


def test_build_filter_include_test_keeps_holdout_crops():
    filt = semantic_search._build_filter(
        tab=None,
        cluster_uid=None,
        date_from=None,
        date_to=None,
        max_rank=None,
        min_blur_ratio=None,
        hide_near_duplicates=False,
        include_test=True,
    )
    must_not_clause = next(f for f in filt if 'bool' in f and 'must_not' in f['bool'])
    must_not = must_not_clause['bool']['must_not']
    assert {'term': {'test_holdout': True}} not in must_not


def test_build_filter_cluster_uid():
    filt = semantic_search._build_filter(
        tab=None,
        cluster_uid='6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11',
        date_from=None,
        date_to=None,
        max_rank=None,
        min_blur_ratio=None,
        hide_near_duplicates=False,
        include_test=False,
    )
    assert {'term': {'cluster_uid': '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11'}} in filt


def test_build_filter_date_range():
    filt = semantic_search._build_filter(
        tab=None,
        cluster_uid=None,
        date_from='2026-01-01',
        date_to='2026-06-01',
        max_rank=None,
        min_blur_ratio=None,
        hide_near_duplicates=False,
        include_test=False,
    )
    assert {'range': {'created_at': {'gte': '2026-01-01', 'lte': '2026-06-01'}}} in filt


def test_build_filter_reuses_tab_query():
    filt = semantic_search._build_filter(
        tab='uncertainty',
        cluster_uid=None,
        date_from=None,
        date_to=None,
        max_rank=None,
        min_blur_ratio=None,
        hide_near_duplicates=False,
        include_test=False,
    )
    # Should not raise, and should produce a non-empty filter (the
    # 'uncertainty' tab's must + must_not both feed in).
    assert filt


def test_build_filter_unknown_tab_raises_400():
    from src.core.exceptions import RequestRejectedError

    with pytest.raises(RequestRejectedError):
        semantic_search._build_filter(
            tab='not_a_real_tab',
            cluster_uid=None,
            date_from=None,
            date_to=None,
            max_rank=None,
            min_blur_ratio=None,
            hide_near_duplicates=False,
            include_test=False,
        )


def test_build_filter_hide_near_duplicates():
    filt = semantic_search._build_filter(
        tab=None,
        cluster_uid=None,
        date_from=None,
        date_to=None,
        max_rank=None,
        min_blur_ratio=0.5,
        hide_near_duplicates=True,
        include_test=False,
    )
    # Both null-safe should-clauses present.
    should_fields = [
        next(iter(c['bool']['should'][0].keys()))
        for c in filt
        if 'bool' in c and 'should' in c.get('bool', {})
    ]
    assert 'range' in should_fields or any(
        'range' in c.get('bool', {}).get('should', [{}])[0] for c in filt
    )


# =============================================================================
# build_knn_query
# =============================================================================


def test_build_knn_query_shape():
    vec = [0.1] * PE_DIM
    q = semantic_search.build_knn_query(vec, k=50, filter_clause=[{'term': {'class_id': 1}}])
    assert q == {
        'knn': {
            'pe_embedding': {
                'vector': vec,
                'k': 50,
                'filter': {'bool': {'filter': [{'term': {'class_id': 1}}]}},
            }
        }
    }


def test_build_knn_query_no_filter_omits_filter_key():
    vec = [0.1] * PE_DIM
    q = semantic_search.build_knn_query(vec, k=10, filter_clause=[])
    assert 'filter' not in q['knn']['pe_embedding']


# =============================================================================
# semantic_text_search — end-to-end against stubbed collaborators
# =============================================================================


@pytest.mark.asyncio
async def test_semantic_text_search_empty_query_short_circuits():
    result = await semantic_search.semantic_text_search(
        opensearch=_fake_os([]),
        pe_encoder=_fake_encoder(),
        executor=None,
        query='   ',
        page=1,
        page_size=30,
    )
    assert result == {
        'items': [],
        'total': 0,
        'page': 1,
        'page_size': 30,
        'unembedded_in_scope': 0,
    }


@pytest.mark.asyncio
async def test_semantic_text_search_actually_offloads_to_the_given_executor():
    """Not just "encode_text was called with the right args" (that would
    pass identically whether it ran inline or offloaded) — captures the
    real event loop's ``run_in_executor`` call and asserts it was handed
    the caller-supplied executor and ``encode_text`` itself, proving the
    call genuinely went through the offload path rather than being
    invoked directly on the event loop. The callable handed over is the
    bound-context wrapper (``run_in_executor_bound``), so it is proven by
    running it on the executor thread, not by identity."""
    import threading

    encoder = _fake_encoder()
    encode_threads: list[str] = []
    real_encode = encoder.encode_text.return_value

    def _encode(texts):
        encode_threads.append(threading.current_thread().name)
        return real_encode

    encoder.encode_text.side_effect = _encode
    fake_os = _fake_os([])
    real_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='offload-')
    loop = asyncio.get_running_loop()
    captured: list[tuple[object, object, tuple]] = []
    original_run_in_executor = loop.run_in_executor

    def _spy_run_in_executor(executor, fn, *args):
        captured.append((executor, fn, args))
        return original_run_in_executor(executor, fn, *args)

    loop.run_in_executor = _spy_run_in_executor  # type: ignore[method-assign,assignment]
    try:
        await semantic_search.semantic_text_search(
            opensearch=fake_os,
            pe_encoder=encoder,
            executor=real_executor,
            query='blue sedan',
            page=1,
            page_size=10,
        )
    finally:
        loop.run_in_executor = original_run_in_executor  # type: ignore[method-assign]
        real_executor.shutdown(wait=True)

    assert len(captured) == 1
    executor_arg, _fn_arg, _call_args = captured[0]
    assert executor_arg is real_executor
    encoder.encode_text.assert_called_once_with(['blue sedan'])
    assert len(encode_threads) == 1
    assert encode_threads[0].startswith('offload-')


@pytest.mark.asyncio
async def test_semantic_text_search_min_score_is_sent_to_opensearch():
    """min_score moves into the request body's top-level
    min_score, applied by OpenSearch itself instead of filtered out of
    the full hit list in Python."""
    hits = [
        {'_id': 'a', '_score': 0.9, '_source': {'crop_id': 'a'}},
        {'_id': 'b', '_score': 0.2, '_source': {'crop_id': 'b'}},
    ]
    fake_os = _fake_os_paged(hits)
    result = await semantic_search.semantic_text_search(
        opensearch=fake_os,
        pe_encoder=_fake_encoder(),
        executor=None,
        query='white pickup truck',
        page=1,
        page_size=30,
        min_score=0.5,
    )
    assert fake_os.search.await_args is not None
    assert fake_os.search.await_args.kwargs['body']['min_score'] == 0.5
    assert result['total'] == 1
    assert result['items'][0]['crop_id'] == 'a'


@pytest.mark.asyncio
async def test_semantic_text_search_pages_via_from_size_not_python_slicing():
    """OpenSearch's from/size do the paging now (not a Python
    slice over an over-fetched k-sized hit list)."""
    hits = [{'_id': str(i), '_score': 1.0, '_source': {'crop_id': str(i)}} for i in range(5)]
    fake_os = _fake_os_paged(hits)
    result = await semantic_search.semantic_text_search(
        opensearch=fake_os,
        pe_encoder=_fake_encoder(),
        executor=None,
        query='q',
        page=2,
        page_size=2,
    )
    assert fake_os.search.await_args is not None
    body = fake_os.search.await_args.kwargs['body']
    assert body['from'] == 2
    assert body['size'] == 2
    assert result['total'] == 5
    assert [i['crop_id'] for i in result['items']] == ['2', '3']


@pytest.mark.asyncio
async def test_semantic_text_search_from_size_paging_matches_old_python_slicing():
    """Load-bearing regression: for a fixed ANN candidate set, paging via
    from/size returns the identical ids/order across pages 1-3 that the
    old "fetch k, slice in Python" approach produced."""
    hits = [
        {'_id': str(i), '_score': 1.0 - i * 0.01, '_source': {'crop_id': str(i)}} for i in range(9)
    ]
    page_size = 3
    all_page_items: list[list[str]] = []
    for page in (1, 2, 3):
        result = await semantic_search.semantic_text_search(
            opensearch=_fake_os_paged(hits),
            pe_encoder=_fake_encoder(),
            executor=None,
            query='q',
            page=page,
            page_size=page_size,
        )
        all_page_items.append([i['crop_id'] for i in result['items']])

    # Equivalent to slicing the same score-ordered candidate list in
    # Python page by page — same ids, same order, no gaps/overlaps.
    expected = [str(i) for i in range(9)]
    assert all_page_items == [expected[0:3], expected[3:6], expected[6:9]]


@pytest.mark.asyncio
async def test_semantic_text_search_k_is_ann_depth_not_page_size():
    """`k` (ANN candidate depth) stays page*page_size — it must not
    collapse to page_size just because from/size now do the slicing."""
    fake_os = _fake_os_paged([])
    await semantic_search.semantic_text_search(
        opensearch=fake_os,
        pe_encoder=_fake_encoder(),
        executor=None,
        query='q',
        page=3,
        page_size=20,
    )
    assert fake_os.search.await_args is not None
    body = fake_os.search.await_args.kwargs['body']
    assert body['query']['knn']['pe_embedding']['k'] == 60  # page(3) * page_size(20)
    assert body['size'] == 20
    assert body['from'] == 40


@pytest.mark.asyncio
async def test_semantic_text_search_no_hits_returns_empty_items():
    result = await semantic_search.semantic_text_search(
        opensearch=_fake_os([]),
        pe_encoder=_fake_encoder(),
        executor=None,
        query='nonexistent',
        page=1,
        page_size=30,
    )
    assert result['items'] == []
    assert result['total'] == 0


@pytest.mark.asyncio
async def test_semantic_text_search_excludes_embedding_fields_from_source():
    fake_os = _fake_os([])
    await semantic_search.semantic_text_search(
        opensearch=fake_os,
        pe_encoder=_fake_encoder(),
        executor=None,
        query='q',
        page=1,
        page_size=30,
    )
    _args, kwargs = fake_os.search.call_args
    body = kwargs['body']
    # Also excludes class_id_history — this is a paginated list
    # endpoint, and no list renderer reads it.
    assert set(body['_source']['excludes']) == {
        'pe_embedding',
        'region_box_embeddings',
        'class_id_history',
        'mask_polygon',
    }


@pytest.mark.asyncio
async def test_semantic_text_search_runs_search_and_unembedded_count_together():
    """The count must not wait for the kNN search (#384): each call blocks until the
    other has started, so a sequential implementation times out."""
    search_started = asyncio.Event()
    count_started = asyncio.Event()
    fake_os = _fake_os([])

    async def _search(*, index: str, body: dict) -> dict:
        search_started.set()
        await asyncio.wait_for(count_started.wait(), timeout=2)
        return {'hits': {'hits': [], 'total': {'value': 0}}}

    async def _count(*, index: str, body: dict) -> dict:
        count_started.set()
        await asyncio.wait_for(search_started.wait(), timeout=2)
        return {'count': 7}

    fake_os.search = AsyncMock(side_effect=_search)
    fake_os.count = AsyncMock(side_effect=_count)
    result = await semantic_search.semantic_text_search(
        opensearch=fake_os,
        pe_encoder=_fake_encoder(),
        executor=None,
        query='q',
        page=1,
        page_size=10,
    )
    assert result['unembedded_in_scope'] == 7


@pytest.mark.asyncio
async def test_semantic_text_search_class_filter_is_the_knn_filter_and_the_count_scope():
    """The class filter reaches the kNN `filter` clause and the count with the same
    clauses; the count adds only the not-embedded clause."""
    from src.domain.embedding_state import not_embedded_clause
    from src.services.curation.item_filter import ItemFilter

    fake_os = _fake_os([])
    await semantic_search.semantic_text_search(
        opensearch=fake_os,
        pe_encoder=_fake_encoder(),
        executor=None,
        query='q',
        page=1,
        page_size=10,
        item_filter=ItemFilter(class_names=['car']),
    )
    knn = fake_os.search.await_args.kwargs['body']['query']['knn']['pe_embedding']
    clauses = knn['filter']['bool']['filter']
    name_should = clauses[0]['bool']['should']
    assert {tuple(c['term']) for c in name_should} == {('class_name',), ('proposal_name',)}
    assert all(next(iter(c['term'].values()))['value'] == 'car' for c in name_should)
    count_body = fake_os.count.await_args.kwargs['body']
    assert count_body == {'query': {'bool': {'filter': [*clauses, not_embedded_clause()]}}}


# =============================================================================
# Similar search: the stored vector of one crop as the query (GET /search/similar)
# =============================================================================

STORED_VECTOR = [0.25] * 4
SELF_EXCLUSION = {'bool': {'must_not': [{'ids': {'values': ['c1']}}]}}


def _fake_os_similar(
    *,
    source_hits: list[dict] | None = None,
    knn_hits: list[dict] | None = None,
    unembedded: int = 0,
) -> AsyncMock:
    """Answers the ``ids`` read of the source crop, then the kNN page."""
    if source_hits is None:
        source_hits = [{'_id': 'c1', '_source': {'pe_embedding': STORED_VECTOR}}]
    knn_hits = knn_hits or []
    fake = AsyncMock()

    async def _search(*, index: str, body: dict) -> dict:
        if 'ids' in body.get('query', {}):
            return {'hits': {'hits': source_hits, 'total': {'value': len(source_hits)}}}
        return {'hits': {'hits': knn_hits, 'total': {'value': len(knn_hits)}}}

    fake.search = AsyncMock(side_effect=_search)
    fake.count = AsyncMock(return_value={'count': unembedded})
    return fake


def _knn_body(fake: AsyncMock) -> dict:
    bodies = [c.kwargs['body'] for c in fake.search.call_args_list]
    return next(b for b in bodies if 'knn' in b['query'])


def _knn_filter(fake: AsyncMock) -> list[dict]:
    return _knn_body(fake)['query']['knn']['pe_embedding']['filter']['bool']['filter']


@pytest.mark.asyncio
async def test_similar_uses_the_stored_vector_as_the_knn_query():
    fake = _fake_os_similar()
    await semantic_search.semantic_similar_search(
        opensearch=fake, crop_id='c1', page=2, page_size=10
    )
    knn = _knn_body(fake)['query']['knn']['pe_embedding']
    assert knn['vector'] == STORED_VECTOR
    assert knn['k'] == 20
    params = inspect.signature(semantic_search.semantic_similar_search).parameters
    assert 'pe_encoder' not in params


@pytest.mark.asyncio
async def test_similar_excludes_self_by_default():
    fake = _fake_os_similar()
    await semantic_search.semantic_similar_search(
        opensearch=fake, crop_id='c1', page=1, page_size=10
    )
    assert SELF_EXCLUSION in _knn_filter(fake)


@pytest.mark.asyncio
async def test_similar_keeps_self_when_exclude_self_false():
    fake = _fake_os_similar()
    await semantic_search.semantic_similar_search(
        opensearch=fake, crop_id='c1', exclude_self=False, page=1, page_size=10
    )
    assert SELF_EXCLUSION not in _knn_filter(fake)


@pytest.mark.asyncio
async def test_similar_applies_the_text_search_filters():
    item_filter = ItemFilter(class_names=['car'])
    kwargs = {
        'cluster_uid': '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11',
        'include_test': False,
        'item_filter': item_filter,
    }
    fake = _fake_os_similar()
    await semantic_search.semantic_similar_search(
        opensearch=fake, crop_id='c1', page=1, page_size=10, **kwargs
    )
    text_fake = _fake_os_paged([])
    await semantic_search.semantic_text_search(
        opensearch=text_fake,
        pe_encoder=_fake_encoder(),
        executor=None,
        query='x',
        page=1,
        page_size=10,
        **kwargs,
    )
    text = text_fake.search.call_args.kwargs['body']['query']['knn']['pe_embedding']['filter'][
        'bool'
    ]['filter']
    assert _knn_filter(fake) == [*text, SELF_EXCLUSION]
    assert {'term': {'cluster_uid': '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11'}} in text


@pytest.mark.asyncio
async def test_similar_unknown_crop_is_404_not_found():
    fake = _fake_os_similar(source_hits=[])
    with pytest.raises(RequestRejectedError) as ei:
        await semantic_search.semantic_similar_search(
            opensearch=fake, crop_id='nope', page=1, page_size=10
        )
    assert ei.value.status_code == 404
    assert ei.value.detail['error'] == 'not_found'
    assert all('knn' not in c.kwargs['body']['query'] for c in fake.search.call_args_list)


@pytest.mark.asyncio
async def test_similar_crop_without_vector_is_409():
    fake = _fake_os_similar(source_hits=[{'_id': 'c1', '_source': {'embedding_state': 'deferred'}}])
    with pytest.raises(RequestRejectedError) as ei:
        await semantic_search.semantic_similar_search(
            opensearch=fake, crop_id='c1', page=1, page_size=10
        )
    assert ei.value.status_code == 409
    assert ei.value.detail['error'] == 'crop_not_embedded'
    assert 'deferred' in ei.value.detail['message']


@pytest.mark.asyncio
async def test_similar_reads_only_the_bound_project_index():
    cfg = CurationConfig(items_index='op_a__items')
    fake = AsyncMock()
    seen: list[str] = []

    async def _search(*, index: str, body: dict) -> dict:
        seen.append(index)
        if index == 'op_b__items':  # another project's index would know the crop
            return {'hits': {'hits': [{'_id': 'c1', '_source': {'pe_embedding': STORED_VECTOR}}]}}
        return {'hits': {'hits': [], 'total': {'value': 0}}}

    fake.search = AsyncMock(side_effect=_search)
    fake.count = AsyncMock(return_value={'count': 0})
    with pytest.raises(RequestRejectedError) as ei:
        await semantic_search.semantic_similar_search(
            opensearch=fake, crop_id='c1', page=1, page_size=10, config=cfg
        )
    assert ei.value.status_code == 404
    assert set(seen) == {'op_a__items'}


@pytest.mark.asyncio
async def test_similar_reports_unembedded_in_scope():
    fake = _fake_os_similar(unembedded=5)
    out = await semantic_search.semantic_similar_search(
        opensearch=fake, crop_id='c1', page=1, page_size=10
    )
    assert out['unembedded_in_scope'] == 5
