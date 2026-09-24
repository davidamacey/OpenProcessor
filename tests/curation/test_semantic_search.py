"""Unit tests for :mod:`src.services.curation.semantic_search`.

No OpenSearch/torch involved — pure filter-composition + kNN-query-shape
+ hydration/pagination checks against a stubbed opensearch client and a
stubbed PEEncoder, same stubbing style ``test_pe_encoder.py`` uses.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.services.curation import semantic_search


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
    return fake


def _fake_os_paged(candidates: list[dict]) -> AsyncMock:
    """Simulates OpenSearch's real from/size/min_score slicing (F-24) —
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
    return fake


# =============================================================================
# _build_filter
# =============================================================================


def test_build_filter_default_excludes_validated_and_dismissed_and_holdout():
    filt = semantic_search._build_filter(
        tab=None,
        class_id=None,
        cluster_id=None,
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
    # F-4: an excluded item must never surface in semantic search either.
    assert {'term': {'class_excluded': True}} in must_not


def test_build_filter_include_test_keeps_holdout_crops():
    filt = semantic_search._build_filter(
        tab=None,
        class_id=None,
        cluster_id=None,
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


def test_build_filter_class_and_cluster_ids():
    filt = semantic_search._build_filter(
        tab=None,
        class_id=7,
        cluster_id=42,
        date_from=None,
        date_to=None,
        max_rank=None,
        min_blur_ratio=None,
        hide_near_duplicates=False,
        include_test=False,
    )
    assert {'term': {'class_id': 7}} in filt
    assert {'term': {'cluster_id': 42}} in filt


def test_build_filter_date_range():
    filt = semantic_search._build_filter(
        tab=None,
        class_id=None,
        cluster_id=None,
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
        class_id=None,
        cluster_id=None,
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
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        semantic_search._build_filter(
            tab='not_a_real_tab',
            class_id=None,
            cluster_id=None,
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
        class_id=None,
        cluster_id=None,
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
    assert result == {'items': [], 'total': 0, 'page': 1, 'page_size': 30}


@pytest.mark.asyncio
async def test_semantic_text_search_actually_offloads_to_the_given_executor():
    """Not just "encode_text was called with the right args" (that would
    pass identically whether it ran inline or offloaded) — captures the
    real event loop's ``run_in_executor`` call and asserts it was handed
    the caller-supplied executor and ``encode_text`` itself, proving the
    call genuinely went through the offload path rather than being
    invoked directly on the event loop."""
    encoder = _fake_encoder()
    fake_os = _fake_os([])
    real_executor = ThreadPoolExecutor(max_workers=1)
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
    executor_arg, fn_arg, call_args = captured[0]
    assert executor_arg is real_executor
    assert fn_arg is encoder.encode_text
    assert call_args == (['blue sedan'],)


@pytest.mark.asyncio
async def test_semantic_text_search_min_score_is_sent_to_opensearch():
    """F-24: min_score moves into the request body's top-level
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
    """F-24: OpenSearch's from/size do the paging now (not a Python
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
    """F-24: `k` (ANN candidate depth) stays page*page_size — it must not
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
    # F-25: also excludes class_id_history — this is a paginated list
    # endpoint, and no list renderer reads it.
    assert set(body['_source']['excludes']) == {
        'pe_embedding',
        'v6_embedding',
        'region_embedding',
        'class_id_history',
    }
