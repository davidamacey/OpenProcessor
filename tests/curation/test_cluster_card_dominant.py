"""``GET /clusters`` must not name a candidate cluster after a weak plurality."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.routers.curation.clusters import list_clusters
from src.services.curation.clustering.orchestrator import RESIDUAL_CLUSTER_ID_OFFSET


CANDIDATE = RESIDUAL_CLUSTER_ID_OFFSET


def _bucket(
    cid: int, size: int, classes: list[tuple[str, int]], labelled: int | None = None
) -> dict[str, Any]:
    return {
        'key': cid,
        'doc_count': size,
        'top_class': {'buckets': [{'key': k, 'doc_count': n} for k, n in classes[:3]]},
        'labelled': {'doc_count': sum(n for _k, n in classes) if labelled is None else labelled},
        'validated': {'doc_count': 0},
        'subclusters': {'value': 0},
        'latest_update': {},
    }


async def _cards(*buckets: dict[str, Any]) -> dict[int, dict[str, Any]]:
    os_client = AsyncMock()
    os_client.search = AsyncMock(
        return_value={'aggregations': {'clusters': {'buckets': list(buckets)}}}
    )
    resp = await list_clusters(
        os_client,
        per_cluster=0,
        max_clusters=100,
        kind='all',
        class_id=None,
        cluster_id=None,
        max_rank=None,
        min_blur_ratio=None,
        class_source=None,
    )
    return {c['cluster_id']: c for c in resp['items']}


@pytest.mark.asyncio
async def test_candidate_tie_among_few_labelled_members_is_not_named() -> None:
    cards = await _cards(_bucket(CANDIDATE, 116, [('dumptruck', 1), ('van', 1), ('bus', 1)]))
    card = cards[CANDIDATE]
    assert card['dominant_class_name'] is None
    assert card['dominant_class_id'] is None
    assert card['dominant_count'] == 1
    assert card['labelled_count'] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('classes', 'named'),
    [
        ([('van', 2)], None),  # below the member floor
        ([('van', 3), ('bus', 3)], None),  # tie — no unique winner
        ([('van', 3), ('bus', 2), ('car', 2)], None),  # 3/7 < half of labelled
        ([('van', 4), ('bus', 2), ('car', 1)], 'van'),
    ],
)
async def test_candidate_naming_floors(classes: list[tuple[str, int]], named: str | None) -> None:
    cards = await _cards(_bucket(CANDIDATE + 1, 50, classes))
    assert cards[CANDIDATE + 1]['dominant_class_name'] == named


@pytest.mark.asyncio
async def test_labelled_count_includes_classes_beyond_top_buckets() -> None:
    classes = [('van', 4), ('bus', 1), ('car', 1), ('suv', 1), ('cab', 1)]
    cards = await _cards(_bucket(CANDIDATE + 2, 40, classes))
    card = cards[CANDIDATE + 2]
    assert card['labelled_count'] == 8
    assert card['purity'] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_class_cluster_keeps_its_dominant_class() -> None:
    cards = await _cards(_bucket(7, 5, [('van', 1)]))
    assert cards[7]['dominant_class_name'] == 'van'
    assert cards[7]['dominant_class_id'] == 7


# =============================================================================
# F-12 — kind push-down: kind='class'/'candidate' must filter the query
# *before* aggregating (a bounded cluster_id range), not terms-aggregate
# everything then drop mismatched-kind buckets in Python. Otherwise, with
# more than max_clusters distinct candidate ids outranking the ~80 class
# ids by _count desc, a kind='class' request could come back missing real
# class clusters entirely.
# =============================================================================


@pytest.mark.asyncio
async def test_kind_class_pushes_a_cluster_id_range_filter_into_the_query() -> None:
    os_client = AsyncMock()
    os_client.search = AsyncMock(return_value={'aggregations': {'clusters': {'buckets': []}}})
    await list_clusters(
        os_client,
        per_cluster=0,
        max_clusters=100,
        kind='class',
        class_id=None,
        cluster_id=None,
        max_rank=None,
        min_blur_ratio=None,
        class_source=None,
    )
    assert os_client.search.await_args is not None
    body = os_client.search.await_args.kwargs['body']
    filt = body['query']['bool']['filter']
    assert {'range': {'cluster_id': {'gte': 0, 'lt': CANDIDATE}}} in filt


@pytest.mark.asyncio
async def test_kind_candidate_pushes_a_cluster_id_range_filter_into_the_query() -> None:
    os_client = AsyncMock()
    os_client.search = AsyncMock(return_value={'aggregations': {'clusters': {'buckets': []}}})
    await list_clusters(
        os_client,
        per_cluster=0,
        max_clusters=100,
        kind='candidate',
        class_id=None,
        cluster_id=None,
        max_rank=None,
        min_blur_ratio=None,
        class_source=None,
    )
    assert os_client.search.await_args is not None
    body = os_client.search.await_args.kwargs['body']
    filt = body['query']['bool']['filter']
    assert {'range': {'cluster_id': {'gte': CANDIDATE}}} in filt


@pytest.mark.asyncio
async def test_kind_class_returns_every_class_bucket_even_with_far_more_candidate_buckets() -> None:
    """The load-bearing regression: with 1200 candidate buckets (each
    outranking every class bucket by doc_count) and max_clusters=100, a
    kind='class' request must still return every class-kind bucket --
    because the range filter keeps candidate docs out of the aggregation
    entirely, not because Python got lucky sorting through the truncated
    top-100."""
    class_buckets = [_bucket(cid, 5, [('van', 1)]) for cid in range(80)]
    # Simulate what a real per-kind-filtered query would return: with the
    # range filter applied server-side, only class buckets exist in this
    # response at all (candidate docs never enter the agg).
    os_client = AsyncMock()
    os_client.search = AsyncMock(
        return_value={'aggregations': {'clusters': {'buckets': class_buckets}}}
    )
    resp = await list_clusters(
        os_client,
        per_cluster=0,
        max_clusters=100,
        kind='class',
        class_id=None,
        cluster_id=None,
        max_rank=None,
        min_blur_ratio=None,
        class_source=None,
    )
    assert {c['cluster_id'] for c in resp['items']} == set(range(80))
