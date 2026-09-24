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
