"""Cluster purity: one rule for the auto-promote gate and the cluster cards.

- ``auto_promote_clusters`` must count every labelled member in the
  purity denominator, not just the classes in its top-5 terms buckets
  (``sum_other_doc_count`` carries the rest) — otherwise a many-class
  cluster reads as purer than it is and gets promoted.
- ``GET /clusters`` serves ``purity_tier`` and ``promotable`` per card
  from the same thresholds, plus the thresholds themselves and the
  core-member similarity cut line.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.curation.clustering import orchestrator as _orchestrator


_ = _orchestrator.ITEMS_INDEX  # orchestrator must load before auto_promote

from src.routers.curation.clusters import list_clusters  # noqa: E402
from src.services.curation.cluster_ids import CORE_SIMILARITY_MIN  # noqa: E402
from src.services.curation.cluster_purity import (  # noqa: E402
    PROMOTE_MIN_MEMBERS,
    PROMOTE_MIN_PURITY,
    PURITY_MIXED_MIN,
    is_promotable,
    purity_tier,
)
from src.services.curation.clustering.auto_promote import auto_promote_clusters  # noqa: E402


@pytest.mark.asyncio
async def test_auto_promote_purity_counts_classes_beyond_the_top_buckets() -> None:
    # 90 of 100 labelled are 'a' in the top buckets, but 60 more labelled
    # members sit in classes past the top 5: true purity 90/160 = 0.5625.
    bucket = {
        # F-29: composite-agg bucket key is a dict, not a bare scalar.
        'key': {'cluster_id': 3},
        'doc_count': 170,
        'top_class': {
            'buckets': [{'key': 'a', 'doc_count': 90}, {'key': 'b', 'doc_count': 10}],
            'sum_other_doc_count': 60,
        },
    }
    client = MagicMock()
    client.search = AsyncMock(return_value={'aggregations': {'clusters': {'buckets': [bucket]}}})
    out = await auto_promote_clusters(client, dry_run=True)
    card = out['clusters'][0]
    assert card['labelled_total'] == 160
    assert card['purity'] == pytest.approx(90 / 160, abs=1e-4)
    assert card['promote'] is False


@pytest.mark.parametrize(
    ('purity', 'tier'),
    [
        (None, None),
        (PROMOTE_MIN_PURITY, 'pure'),
        (PROMOTE_MIN_PURITY - 0.01, 'mixed'),
        (PURITY_MIXED_MIN, 'mixed'),
        (PURITY_MIXED_MIN - 0.01, 'noisy'),
    ],
)
def test_purity_tier_uses_the_promote_gate(purity: float | None, tier: str | None) -> None:
    assert purity_tier(purity) == tier


def test_is_promotable() -> None:
    assert is_promotable(members=10, labelled=10, purity=0.9)
    assert not is_promotable(members=PROMOTE_MIN_MEMBERS - 1, labelled=3, purity=1.0)
    assert not is_promotable(members=10, labelled=4, purity=1.0)  # < half labelled
    assert not is_promotable(members=10, labelled=10, purity=None)


async def _cards(*buckets: dict[str, Any]) -> dict[str, Any]:
    os_client = AsyncMock()
    os_client.search = AsyncMock(
        return_value={'aggregations': {'clusters': {'buckets': list(buckets)}}}
    )
    return await list_clusters(
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


def _bucket(cid: int, size: int, classes: list[tuple[str, int]]) -> dict[str, Any]:
    return {
        'key': cid,
        'doc_count': size,
        'top_class': {'buckets': [{'key': k, 'doc_count': n} for k, n in classes]},
        'labelled': {'doc_count': sum(n for _k, n in classes)},
        'validated': {'doc_count': 0},
        'subclusters': {'value': 0},
        'latest_update': {},
    }


@pytest.mark.asyncio
async def test_cluster_cards_serve_tier_promotable_and_thresholds() -> None:
    resp = await _cards(
        _bucket(1, 10, [('a', 9), ('b', 1)]),  # 0.9 -> pure, promotable
        _bucket(2, 10, [('a', 8), ('b', 2)]),  # 0.8 -> mixed (below the 0.85 gate)
        _bucket(3, 10, [('a', 5), ('b', 5)]),  # 0.5 -> noisy
    )
    cards = {c['cluster_id']: c for c in resp['items']}
    assert (cards[1]['purity_tier'], cards[1]['promotable']) == ('pure', True)
    assert (cards[2]['purity_tier'], cards[2]['promotable']) == ('mixed', False)
    assert (cards[3]['purity_tier'], cards[3]['promotable']) == ('noisy', False)
    assert resp['purity_thresholds'] == {
        'pure_min': PROMOTE_MIN_PURITY,
        'mixed_min': PURITY_MIXED_MIN,
        'promote_min_members': PROMOTE_MIN_MEMBERS,
        'promote_min_labelled_share': 0.5,
    }
    assert resp['core_similarity_min'] == CORE_SIMILARITY_MIN
