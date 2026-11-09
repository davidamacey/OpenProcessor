"""Cluster purity of the auto-promote gate.

``auto_promote_clusters`` must count every labelled member in the
purity denominator, not just the classes in its top-5 terms buckets
(``sum_other_doc_count`` carries the rest) — otherwise a many-class
cluster reads as purer than it is and gets promoted.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.curation.cluster_purity import PROMOTE_MIN_MEMBERS, is_promotable
from src.services.curation.clustering.auto_promote import auto_promote_clusters


@pytest.mark.asyncio
async def test_auto_promote_purity_counts_classes_beyond_the_top_buckets() -> None:
    # 90 of 100 labelled are 'a' in the top buckets, but 60 more labelled
    # members sit in classes past the top 5: true purity 90/160 = 0.5625.
    bucket = {
        # Composite-agg bucket key is a dict, not a bare scalar.
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


def test_is_promotable() -> None:
    assert is_promotable(members=10, labelled=10, purity=0.9)
    assert not is_promotable(members=PROMOTE_MIN_MEMBERS - 1, labelled=3, purity=1.0)
    assert not is_promotable(members=10, labelled=4, purity=1.0)  # < half labelled
    assert not is_promotable(members=10, labelled=10, purity=None)
