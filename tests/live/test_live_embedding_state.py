"""Live: items, stats and search expose which items have no embedding."""

from __future__ import annotations

from typing import Any

import pytest


pytestmark = pytest.mark.live


def test_item_reads_carry_embedding_state(api_client: Any) -> None:
    resp = api_client.get('/crops', params={'page_size': 5})
    resp.raise_for_status()
    crops = resp.json()['crops']
    assert crops
    assert all('embedding_state' in c for c in crops)


def test_dataset_stats_break_down_embedding(api_client: Any) -> None:
    resp = api_client.get('/stats/dataset')
    resp.raise_for_status()
    body = resp.json()
    emb = body['embedding']
    assert emb['embedded'] + emb['not_embedded'] == body['total_crops']
    assert isinstance(emb['by_state'], dict)


def test_ordered_page_reports_unembedded(api_client: Any) -> None:
    resp = api_client.get('/crops', params={'order': 'diverse', 'k': 5})
    if resp.status_code != 200 or resp.json().get('method') != 'diverse':
        pytest.skip('diverse ordering is not enabled on this harness')
    body = resp.json()
    assert body['n_unembedded'] is not None
    assert body['n_unembedded'] <= body['n_pool']
