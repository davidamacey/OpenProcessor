"""``GET /clusters`` serves why the list is empty and the partition waiting for activation."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from curation._cluster_cards_helpers import bucket, curation_client, os_returning
from src.services.curation import cluster_empty_state as ces


if TYPE_CHECKING:
    import pytest


def _row(*, serving: bool, status: str = 'ready', clusters: int = 7, items: int = 90) -> Any:
    return SimpleNamespace(
        id=uuid.UUID(int=clusters),
        serving=serving,
        status=status,
        n_clusters=clusters,
        n_items=items,
    )


def test_pending_is_the_newest_ready_partition_that_is_not_serving() -> None:
    rows = [_row(serving=False, clusters=9), _row(serving=True, clusters=3)]
    state = ces.partition_state(rows)
    assert state.serving is True
    assert state.pending is not None
    assert state.pending['clusters'] == 9


def test_failed_and_building_partitions_are_not_pending() -> None:
    rows = [_row(serving=False, status='building'), _row(serving=False, status='failed')]
    state = ces.partition_state(rows)
    assert state.pending is None


def test_no_partition_and_nothing_waiting() -> None:
    reason, message = ces.empty_reason(ces.partition_state([]), total=0, filtered=False)
    assert reason == 'no_partition'
    assert 'clustering' in (message or '').lower()


def test_pending_activation_when_a_ready_partition_waits() -> None:
    state = ces.partition_state([_row(serving=False)])
    reason, message = ces.empty_reason(state, total=0, filtered=False)
    assert reason == 'pending_activation'
    assert 'activate' in (message or '').lower()


def test_filtered_empty_wins_when_filters_are_set() -> None:
    state = ces.partition_state([_row(serving=True)])
    assert ces.empty_reason(state, total=0, filtered=True)[0] == 'filtered_empty'


def test_nothing_to_explain_when_cards_are_served() -> None:
    assert ces.empty_reason(ces.partition_state([]), total=3, filtered=False) == (None, None)


def test_route_serves_the_reason_and_the_pending_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    state = ces.partition_state([_row(serving=False, clusters=9, items=120)])
    client = curation_client(os_returning(), monkeypatch)
    monkeypatch.setattr(
        'src.routers.curation.clusters.read_partition_state', AsyncMock(return_value=state)
    )
    body = client.get('/curation/projects/default/clusters', params={'per_cluster': 0}).json()
    assert body['empty_reason'] == 'pending_activation'
    assert body['pending_partition']['clusters'] == 9
    assert body['pending_partition']['items'] == 120


def test_route_has_no_reason_when_cards_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    state = ces.partition_state([_row(serving=True)])
    client = curation_client(os_returning(bucket(1, 5)), monkeypatch)
    monkeypatch.setattr(
        'src.routers.curation.clusters.read_partition_state', AsyncMock(return_value=state)
    )
    body = client.get('/curation/projects/default/clusters', params={'per_cluster': 0}).json()
    assert body['empty_reason'] is None
    assert body['pending_partition'] is None
