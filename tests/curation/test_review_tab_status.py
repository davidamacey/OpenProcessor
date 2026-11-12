"""``GET /review/tabs`` lists the queues that can fill, each with the items it holds under the filter."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.curation import _raw_opensearch_dep, router as curation_router
from src.services.curation import review_tab_status as rts
from src.services.curation.review_filters import ReviewFilters


def test_the_audit_queue_is_absent_without_a_pending_sample() -> None:
    assert rts.absent_queues(has_pending_audit=True) == {}
    absent = rts.absent_queues(has_pending_audit=False)
    assert list(absent) == ['audit']
    assert 'audit sample' in absent['audit'].lower()


@pytest.mark.asyncio
async def test_counts_come_from_one_msearch_with_exact_totals() -> None:
    fake = AsyncMock()
    fake.msearch = AsyncMock(
        return_value={
            'responses': [
                {'hits': {'total': {'value': 12}}},
                {'hits': {'total': {'value': 3}}},
            ]
        }
    )
    counts = await rts.tab_counts(fake, ['to_label', 'check_machine'], ReviewFilters())
    assert counts == {'to_label': 12, 'check_machine': 3}
    fake.msearch.assert_awaited_once()
    payload = fake.msearch.await_args.kwargs['body']
    assert len(payload) == 4
    assert payload[1]['size'] == 0
    assert payload[1]['track_total_hits'] is True


@pytest.mark.asyncio
async def test_a_failed_msearch_entry_fails_the_call() -> None:
    fake = AsyncMock()
    fake.msearch = AsyncMock(return_value={'responses': [{'error': {'type': 'boom'}}]})
    with pytest.raises(RuntimeError):
        await rts.tab_counts(fake, ['to_label'], ReviewFilters())


def _client(fake: Any) -> TestClient:
    from _curation_app import mount_curation_routers

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def test_route_serves_a_count_per_listed_queue() -> None:
    fake = AsyncMock()
    fake.count = AsyncMock(return_value={'count': 0})

    async def msearch(*, body: list[dict[str, Any]]) -> dict[str, Any]:
        return {'responses': [{'hits': {'total': {'value': 5}}} for _ in body[::2]]}

    fake.msearch = msearch
    tabs = {
        t['id']: t
        for t in _client(fake).get('/curation/projects/default/review/tabs').json()['tabs']
    }
    # No audit sample is pending (every count is 0), so that queue is not listed.
    assert 'audit' not in tabs
    assert {t['count'] for t in tabs.values()} == {5}
    assert {'available', 'unavailable_reason'}.isdisjoint(tabs['to_label'])
