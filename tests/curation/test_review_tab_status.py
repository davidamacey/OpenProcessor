"""``GET /review/tabs`` serves, per tab, whether it can fill, why not, and how many items it holds."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation import review_tab_status as rts


_NO_PROBE = {
    'has_probe_predictions': False,
    'has_item_scores': True,
    'has_imported_labels': False,
    'has_unembedded_items': False,
    'suggested_reprocess': None,
}


def test_probe_tabs_are_unavailable_without_probe_predictions() -> None:
    for tab in ('uncertainty', 'model_disagreements'):
        reason = rts.unavailable_reason(tab, _NO_PROBE)
        assert reason is not None
        assert 'probe' in reason.lower()
    assert (
        rts.unavailable_reason('uncertainty', {**_NO_PROBE, 'has_probe_predictions': True}) is None
    )


def test_imported_tab_needs_imported_labels() -> None:
    assert 'import' in (rts.unavailable_reason('imported', _NO_PROBE) or '').lower()
    assert rts.unavailable_reason('imported', {**_NO_PROBE, 'has_imported_labels': True}) is None


def test_ordinary_tabs_are_available() -> None:
    assert rts.unavailable_reason('all', _NO_PROBE) is None
    assert rts.unavailable_reason('mismatches', _NO_PROBE) is None


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
    counts = await rts.tab_counts(fake, ['all', 'mismatches'])
    assert counts == {'all': 12, 'mismatches': 3}
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
        await rts.tab_counts(fake, ['all'])


def _client(fake: Any) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def test_route_serves_available_reason_and_count_per_tab() -> None:
    fake = AsyncMock()
    fake.count = AsyncMock(return_value={'count': 0})

    async def msearch(*, body: list[dict[str, Any]]) -> dict[str, Any]:
        return {'responses': [{'hits': {'total': {'value': 5}}} for _ in body[::2]]}

    fake.msearch = msearch
    tabs = {
        t['id']: t
        for t in _client(fake).get('/curation/projects/default/review/tabs').json()['tabs']
    }
    assert tabs['uncertainty']['available'] is False
    assert 'probe' in tabs['uncertainty']['unavailable_reason'].lower()
    assert tabs['uncertainty']['count'] == 0
    assert tabs['all']['available'] is True
    assert tabs['all']['unavailable_reason'] is None
    assert tabs['all']['count'] == 5
