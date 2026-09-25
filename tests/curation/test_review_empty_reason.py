"""C3: `GET /review/{tab}`'s `empty_reason` on a zero-result page, and
`GET /review/tabs`'s `empty_state` summary.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation import review_empty_reason as rer


def _client(fake) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


@pytest.mark.asyncio
async def test_compute_empty_reason_no_probe_predictions() -> None:
    fake = AsyncMock()
    fake.count = AsyncMock(return_value={'count': 0})
    reason = await rer.compute_empty_reason(
        'uncertainty', SimpleNamespace(min_mistakenness=None), fake
    )
    assert reason == 'no probe predictions — run a probe'


@pytest.mark.asyncio
async def test_compute_empty_reason_probe_tab_with_predictions_falls_through() -> None:
    fake = AsyncMock()
    fake.count = AsyncMock(return_value={'count': 5})
    reason = await rer.compute_empty_reason(
        'uncertainty', SimpleNamespace(min_mistakenness=None), fake
    )
    assert reason == 'no items match'


@pytest.mark.asyncio
async def test_compute_empty_reason_model_disagreements_probe_gate() -> None:
    fake = AsyncMock()
    fake.count = AsyncMock(return_value={'count': 0})
    reason = await rer.compute_empty_reason(
        'model_disagreements', SimpleNamespace(min_mistakenness=None), fake
    )
    assert reason == 'no probe predictions — run a probe'


@pytest.mark.asyncio
async def test_compute_empty_reason_item_scores_never_computed() -> None:
    fake = AsyncMock()
    fake.count = AsyncMock(return_value={'count': 0})
    reason = await rer.compute_empty_reason('all', SimpleNamespace(min_mistakenness=0.5), fake)
    assert reason == 'item scores never computed'


@pytest.mark.asyncio
async def test_compute_empty_reason_no_unclassified_proposals() -> None:
    fake = AsyncMock()
    fake.count = AsyncMock(return_value={'count': 0})
    reason = await rer.compute_empty_reason(
        'new_class_proposals', SimpleNamespace(min_mistakenness=None), fake
    )
    assert reason == 'no unclassified proposals'


@pytest.mark.asyncio
async def test_compute_empty_reason_default_no_items_match() -> None:
    fake = AsyncMock()
    fake.count = AsyncMock(return_value={'count': 0})
    reason = await rer.compute_empty_reason('all', SimpleNamespace(min_mistakenness=None), fake)
    assert reason == 'no items match'


@pytest.mark.asyncio
async def test_review_tabs_empty_state_flags() -> None:
    fake = AsyncMock()
    fake.count = AsyncMock(side_effect=[{'count': 3}, {'count': 0}])
    state = await rer.review_tabs_empty_state(fake)
    assert state == {'has_probe_predictions': True, 'has_item_scores': False}


def test_review_queue_serves_empty_reason_when_zero_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = AsyncMock()
    fake.search = AsyncMock(return_value={'hits': {'total': {'value': 0}, 'hits': []}})
    fake.count = AsyncMock(return_value={'count': 0})
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    r = _client(fake).get('/curation/review/uncertainty')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['total'] == 0
    assert body['empty_reason'] == 'no probe predictions — run a probe'


def test_review_queue_empty_reason_is_null_when_items_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = AsyncMock()
    fake.search = AsyncMock(
        return_value={
            'hits': {
                'total': {'value': 1},
                'hits': [{'_id': 'c1', '_source': {'crop_id': 'c1'}}],
            }
        }
    )
    fake.count = AsyncMock(return_value={'count': 0})
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    r = _client(fake).get('/curation/review/all')
    assert r.status_code == 200, r.text
    assert r.json()['empty_reason'] is None


def test_review_tabs_serves_empty_state_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = AsyncMock()
    fake.count = AsyncMock(side_effect=[{'count': 0}, {'count': 7}])
    r = _client(fake).get('/curation/review/tabs')
    assert r.status_code == 200, r.text
    assert r.json()['empty_state'] == {'has_probe_predictions': False, 'has_item_scores': True}
