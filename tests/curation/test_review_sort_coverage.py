"""Review sorts whose backing field no item carries.

A tab's default (or the deployment-pinned) sort on a field with 0%
coverage orders nothing: every item is "missing" and the queue falls back
to crop_id order silently. The review queue must instead fall back to the
tab's next sensible sort and say so (``sort_fallback_reason``), and
``PUT /settings`` must refuse to pin such a sort as the deployment default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from curation.test_curation_settings_client import FakeSettingsOpenSearch
from src.config import get_curation_config


if TYPE_CHECKING:
    from collections.abc import Iterator


ITEMS = get_curation_config().items_index


@pytest.fixture(autouse=True)
def _fresh_coverage(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from src.services.curation.strategy_registry import _reset_field_coverage_cache

    monkeypatch.delenv('OP_SCORES_ENABLED', raising=False)
    monkeypatch.delenv('OP_SCORES_SHADOW', raising=False)
    _reset_field_coverage_cache()
    yield
    _reset_field_coverage_cache()


def _item(crop_id: str, **fields: Any) -> dict[str, Any]:
    return {'crop_id': crop_id, 'updated_at': '2026-09-01T00:00:00+00:00', **fields}


class _CoverageOpenSearch:
    """``count`` answers exists-queries from a fixed per-field table."""

    def __init__(self, coverage: dict[str, int], total: int = 10) -> None:
        self.coverage = coverage
        self.total = total

    async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        query = body['query']
        if 'exists' in query:
            return {'count': self.coverage.get(query['exists']['field'], 0)}
        return {'count': self.total}

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        """F-28: coverage is one ``size: 0`` search with a ``filter: exists``
        sub-agg per field (plus ``track_total_hits`` for the denominator)."""
        aggs = {
            name: {'doc_count': self.coverage.get(agg['filter']['exists']['field'], 0)}
            for name, agg in (body.get('aggs') or {}).items()
        }
        return {'hits': {'total': {'value': self.total}}, 'aggregations': aggs}


@pytest.mark.asyncio
async def test_zero_coverage_tab_default_falls_back_with_reason() -> None:
    from src.services.curation import review_sorts

    os_ = _CoverageOpenSearch({'cluster_distance': 5})
    clause, applied, reason = await review_sorts.build_sort(None, tab='uncertainty', opensearch=os_)
    assert applied == 'atypicality'
    assert clause == review_sorts.get_review_sorts()['atypicality'].clause
    assert reason is not None
    assert 'uncertainty_entropy' in reason
    assert 'probe_pred_entropy' in reason


@pytest.mark.asyncio
async def test_fallback_chain_ends_at_recent() -> None:
    from src.services.curation import review_sorts

    os_ = _CoverageOpenSearch({})
    for tab in ('all', 'outliers', 'uncertainty', 'model_disagreements', 'regions'):
        _clause, applied, reason = await review_sorts.build_sort(None, tab=tab, opensearch=os_)
        assert applied == 'recent', tab
        assert reason, tab


@pytest.mark.asyncio
async def test_covered_default_is_kept() -> None:
    from src.services.curation import review_sorts

    os_ = _CoverageOpenSearch({'probe_pred_entropy': 1})
    _clause, applied, reason = await review_sorts.build_sort(
        None, tab='uncertainty', opensearch=os_
    )
    assert applied == 'uncertainty_entropy'
    assert reason is None


@pytest.mark.asyncio
async def test_unknown_coverage_never_triggers_a_fallback() -> None:
    """A failing coverage count is 'unknown', not 'zero'."""
    from src.services.curation import review_sorts

    class _Broken:
        async def count(self, **_kw: Any) -> dict[str, Any]:
            raise RuntimeError('down')

    _clause, applied, reason = await review_sorts.build_sort(
        None, tab='uncertainty', opensearch=_Broken()
    )
    assert applied == 'uncertainty_entropy'
    assert reason is None


@pytest.mark.asyncio
async def test_zero_coverage_pinned_default_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation import review_sorts

    monkeypatch.setattr(
        'src.services.curation.strategy_registry.resolve_effective_default',
        AsyncMock(return_value='uncertainty_entropy'),
    )
    _clause, applied, reason = await review_sorts.build_sort(
        None, tab='new_class_proposals', opensearch=_CoverageOpenSearch({})
    )
    assert applied == 'recent'
    assert reason is not None
    assert 'uncertainty_entropy' in reason


@pytest.mark.asyncio
async def test_explicit_sort_is_honored_even_at_zero_coverage() -> None:
    from src.services.curation import review_sorts

    _clause, applied, reason = await review_sorts.build_sort(
        'uncertainty_entropy', tab='all', opensearch=_CoverageOpenSearch({})
    )
    assert applied == 'uncertainty_entropy'
    assert reason is None


@pytest.fixture
def review_client(monkeypatch: pytest.MonkeyPatch):
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake = QueryFakeOpenSearch(
        {ITEMS: {'a': _item('a', cluster_distance=0.3), 'b': _item('b', cluster_distance=0.1)}}
    )
    for mod in ('review', 'settings'):
        monkeypatch.setattr(
            f'src.routers.curation.{mod}._ensure_indexes', AsyncMock(return_value=None)
        )
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def test_review_route_reports_the_fallback(review_client: TestClient) -> None:
    r = review_client.get('/curation/review/uncertainty')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['sort_applied'] == 'atypicality'
    assert 'uncertainty_entropy' in body['sort_fallback_reason']

    r = review_client.get('/curation/review/uncertainty/locate', params={'crop_id': 'a'})
    assert r.status_code == 200, r.text
    assert r.json()['sort_applied'] == 'atypicality'
    assert 'uncertainty_entropy' in r.json()['sort_fallback_reason']


class _SettingsWithCoverage(FakeSettingsOpenSearch):
    def __init__(self, coverage: dict[str, int]) -> None:
        super().__init__()
        self._cov = _CoverageOpenSearch(coverage)

    async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._cov.count(index=index, body=body)

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
        aggs = body.get('aggs') or {}
        if aggs and all('filter' in a for a in aggs.values()):
            return await self._cov.search(index=index, body=body)
        return await super().search(index=index, body=body, **kw)


def _settings_client(monkeypatch: pytest.MonkeyPatch, fake: Any) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    for mod in ('review', 'settings'):
        monkeypatch.setattr(
            f'src.routers.curation.{mod}._ensure_indexes', AsyncMock(return_value=None)
        )
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def test_put_refuses_a_zero_coverage_sort_default(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _SettingsWithCoverage({})
    client = _settings_client(monkeypatch, fake)
    r = client.put('/curation/settings', json={'defaults': {'sort': 'uncertainty_entropy'}})
    assert r.status_code == 422, r.text
    assert 'probe_pred_entropy' in r.text
    assert client.get('/curation/settings').json()['defaults'] == {}

    # A sort on a field every item carries is always accepted.
    r = client.put('/curation/settings', json={'defaults': {'sort': 'recent'}})
    assert r.status_code == 200, r.text


def test_put_accepts_a_covered_sort_default(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _SettingsWithCoverage({'probe_pred_entropy': 3})
    client = _settings_client(monkeypatch, fake)
    r = client.put('/curation/settings', json={'defaults': {'sort': 'uncertainty_entropy'}})
    assert r.status_code == 200, r.text
    assert r.json()['defaults'] == {'sort': 'uncertainty_entropy'}
