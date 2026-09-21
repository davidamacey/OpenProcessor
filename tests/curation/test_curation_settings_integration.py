"""Integration proof for the curation deployment-settings feature: a
``PUT /curation/settings`` must change BOTH ``GET /methods``'s per-axis
``default`` flag AND the real endpoint that applies that axis's default
when a request omits the param -- proving the two can never drift because
both read through the same
``src.services.curation.strategy_defaults.resolve_effective_default``
against the same settings document.

Split out from ``test_curation_settings_router.py`` (CRUD/validation only)
because this file depends on the ``strategy_registry.py`` /
``review_sorts.py`` rewiring that derives ``/methods``'s defaults and the
real endpoints' omitted-param resolution from the settings document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.test_curation_settings_client import FakeSettingsOpenSearch


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset_field_coverage_cache() -> Iterator[None]:
    from src.services.curation.strategy_registry import _reset_field_coverage_cache

    _reset_field_coverage_cache()
    yield
    _reset_field_coverage_cache()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as legacy_router

    fake_os = FakeSettingsOpenSearch()
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    app.include_router(legacy_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


def test_methods_reflects_a_stored_cluster_override(app_client: TestClient) -> None:
    from src.services.curation.clustering.methods import DEFAULT_METHOD, available_methods

    non_default = next(m for m in available_methods() if m != DEFAULT_METHOD)
    app_client.put('/curation/settings', json={'defaults': {'cluster': non_default}})

    r = app_client.get('/curation/methods')
    body = r.json()
    cluster_entries = {s['id']: s for s in body['strategies'] if s['axis'] == 'cluster'}
    assert cluster_entries[non_default]['default'] is True
    assert cluster_entries[DEFAULT_METHOD]['default'] is False


@pytest.mark.asyncio
async def test_put_new_default_changes_both_methods_and_real_endpoint_behavior(
    app_client: TestClient,
) -> None:
    """The point of the feature: setting a shared default for the 'sort'
    axis must (a) flip GET /methods's default flag for that entry AND (b)
    change what the real review endpoint does when ?sort is omitted -- the
    two can never drift because both read through
    resolve_effective_default against the same settings document."""
    from src.services.curation import review_sorts

    # Before: 'all' tab's own hardcoded default applies, 'uncertainty_entropy'
    # is not it.
    clause_before, applied_before, _ = await review_sorts.build_sort(
        None, tab='all', opensearch=app_client.fake_os
    )
    assert applied_before == 'atypicality'

    r = app_client.put('/curation/settings', json={'defaults': {'sort': 'uncertainty_entropy'}})
    assert r.status_code == 200

    methods_body = app_client.get('/curation/methods').json()
    sort_entries = {s['id']: s for s in methods_body['strategies'] if s['axis'] == 'sort'}
    assert sort_entries['uncertainty_entropy']['default'] is True
    assert sort_entries['atypicality']['default'] is False

    clause_after, applied_after, _ = await review_sorts.build_sort(
        None, tab='all', opensearch=app_client.fake_os
    )
    assert applied_after == 'uncertainty_entropy'
    assert clause_after != clause_before


@pytest.mark.asyncio
async def test_put_new_cluster_default_changes_real_cluster_residuals_call(
    app_client: TestClient,
) -> None:
    """Same proof as above, for the 'cluster' axis's real call site
    (``clustering.orchestrator.cluster_residuals``) rather than 'sort'."""
    from unittest.mock import patch

    from src.services.curation.clustering import orchestrator
    from src.services.curation.clustering.methods import DEFAULT_METHOD, available_methods

    non_default = next(m for m in available_methods() if m != DEFAULT_METHOD)
    app_client.put('/curation/settings', json={'defaults': {'cluster': non_default}})

    with patch(
        'src.services.curation.clustering.embedding_reduce.fetch_residual_v6_embeddings_parallel',
        new=AsyncMock(return_value=([], [])),
    ):
        result = await orchestrator.cluster_residuals(app_client.fake_os, clustering_method=None)

    assert result['method'] == non_default


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
