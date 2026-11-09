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

from unittest.mock import AsyncMock

import pytest
from config_store.memory_settings import revision_aware
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opensearchpy.exceptions import OpenSearchException


class _UnreachableOpenSearch:
    async def search(self, **_: object) -> dict:
        raise OpenSearchException('down')


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch, memory_settings) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = _UnreachableOpenSearch()
    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = revision_aware(TestClient(app))
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


@pytest.mark.asyncio
async def test_put_new_sort_default_flips_methods_but_never_overrides_a_tab_default(
    app_client: TestClient,
) -> None:
    """Setting a shared 'sort' default (a) flips GET /methods's default flag
    and (b) is what a tab *without* its own default sorts by — but a tab's
    own default sort always wins over it (product rule: the regions tab's
    region-score order must not be replaced by a deployment default)."""
    from src.services.curation import review_sorts

    r = app_client.put(
        '/curation/projects/default/settings', json={'defaults': {'sort': 'uncertainty_entropy'}}
    )
    assert r.status_code == 200

    methods_body = app_client.get('/curation/projects/default/methods').json()
    sort_entries = {s['id']: s for s in methods_body['strategies'] if s['axis'] == 'sort'}
    assert sort_entries['uncertainty_entropy']['default'] is True
    assert sort_entries['atypicality']['default'] is False

    _clause, applied, _ = await review_sorts.build_sort(
        None, tab='all', opensearch=app_client.fake_os
    )
    assert applied == 'atypicality'
    _clause, applied, _ = await review_sorts.build_sort(
        None, tab='new_class_proposals', opensearch=app_client.fake_os
    )
    assert applied == 'uncertainty_entropy'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
