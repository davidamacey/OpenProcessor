"""The project-less ``{prefix}/health`` probes the deployment's ``env`` VLM
endpoint with nothing bound (found live: it reported
``no project is bound`` for a healthy vLLM)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.routers.curation import health
from src.services.labeling import vlm_endpoints, vlm_factory
from src.services.labeling.vlm_endpoints import VlmEndpoint, VlmEndpointBody


class _Labeler:
    async def health(self) -> Any:
        return SimpleNamespace(reachable=True, model='served-model', last_error=None)


class _Store:
    async def ensure_fresh(self, _client: Any) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.unbound
async def test_global_vlm_status_needs_no_bound_project(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = VlmEndpoint(
        name='env',
        source='env',
        revision=None,
        body=VlmEndpointBody(base_url='http://vlm:8000/v1', model='served-model'),
        etag='vlm:env:test',
    )
    monkeypatch.setattr(vlm_endpoints, 'env_builtin', lambda: endpoint)
    monkeypatch.setattr('src.services.config_store.get_global_config_store', _Store)
    monkeypatch.setattr(vlm_factory, 'labeler_for', lambda _endpoint, _pack: _Labeler())

    status = await health.vlm_status(object(), scoped=False)

    assert status == {'reachable': True, 'model': 'served-model'}
