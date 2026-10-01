"""Fixtures shared by the VLM endpoint API tests (W9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.services.labeling.vlm_url_policy as policy
from curation._fake_config_opensearch import FakeConfigOpenSearch
from src.services.labeling.vlm_endpoint_body import VlmProbeRecord


API = '/curation'
SCOPED = f'{API}/projects/default'
GLOBAL_VLM = f'{API}/vlm/endpoints'
ACTIVE = f'{SCOPED}/vlm/endpoints'

#: Name resolution for the tests below: a compose service, a LAN box and a
#: public host. Anything else does not resolve.
DEFAULT_DNS: dict[str, list[str]] = {
    'vlm': ['172.18.0.9'],
    'gpu.lan': ['192.168.1.20'],
    'api.example.com': ['93.184.216.34'],
}


def good_probe(**over: Any) -> VlmProbeRecord:
    fields: dict[str, Any] = {
        'ok': True,
        'probed_at': '2026-09-28T12:00:00+00:00',
        'latency_ms': 12.0,
        'models_listed': ['m'],
        'model_listed': True,
        'root': 'org/real-model',
        'max_model_len': 32768,
        'vision_ok': True,
        'json_mode_supported': True,
        'reasoning_channel': False,
        'image_tokens': 260,
        'max_images_ok': True,
        'issues': [],
    }
    fields.update(over)
    return VlmProbeRecord(**fields)


@dataclass
class VlmApi:
    """The real curation app (scoped + the global router) over an in-memory
    config store, with DNS pinned and the probe scripted."""

    client: TestClient
    #: The same app, but a server error is a 500 response, not an exception.
    lenient: TestClient
    fake_os: FakeConfigOpenSearch
    dns: dict[str, list[str]]
    probe_record: list[VlmProbeRecord] = field(default_factory=lambda: [good_probe()])
    #: The ``api_key`` each probe was given, in call order.
    probe_keys: list[str | None] = field(default_factory=list)

    def body(self, **over: Any) -> dict[str, Any]:
        return {'base_url': 'http://vlm:8000/v1', 'model': 'm', **over}

    def create(self, name: str, **body: Any) -> dict[str, Any]:
        response = self.client.post(
            GLOBAL_VLM, json={'name': name, 'body': self.body(**body), 'description': ''}
        )
        assert response.status_code == 201, response.text
        return response.json()

    def probe(self, name: str) -> dict[str, Any]:
        response = self.client.post(f'{GLOBAL_VLM}/{name}/probe')
        assert response.status_code == 200, response.text
        return response.json()

    def ready(self, name: str, **body: Any) -> dict[str, Any]:
        """Create ``name`` and record a healthy probe for it."""
        doc = self.create(name, **body)
        self.probe(name)
        return doc

    def activate(self, name: str, **payload: Any) -> Any:
        return self.client.post(f'{ACTIVE}/{name}/activate', json=payload)

    def active(self) -> dict[str, Any]:
        response = self.client.get(f'{ACTIVE}/active')
        assert response.status_code == 200, response.text
        return response.json()


@pytest.fixture
def vlm_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> VlmApi:
    from _curation_app import API as _API

    from src.routers.curation import _raw_opensearch_dep, vlm_catalog, vlm_endpoints
    from src.routers.curation._mounting import curation_scoped_routers, mount_curation
    from src.routers.curation._project_deps import install_project_exception_handlers
    from src.routers.curation.projects import global_router

    assert _API == API
    assert vlm_catalog  # registers /vlm/catalog and /vlm/local* on the global router
    fake_os = FakeConfigOpenSearch()
    dns = dict(DEFAULT_DNS)
    monkeypatch.setattr(policy, '_resolve', lambda host: list(dns.get(host, [])))
    monkeypatch.setattr(policy, '_docker_gateway_addresses', lambda: frozenset())
    policy.reset_policy_caches()
    monkeypatch.setenv('OP_VLM_SECRETS_DIR', str(tmp_path / 'secrets'))
    monkeypatch.delenv('OP_VLM_URL', raising=False)
    # A job started through the routes writes its state under this dir.
    import src.config.curation as curation_config_mod

    monkeypatch.setenv('OP_AUTO_LABEL_STATE_DIR', str(tmp_path / 'auto_label'))
    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    monkeypatch.delenv('OP_VLM_EXTERNAL_POLICY', raising=False)
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))

    app = FastAPI()
    mount_curation(
        app,
        api_prefix=API,
        global_router=global_router,
        scoped_routers=list(curation_scoped_routers()),
    )
    install_project_exception_handlers(app)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    api = VlmApi(
        client=TestClient(app),
        lenient=TestClient(app, raise_server_exceptions=False),
        fake_os=fake_os,
        dns=dns,
    )

    async def scripted_probe(body: Any, *, api_key: Any, client: Any = None) -> VlmProbeRecord:
        api.probe_keys.append(api_key)
        return api.probe_record[0]

    monkeypatch.setattr(vlm_endpoints, 'probe_endpoint', scripted_probe)
    return api


class HybridOpenSearch:
    """Config-store documents (a project's ``__configs`` index and the global
    registry) go to :class:`FakeConfigOpenSearch`; everything else (items,
    images, ...) to the query-evaluating item fake, so one client can serve a
    route that reads the endpoint registry AND writes items."""

    def __init__(self, cfg: FakeConfigOpenSearch, items: Any) -> None:
        self.cfg = cfg
        self.items = items
        self.indices = self

    def _pick(self, index: str) -> Any:
        return (
            self.cfg if index.endswith('__configs') or index == 'op_global_configs' else self.items
        )

    async def refresh(self, index: str, **_kw: Any) -> dict[str, Any]:
        return await self._pick(index).indices.refresh(index=index)

    async def get(self, index: str, id: str, **kw: Any) -> dict[str, Any]:  # noqa: A002
        return await self._pick(index).get(index=index, id=id, **kw)

    async def index(self, index: str, **kw: Any) -> dict[str, Any]:
        return await self._pick(index).index(index=index, **kw)

    async def update(self, index: str, **kw: Any) -> dict[str, Any]:
        return await self._pick(index).update(index=index, **kw)

    async def delete(self, index: str, **kw: Any) -> dict[str, Any]:
        return await self._pick(index).delete(index=index, **kw)

    async def search(self, index: str | None = None, **kw: Any) -> dict[str, Any]:
        return await self._pick(index or '').search(index=index, **kw)

    async def count(self, index: str, **kw: Any) -> dict[str, Any]:
        return await self._pick(index).count(index=index, **kw)

    async def mget(self, **kw: Any) -> dict[str, Any]:
        return await self.items.mget(**kw)

    async def bulk(self, **kw: Any) -> dict[str, Any]:
        return await self.items.bulk(**kw)

    async def exists(self, index: str, **_kw: Any) -> bool:
        del index
        return True
