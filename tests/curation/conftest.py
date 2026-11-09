"""Fixtures shared by the VLM endpoint API tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest
from config_store.memory_settings import revision_aware
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fixtures.job_fixtures import kinds, producer  # noqa: F401  # fixture registration: a start queues its job on a fake producer

import src.services.labeling.vlm_url_policy as policy
from src.core.db_session import get_db
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


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """No curation test dials out. A pinned VLM client connects to the vetted
    (fake) address instead of failing on an unresolvable name, so without this
    a stub address such as 10.0.0.7 would be a real TCP attempt: instant on a
    host that rejects it, minutes on one that drops the packets. Tests that
    exercise the connect path override this fixture."""
    import httpcore

    async def refuse(self: Any, host: str, port: int, **_kw: Any) -> Any:
        raise httpcore.ConnectError(f'no network in tests ({host}:{port})')

    monkeypatch.setattr('httpcore.AnyIOBackend.connect_tcp', refuse)


@pytest.fixture
def vlm_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> VlmApi:
    from _curation_app import API as _API

    from src.routers.curation import _raw_opensearch_dep, vlm_catalog, vlm_endpoints
    from src.routers.curation._mounting import curation_scoped_routers, mount_curation
    from src.routers.curation._project_deps import install_project_exception_handlers
    from src.routers.curation.projects import global_router

    assert _API == API
    assert vlm_catalog  # registers /vlm/catalog and /vlm/local* on the global router
    dns = dict(DEFAULT_DNS)
    monkeypatch.setattr(policy, '_resolve', lambda host: list(dns.get(host, [])))
    monkeypatch.setattr(policy, '_docker_gateway_addresses', frozenset)
    policy.reset_policy_caches()
    monkeypatch.setenv('OP_VLM_SECRETS_DIR', str(tmp_path / 'secrets'))
    monkeypatch.delenv('OP_VLM_URL', raising=False)
    # A job started through the routes writes its state under this dir.
    import src.config.curation_accessors as curation_config_mod

    monkeypatch.setenv('OP_AUTO_LABEL_STATE_DIR', str(tmp_path / 'auto_label'))
    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    monkeypatch.delenv('OP_VLM_EXTERNAL_POLICY', raising=False)
    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )

    app = FastAPI()
    mount_curation(
        app,
        api_prefix=API,
        global_router=global_router,
        scoped_routers=list(curation_scoped_routers()),
    )
    install_project_exception_handlers(app)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: None
    # No database in these tests: a route that needs a session gets none and must not touch it.
    app.dependency_overrides[get_db] = lambda: None
    api = VlmApi(
        client=revision_aware(TestClient(app)),
        lenient=revision_aware(TestClient(app, raise_server_exceptions=False)),
        dns=dns,
    )

    async def scripted_probe(
        body: Any, *, api_key: Any, client: Any = None, is_env: bool = False
    ) -> VlmProbeRecord:
        api.probe_keys.append(api_key)
        return api.probe_record[0]

    monkeypatch.setattr(vlm_endpoints, 'probe_endpoint', scripted_probe)
    return api


@pytest.fixture
def stack(vlm_api: VlmApi, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The test-on-crop stack: the app, an item store, a fake VLM and
    a fake segmenter. See ``curation.config_test_stack``."""
    from curation.config_test_stack import build_stack

    return build_stack(vlm_api, tmp_path, monkeypatch)


@pytest.fixture
def readable_crops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every crop image reads as a tiny JPEG, for tests whose items have no file
    behind them but must reach the VLM (a stage whose crops are all unreadable fails)."""
    from types import SimpleNamespace

    # The images router binds THUMBNAIL_CACHE by name at import; import it first so a first
    # import under the patch cannot keep the stand-in for every later test in the process.
    import src.routers.curation.images  # noqa: F401 -- imported for its binding, not used
    from src.services.curation import image_serving

    monkeypatch.setattr(image_serving, 'resolve_crop_root', lambda _p: '/data')
    monkeypatch.setattr(image_serving, 'resolve_safe_path', lambda p, _r: p)
    monkeypatch.setattr(
        image_serving, 'THUMBNAIL_CACHE', SimpleNamespace(get_or_compute=lambda *_a, **_k: b'jpeg')
    )


@pytest.fixture
def label_ops_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    """A temp state dir for a test whose bulk label or move records an operation
    (the record is written under the bound project's ``project_state_dir``)."""
    import src.config.curation_accessors as curation_config_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    return tmp_path / 'state'


@pytest.fixture(autouse=True)
def _ingest_test_detector_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Names for the output indices of the fake ingest detector ``primary_end2end``
    (``ingest_fakes``): a detector that assigns classes maps its output to the registry by name,
    and the fake registries name ids 0 and 1 ``gadget`` and ``widget``. Any other model keeps
    its real label lookup."""
    from src.config import ingest_profiles

    real = ingest_profiles.get_class_names

    def names_for(model_name: str) -> dict[int, str]:
        return {0: 'gadget', 1: 'widget'} if model_name == 'primary_end2end' else real(model_name)

    monkeypatch.setattr(ingest_profiles, 'get_class_names', names_for)


@pytest.fixture
def loop_env(pg: Any, tmp_path: Any, captured_events: Any) -> Any:
    """A project in the test database for the assist loop, with blobs under ``tmp_path``."""
    from curation.assist_loop_fakes import LoopEnv

    return LoopEnv.make(pg, tmp_path)


@pytest.fixture(autouse=True)
def _no_active_bakeoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """The GPU arbiter reads bake-off jobs from PostgreSQL; these tests have no database."""
    from src.services.training import gpu_arbiter

    async def _none() -> bool:
        return False

    monkeypatch.setattr(gpu_arbiter, '_bakeoff_holds_gpus', _none)


@pytest.fixture(autouse=True)
def _no_live_loop_hold_offline(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The auto-label refresh hold reads the job tables; the offline tier has none (a ``db``
    test, such as the assist-loop isolation one, runs the real check)."""
    if request.node.get_closest_marker('db') is not None:
        return
    from src.services.curation.assist_loop import service

    async def _no_loop(_session: Any) -> None:
        return None

    from src.routers.curation import pipeline_public

    monkeypatch.setattr(service, 'require_no_live_loop', _no_loop)
    monkeypatch.setattr(pipeline_public, 'require_no_live_loop', _no_loop)
