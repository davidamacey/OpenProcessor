"""/live, /ready, /health endpoint behavior pins.

* ``/live`` must return 200 with ``status='alive'`` regardless of
  dependency state (kept dependency-free so liveness never flaps on
  Triton / OpenSearch degradation).
* ``/ready`` must return 200 only when *every* dep probe succeeds, and
  503 with per-service detail when at least one fails.
* ``/health`` is preserved as a backward-compat alias of ``/ready``.

The dep probes are patched out so the tests run without a live
Triton/OpenSearch stack.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


def _build_app() -> FastAPI:
    """Build a minimal FastAPI app that mounts ONLY the health router.

    Avoids pulling in the full src.main lifespan (Triton client pool,
    OpenSearch warm-up) which a unit-level probe test must not require.
    """
    from src.routers.health import router as health_router

    app = FastAPI()
    app.include_router(health_router)
    return app


@pytest.fixture
def app() -> FastAPI:
    return _build_app()


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _patch_probes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    triton: tuple[bool, str],
    opensearch: tuple[bool, str],
) -> None:
    """Replace each per-service probe with a deterministic stub."""
    from src.routers import health as health_module

    async def _stub_triton() -> tuple[bool, str]:
        return triton

    async def _stub_http(url: str) -> tuple[bool, str]:
        return opensearch

    monkeypatch.setattr(health_module, '_probe_triton', _stub_triton)
    monkeypatch.setattr(health_module, '_probe_http', _stub_http)


def test_live_returns_200_even_when_deps_down(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Liveness must NOT be coupled to dependency state."""
    _patch_probes(
        monkeypatch,
        triton=(False, 'is_server_live timeout after 2.0s'),
        opensearch=(False, 'timeout after 2.0s'),
    )
    response = client.get('/live')
    assert response.status_code == 200
    assert response.json() == {'status': 'alive'}


def test_ready_returns_200_when_all_deps_up(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_probes(
        monkeypatch,
        triton=(True, 'is_server_live OK'),
        opensearch=(True, 'HTTP 200'),
    )
    response = client.get('/ready')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ready'
    assert payload['services']['triton']['ok'] is True
    assert payload['services']['opensearch']['ok'] is True


@pytest.mark.parametrize(
    ('triton_ok', 'opensearch_ok', 'down_service'),
    [
        (False, True, 'triton'),
        (True, False, 'opensearch'),
        (False, False, 'triton'),
    ],
)
def test_ready_returns_503_with_detail_when_a_dep_is_down(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    triton_ok: bool,
    opensearch_ok: bool,
    down_service: str,
) -> None:
    _patch_probes(
        monkeypatch,
        triton=(triton_ok, 'is_server_live OK' if triton_ok else 'ConnectError: refused'),
        opensearch=(opensearch_ok, 'HTTP 200' if opensearch_ok else 'timeout after 2.0s'),
    )
    response = client.get('/ready')
    assert response.status_code == 503
    payload = response.json()
    assert payload['status'] == 'not_ready'
    assert payload['services'][down_service]['ok'] is False
    # Per-service detail string must carry the failure reason.
    assert payload['services'][down_service]['detail']


def test_health_is_alias_of_ready(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_probes(
        monkeypatch,
        triton=(True, 'is_server_live OK'),
        opensearch=(False, 'timeout after 2.0s'),
    )
    ready = client.get('/ready')
    health = client.get('/health')
    assert health.status_code == ready.status_code == 503
    assert health.json()['services'].keys() == ready.json()['services'].keys()
