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
    models: tuple[bool, str, list[dict[str, str]]] = (True, 'all required models READY', []),
) -> None:
    """Replace each per-service probe with a deterministic stub."""
    from src.routers import health as health_module

    async def _stub_triton() -> tuple[bool, str]:
        return triton

    async def _stub_http(url: str) -> tuple[bool, str]:
        return opensearch

    async def _stub_models() -> tuple[bool, str, list[dict[str, str]]]:
        return models

    async def _stub_dependency(name: str) -> tuple[bool, str]:
        return True, 'ok'

    monkeypatch.setattr(health_module, '_probe_dependency', _stub_dependency)
    monkeypatch.setattr(health_module, '_probe_triton', _stub_triton)
    monkeypatch.setattr(health_module, '_probe_http', _stub_http)
    monkeypatch.setattr(health_module, '_probe_required_models', _stub_models)


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


def test_cuda_block_is_named_for_what_it_measures(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API process's own CUDA allocation is not GPU telemetry, so it must
    not sit under a ``gpu`` key."""
    from src.routers import health as health_module

    _patch_probes(monkeypatch, triton=(True, 'ok'), opensearch=(True, 'ok'))
    monkeypatch.setattr(health_module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(health_module.torch.cuda, 'memory_allocated', lambda: 3 * 1024 * 1024)
    monkeypatch.setattr(health_module.torch.cuda, 'memory_reserved', lambda: 5 * 1024 * 1024)
    resources = client.get('/ready').json()['resources']
    assert 'gpu' not in resources
    assert resources['process_cuda_allocated_mb'] == 3.0
    assert resources['process_cuda_reserved_mb'] == 5.0


_PE_DOWN = [{'name': 'pe_image_encoder', 'state': 'UNAVAILABLE', 'reason': 'no model.plan'}]


@pytest.mark.parametrize('path', ['/ready', '/health'])
def test_not_ready_and_names_the_model_when_a_required_model_is_not_ready(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    _patch_probes(
        monkeypatch,
        triton=(True, 'is_server_live OK'),
        opensearch=(True, 'HTTP 200'),
        models=(False, 'not READY: pe_image_encoder', _PE_DOWN),
    )
    response = client.get(path)
    assert response.status_code == 503
    payload = response.json()
    assert payload['status'] == 'not_ready'
    assert payload['services']['triton']['ok'] is True
    assert payload['services']['models']['ok'] is False
    assert payload['services']['models']['not_ready'] == _PE_DOWN
    assert 'pe_image_encoder' in payload['services']['models']['detail']


def test_live_and_root_stay_200_when_a_model_is_not_ready(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_probes(
        monkeypatch,
        triton=(True, 'ok'),
        opensearch=(True, 'ok'),
        models=(False, 'not READY: pe_image_encoder', _PE_DOWN),
    )
    assert client.get('/live').status_code == 200
    assert client.get('/').status_code == 200


def test_models_are_not_checked_when_triton_is_down(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control-plane-only installs have no Triton: the body must still show
    OpenSearch up, and carry no model verdict the installer could misread."""
    from src.routers import health as health_module

    _patch_probes(monkeypatch, triton=(False, 'ConnectError'), opensearch=(True, 'HTTP 200'))

    async def _boom() -> tuple[bool, str, list[dict[str, str]]]:
        raise AssertionError('models must not be probed without Triton')

    monkeypatch.setattr(health_module, '_probe_required_models', _boom)
    response = client.get('/health')
    assert response.status_code == 503
    assert '"opensearch":{"ok":true' in response.text.replace(' ', '')
    assert 'models' not in response.json()['services']


def _stub_face_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.routers import health as health_module
    from src.services.face_privacy import FaceDataCounts

    counts = FaceDataCounts(
        face_index_exists=True,
        face_docs=3,
        persons=2,
        named_persons=1,
        thumbnails=0,
        people_face_links=0,
        faiss_files=['faces.index'],
    )

    async def _stub() -> FaceDataCounts:
        return counts

    monkeypatch.setattr(health_module, '_face_data_counts', _stub)


def _triton_index_without_face_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services import triton_control

    names = [
        'yolov11_small_trt_end2end',
        'mobileclip2_s2_image_encoder',
        'mobileclip2_s2_text_encoder',
        'pe_image_encoder',
    ]

    async def _index(_self: object) -> list[dict[str, str]]:
        return [{'name': n, 'version': '1', 'state': 'READY', 'reason': ''} for n in names]

    monkeypatch.setattr(triton_control.TritonControlService, 'get_repository_index', _index)


def _patch_infra_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.routers import health as health_module

    async def _live() -> tuple[bool, str]:
        return True, 'is_server_live OK'

    async def _http(_url: str) -> tuple[bool, str]:
        return True, 'HTTP 200'

    async def _dependency(_name: str) -> tuple[bool, str]:
        return True, 'ok'

    monkeypatch.setattr(health_module, '_probe_dependency', _dependency)
    monkeypatch.setattr(health_module, '_probe_triton', _live)
    monkeypatch.setattr(health_module, '_probe_http', _http)


def test_ready_ignores_face_models_and_reports_stored_data_when_faces_off(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_REQUIRED_MODELS', raising=False)
    monkeypatch.setenv('OP_FACES_ENABLED', 'false')
    _patch_infra_only(monkeypatch)
    _triton_index_without_face_models(monkeypatch)
    _stub_face_counts(monkeypatch)
    response = client.get('/ready')
    assert response.status_code == 200
    faces = response.json()['services']['faces']
    assert faces['enabled'] is False
    assert faces['stored']['face_docs'] == 3
    assert 'faces purge' in faces['detail']


def test_ready_requires_face_models_when_faces_on(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_REQUIRED_MODELS', raising=False)
    monkeypatch.delenv('OP_FACES_ENABLED', raising=False)
    _patch_infra_only(monkeypatch)
    _triton_index_without_face_models(monkeypatch)
    response = client.get('/ready')
    assert response.status_code == 503
    assert 'scrfd_10g_bnkps' in response.json()['services']['models']['detail']
    assert response.json()['services']['faces'] == {'enabled': True}
