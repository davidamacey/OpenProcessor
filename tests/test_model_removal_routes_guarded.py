"""Every route that removes or unloads a Triton model must go through the one
shared ``check_unload`` guard. The routes are discovered from the real app, so
a new unload/delete route added without the guard fails here."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _route_helpers import api_routes
from fastapi.testclient import TestClient

from src.main import create_app
from src.routers.curation._common import _raw_opensearch_dep


def _removal_routes() -> list[tuple[str, str]]:
    return sorted(
        (method, route.path)
        for route in api_routes(create_app())
        if '{model_name}' in route.path
        for method in route.methods
        if method == 'DELETE' or route.path.endswith('/unload')
    )


REMOVAL_ROUTES = _removal_routes()


def test_discovery_finds_the_known_removal_routes() -> None:
    paths = {path for _, path in REMOVAL_ROUTES}
    assert '/models/{model_name}/unload' in paths
    assert '/models/{model_name}' in paths
    assert '/curation/projects/{project}/models/{model_name}' in paths


@pytest.mark.parametrize(('method', 'path'), REMOVAL_ROUTES)
@pytest.mark.parametrize('force', [False, True])
def test_route_refuses_an_ocr_model_and_removes_nothing(
    method: str, path: str, force: bool, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from src.routers import models as public
    from src.services.training import triton_promote

    removers = [AsyncMock(return_value=(True, ''))]
    monkeypatch.setattr(public.TritonControlService, 'unload_model', removers[0])
    monkeypatch.setattr(public, 'TRITON_MODELS_DIR', tmp_path / 'models')
    (tmp_path / 'models' / 'paddleocr_det_trt').mkdir(parents=True)
    curation_unload = AsyncMock()
    monkeypatch.setattr('src.routers.curation.models.unload_triton_model', curation_unload)
    monkeypatch.setattr('src.routers.curation.models.project_owns_model', lambda _n: True)
    monkeypatch.setattr(triton_promote, 'unload_triton_model', curation_unload)
    removers.append(curation_unload)

    url = path.replace('{project}', 'default').replace('{model_name}', 'paddleocr_det')
    if path.endswith('/unload') or '/curation/' in path:
        url = url.replace('paddleocr_det', 'paddleocr_det_trt')
    app = create_app()
    # The project delete also reads the ingest policy; the OCR guard refuses
    # before that, so no OpenSearch is needed (or reachable) here.
    app.dependency_overrides[_raw_opensearch_dep] = lambda: object()
    resp = TestClient(app).request(method, url, params={'force': str(force).lower()})

    assert resp.status_code == 403, (path, resp.text)
    for remover in removers:
        remover.assert_not_awaited()
    assert (tmp_path / 'models' / 'paddleocr_det_trt').exists()
