"""Found live: ``DELETE /models/{name}`` removed the TensorRT directories but
left the ONNX End2End intermediate the export had written (still listed by
Triton)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers import models as models_router


if TYPE_CHECKING:
    from pathlib import Path


class _Triton:
    async def unload_model(self, _name: str) -> tuple[bool, str]:
        return True, ''


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(models_router, 'TRITON_MODELS_DIR', tmp_path / 'models')
    monkeypatch.setattr(models_router, 'PYTORCH_MODELS_DIR', tmp_path / 'pytorch_models')
    monkeypatch.setattr(models_router, 'TritonControlService', _Triton)
    app = FastAPI()
    app.include_router(models_router.router)
    return TestClient(app)


def test_delete_removes_every_directory_the_export_wrote(
    client: TestClient, tmp_path: Path
) -> None:
    for name in ('m_trt_end2end', 'm_end2end', 'other_trt'):
        (tmp_path / 'models' / name).mkdir(parents=True)
        (tmp_path / 'models' / name / 'config.pbtxt').write_text('x')

    resp = client.delete(f'{models_router.router.prefix}/m')

    assert resp.status_code == 200, resp.text
    assert sorted(p.name for p in (tmp_path / 'models').iterdir()) == ['other_trt']


def _dirs(tmp_path: Path, *names: str) -> None:
    for name in names:
        (tmp_path / 'models' / name).mkdir(parents=True)
        (tmp_path / 'models' / name / 'config.pbtxt').write_text('x')


def _names(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in (tmp_path / 'models').iterdir())


def test_delete_refuses_the_configured_detector_even_with_force(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Found live: ``DELETE /models/yolov11_small`` removed the primary
    detector's files with no guard."""
    monkeypatch.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'm_trt_end2end')
    _dirs(tmp_path, 'm_trt_end2end', 'm_end2end')
    (tmp_path / 'pytorch_models').mkdir()
    (tmp_path / 'pytorch_models' / 'm.pt').write_text('w')
    for params in ({}, {'force': 'true'}):
        resp = client.delete(f'{models_router.router.prefix}/m', params=params)
        assert resp.status_code == 403, resp.text
    assert _names(tmp_path) == ['m_end2end', 'm_trt_end2end']  # nothing was removed
    assert (tmp_path / 'pytorch_models' / 'm.pt').exists()


def test_delete_of_a_core_model_needs_force(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config.settings import TritonModelConfig

    monkeypatch.setattr(TritonModelConfig, 'FACE_DETECT_MODEL', 'face_x_trt')
    _dirs(tmp_path, 'face_x_trt')
    refused = client.delete(f'{models_router.router.prefix}/face_x')
    assert refused.status_code == 409, refused.text
    assert _names(tmp_path) == ['face_x_trt']
    forced = client.delete(f'{models_router.router.prefix}/face_x', params={'force': 'true'})
    assert forced.status_code == 200, forced.text
    assert _names(tmp_path) == []


def test_unload_refuses_a_region_protected_model_even_with_force(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Found in review: ``POST /models/{name}/unload`` skipped the guard that
    ``DELETE`` applies, so the OCR models could be unloaded with one POST."""
    unload = AsyncMock(return_value=(True, ''))
    monkeypatch.setattr(_Triton, 'unload_model', unload)
    for name in ('paddleocr_det_trt', 'paddleocr_rec_trt'):
        for params in ({}, {'force': 'true'}):
            resp = client.post(f'{models_router.router.prefix}/{name}/unload', params=params)
            assert resp.status_code == 403, resp.text
    unload.assert_not_awaited()


def test_unload_of_a_core_model_needs_force(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config.settings import TritonModelConfig

    unload = AsyncMock(return_value=(True, ''))
    monkeypatch.setattr(_Triton, 'unload_model', unload)
    url = f'{models_router.router.prefix}/{TritonModelConfig.FACE_DETECT_MODEL}/unload'
    assert client.post(url).status_code == 409
    unload.assert_not_awaited()
    assert client.post(url, params={'force': 'true'}).status_code == 200
    unload.assert_awaited_once()


def _promoted(tmp_path: Path, name: str, owner: str) -> None:
    _dirs(tmp_path, name)
    (tmp_path / 'models' / name / 'promote.json').write_text(f'{{"project": "{owner}"}}')


def test_global_delete_of_a_project_promoted_model_is_a_typed_409_naming_the_owner(
    client: TestClient, tmp_path: Path
) -> None:
    """GH #86: a model /models/status lists READY (a project promote, whose name
    has no export suffix) used to 404 here. It is owned by a project, whose own
    route carries the in-use / sharing guards this one cannot, so it is refused
    with the route to use."""
    _promoted(tmp_path, 'alpha__wheels_v1', 'alpha')
    resp = client.delete(f'{models_router.router.prefix}/alpha__wheels_v1')
    assert resp.status_code == 409, resp.text
    detail = resp.json()['detail']
    assert detail['error'] == 'project_owned_model'
    assert detail['owner_project'] == 'alpha'
    assert 'alpha' in detail['message']
    assert _names(tmp_path) == ['alpha__wheels_v1']


def test_global_delete_addresses_an_exact_directory_name(
    client: TestClient, tmp_path: Path
) -> None:
    """Every name /models/status shows is addressable: the exact directory,
    not only ``{name}`` + an export suffix."""
    _dirs(tmp_path, 'm_trt_end2end', 'other_trt')
    resp = client.delete(f'{models_router.router.prefix}/m_trt_end2end')
    assert resp.status_code == 200, resp.text
    assert _names(tmp_path) == ['other_trt']


def test_global_load_resolves_the_same_names(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    load = AsyncMock(return_value=(True, 'ok'))
    monkeypatch.setattr(_Triton, 'load_model', load, raising=False)
    _promoted(tmp_path, 'alpha__wheels_v1', 'alpha')
    _dirs(tmp_path, 'm_trt')
    assert client.post(f'{models_router.router.prefix}/alpha__wheels_v1/load').status_code == 200
    assert client.post(f'{models_router.router.prefix}/m/load').json()['model_name'] == 'm_trt'
