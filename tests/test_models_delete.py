"""Found live: ``DELETE /models/{name}`` removed the TensorRT directories but
left the ONNX End2End intermediate the export had written (still listed by
Triton)."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
