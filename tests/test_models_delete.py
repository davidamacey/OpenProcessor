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
