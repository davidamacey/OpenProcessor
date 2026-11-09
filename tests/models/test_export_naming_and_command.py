"""Names become files, model directories and the export script's ``pt:name:batch`` argument, so
only a slug passes (the export command itself: tests/export/test_export_worker_command.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from src.services import model_export, model_export_jobs


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    from src.app.factory import create_app

    monkeypatch.setattr(model_export_jobs, '_consumer_listens', lambda: True)
    monkeypatch.setattr(model_export, 'PYTORCH_MODELS_DIR', tmp_path / 'pytorch_models')
    return TestClient(create_app(), raise_server_exceptions=False)


def _pt(name: str = 'car.pt') -> dict[str, tuple[str, bytes, str]]:
    return {'file': (name, b'x' * 2000, 'application/octet-stream')}


@pytest.mark.parametrize('bad', ['../../x', 'a:b', 'x/y', 'c*', '_lead', 'a' * 65])
def test_upload_refuses_a_triton_name_that_is_not_a_slug(
    client: TestClient, tmp_path: Path, bad: str
) -> None:
    resp = client.post('/models/upload', files=_pt(), data={'triton_name': bad})
    assert resp.status_code == 400, resp.text
    assert 'name' in resp.json()['detail']
    assert not (tmp_path / 'pytorch_models').exists()


def test_upload_refuses_a_file_name_that_does_not_make_a_slug(client: TestClient) -> None:
    resp = client.post('/models/upload', files=_pt('a:b.pt'))
    assert resp.status_code == 400, resp.text


@pytest.mark.parametrize('bad', ['c*', 'a:b', 'car[1]'])
def test_export_route_refuses_a_model_name_that_is_not_a_slug(client: TestClient, bad: str) -> None:
    resp = client.post(f'/models/{bad}/export', json={})
    assert resp.status_code == 400, resp.text


def test_triton_names_are_normalised_then_checked() -> None:
    assert model_export.generate_triton_name('My-Model best.pt') == 'my_model'
    assert model_export.generate_triton_name('car_best.pt') == 'car'
    assert model_export.generate_triton_name('x.pt', 'Shop-Cam 2') == 'shop_cam_2'
    with pytest.raises(ValueError, match='name'):
        model_export.generate_triton_name('x.pt', '../x')
