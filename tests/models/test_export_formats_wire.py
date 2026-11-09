"""The export formats on the wire are ``onnx``, ``trt`` and ``all``.

``onnx_end2end`` and ``trt_end2end`` named the EfficientNMS toolchain (an ONNX with plugin
operators, then an engine built from it); every family now exports through the one fused path
(``export/ultralytics_export.py``), so there is nothing for them to select. ``normalize_boxes``
went with the plugin: the fused tensor is in letterboxed input pixels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.routers import models_export
from src.schemas.model_admin import ExportRequest
from src.schemas.models import ExportFormat


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    from src.app.factory import create_app

    async def listening() -> None:
        """A consumer exists: only the request itself is under test."""

    monkeypatch.setattr(models_export, '_require_worker', listening)
    monkeypatch.setattr(models_export, 'PYTORCH_MODELS_DIR', tmp_path / 'pytorch_models')
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.mark.parametrize('removed', ['onnx_end2end', 'trt_end2end'])
def test_a_removed_format_is_a_422_on_the_export_route(client: TestClient, removed: str) -> None:
    resp = client.post('/models/anything/export', json={'formats': [removed]})

    assert resp.status_code == 422, resp.text
    assert removed in resp.text


@pytest.mark.parametrize('removed', ['onnx_end2end', 'trt_end2end'])
def test_a_removed_format_is_a_422_on_upload(client: TestClient, removed: str) -> None:
    resp = client.post(
        '/models/upload',
        files={'file': ('car.pt', b'x' * 2000, 'application/octet-stream')},
        data={'formats': removed},
    )

    assert resp.status_code == 422, resp.text


@pytest.mark.parametrize('kept', ['onnx', 'trt', 'all'])
def test_the_kept_formats_pass_validation(client: TestClient, kept: str) -> None:
    """Control: the request is valid and reaches the handler, which answers 404 (no weights)."""
    resp = client.post('/models/anything/export', json={'formats': [kept]})

    assert resp.status_code == 404, resp.text


def test_the_request_defaults_to_the_engine_and_has_no_normalize_boxes() -> None:
    request = ExportRequest()

    assert request.formats == [ExportFormat.TRT]
    assert 'normalize_boxes' not in ExportRequest.model_fields
    with pytest.raises(ValidationError):
        ExportRequest(formats=['trt_end2end'])  # type: ignore[list-item]
