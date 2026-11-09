"""The ONNX hash of an engine row is the hash of the file the exporter leaves for it:
``models/{name}_end2end/1/model.onnx`` for both head kinds (``export/ultralytics_export.py``)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'export'))

import export_tasks  # follows the sys.path setup


@pytest.mark.parametrize('engine', ['car_trt_end2end', 'car_trt'])
def test_both_engine_kinds_read_the_onnx_of_their_model(
    engine: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(export_tasks, 'MODELS_DIR', tmp_path)
    onnx = tmp_path / 'car_end2end' / '1' / 'model.onnx'
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(b'fused onnx')

    assert export_tasks._onnx_sha256(engine) == hashlib.sha256(b'fused onnx').hexdigest()


def test_an_engine_without_its_onnx_has_no_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(export_tasks, 'MODELS_DIR', tmp_path)

    assert export_tasks._onnx_sha256('car_trt_end2end') is None
