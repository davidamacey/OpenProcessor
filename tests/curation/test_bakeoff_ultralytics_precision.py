"""The bake-off tells Ultralytics its precision with ``quantize``, never the deprecated ``half``.

Ultralytics 8.4 export/predict docs: ``quantize`` "replaces the deprecated ``half``/``int8``
flags" (https://docs.ultralytics.com/modes/export/, https://docs.ultralytics.com/modes/predict/).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest


class _FakeYolo:
    calls: list[dict[str, Any]] = []

    def __init__(self, _weights: str) -> None:
        self.names = {0: 'a'}

    def export(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return str(Path(kwargs['_out']))

    def predict(self, _image: Any, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        return []


@pytest.fixture
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> type[_FakeYolo]:
    _FakeYolo.calls = []
    module = ModuleType('ultralytics')
    module.YOLO = _FakeYolo  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'ultralytics', module)
    return _FakeYolo


@pytest.mark.parametrize(('half', 'quantize'), [(True, 16), (False, 32)])
def test_the_detector_predicts_with_quantize(
    fake_ultralytics: type[_FakeYolo], half: bool, quantize: int
) -> None:
    from scripts.curation.bakeoff.backends.ultralytics_pt import UltralyticsDetector

    detector = UltralyticsDetector('w.pt', device='cpu', half=half)
    detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

    (kwargs,) = fake_ultralytics.calls
    assert 'half' not in kwargs
    assert kwargs['quantize'] == quantize


@pytest.mark.parametrize(('half', 'quantize'), [(True, 16), (False, None)])
def test_the_onnx_export_uses_quantize(
    fake_ultralytics: type[_FakeYolo],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    half: bool,
    quantize: int | None,
) -> None:
    from scripts.curation.bakeoff import quantize as quantize_module

    out = tmp_path / 'm.onnx'
    out.write_bytes(b'x')
    real_export = _FakeYolo.export

    def export(self: _FakeYolo, **kwargs: Any) -> str:
        return real_export(self, _out=out, **kwargs)

    monkeypatch.setattr(_FakeYolo, 'export', export)
    monkeypatch.setitem(
        sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    )

    quantize_module._ultralytics_export_onnx(tmp_path / 'm.pt', out, 640, half=half)

    (kwargs,) = fake_ultralytics.calls
    assert 'half' not in kwargs
    assert kwargs.get('quantize') == quantize
