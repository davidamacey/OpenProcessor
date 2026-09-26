"""
opfinal Prometheus alert (2026-09-26): ~300+ `paddleocr_rec_trt` Triton
BACKEND inference failures in 10 minutes under plain read-only /ocr and
/analyze traffic, while every HTTP response was still 200.

Root cause: `models/ocr_pipeline/1/model.py::_call_recognition` clamped
crop width to a local `MIN_WIDTH = 8`, but the actual TensorRT engine
(built by `export/export_paddleocr_rec.py`, `MIN_WIDTH = 48`) only accepts
widths in `[48, 2048]`. Any text crop narrower than 48px after the
height=48 aspect-ratio resize produced a real Triton binding-dimension
error:

    request specifies invalid shape for input 'x' for paddleocr_rec_trt_0_0.
    Error details: model expected the shape of dimension 3 to be between
    48 and 2048 but received 40

...which `_call_recognition` caught and silently converted into
`('', 0.0)` -- indistinguishable from "recognition ran and found nothing".
The API kept returning HTTP 200 with missing text for every narrow crop.

This test reproduces the bug directly against `_call_recognition` (no
Triton/GPU needed -- the recognition call is mocked) by driving a crop
whose aspect ratio maps to a target width below the true engine minimum,
and asserts:
  1. the pre-fix `MIN_WIDTH = 8` would have sent an out-of-range width to
     the (mocked) engine, i.e. the width is clamped to >= 48 post-fix;
  2. a genuine backend failure is never reported as score 0.0 (which is a
     legitimate low-confidence result) -- it must be a value real CTC
     scores can never produce, so callers can detect it.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest


if TYPE_CHECKING:
    from collections.abc import Callable


MODEL_PY = Path(__file__).resolve().parents[1] / 'models' / 'ocr_pipeline' / '1' / 'model.py'


class _FakeTensor:
    def __init__(self, name: str, array: np.ndarray) -> None:
        self.name = name
        self.array = array

    def as_numpy(self) -> np.ndarray:
        return self.array


class _FakeInferenceRequest:
    """Records the shape it was asked to run and returns a scripted response."""

    last_shape: tuple[int, ...] | None = None
    response_factory: Callable[[tuple[int, ...]], object] | None = None

    def __init__(self, model_name, requested_output_names, inputs):
        self.model_name = model_name
        self.requested_output_names = requested_output_names
        self.inputs = inputs
        _FakeInferenceRequest.last_shape = inputs[0].array.shape

    def exec(self):
        assert _FakeInferenceRequest.response_factory is not None
        return _FakeInferenceRequest.response_factory(self.inputs[0].array.shape)


def _make_fake_pb_utils() -> types.ModuleType:
    mod = types.ModuleType('triton_python_backend_utils')
    mod.Tensor = _FakeTensor  # type: ignore[attr-defined]
    mod.InferenceRequest = _FakeInferenceRequest  # type: ignore[attr-defined]
    mod.get_output_tensor_by_name = lambda response, _name: response  # type: ignore[attr-defined]
    mod.get_input_tensor_by_name = lambda _request, _name: None  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def model_module(monkeypatch: pytest.MonkeyPatch):
    """Import ocr_pipeline's model.py in isolation with a fake pb_utils."""
    fake_pb_utils = _make_fake_pb_utils()
    monkeypatch.setitem(sys.modules, 'triton_python_backend_utils', fake_pb_utils)
    # cupy/torch are optional (HAS_GPU=False path) -- force that path so
    # this test doesn't need a GPU.
    monkeypatch.setitem(sys.modules, 'cupy', None)
    monkeypatch.setitem(sys.modules, 'torch', None)

    spec = importlib.util.spec_from_file_location('ocr_pipeline_model', MODEL_PY)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['ocr_pipeline_model'] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules['ocr_pipeline_model']


class _Engine:
    """Mocked paddleocr_rec_trt: accepts widths in [48, 2048], errors otherwise."""

    MIN_WIDTH = 48
    MAX_WIDTH = 2048

    @classmethod
    def response_for(cls, shape: tuple[int, ...]):
        width = shape[-1]
        response = MagicMock()
        if cls.MIN_WIDTH <= width <= cls.MAX_WIDTH:
            response.has_error.return_value = False
            # [batch, T, num_classes] -- minimal decodable CTC output.
            response.array = np.zeros((1, 1, 4), dtype=np.float32)
            response.array[0, 0, 0] = 1.0  # argmax on the blank/first class
            return response
        response.has_error.return_value = True
        err = MagicMock()
        err.message.return_value = (
            f"request specifies invalid shape for input 'x' for "
            f'paddleocr_rec_trt_0_0. Error details: model expected the '
            f'shape of dimension 3 to be between {cls.MIN_WIDTH} and '
            f'{cls.MAX_WIDTH} but received {width}'
        )
        response.error.return_value = err
        return response


def _make_recognizer(model_module):
    """Build a TritonPythonModel instance without running initialize()."""
    obj: Any = model_module.TritonPythonModel.__new__(model_module.TritonPythonModel)
    obj.rec_height = model_module.REC_HEIGHT
    obj.ctc_decode = lambda _output: [('x', 0.9)]
    # _triton_to_numpy just needs to pass the mocked response array through.
    obj._triton_to_numpy = lambda tensor: tensor.array
    return obj


def test_narrow_crop_clamped_to_engine_min_width(model_module, monkeypatch):
    """RED (pre-fix MIN_WIDTH=8) would send width=8..47 to the engine and
    get a real BACKEND error every time. GREEN (MIN_WIDTH=48) clamps first.
    """
    _FakeInferenceRequest.response_factory = _Engine.response_for
    recognizer = _make_recognizer(model_module)

    # A very short/tall crop: h=48, w=4 -> aspect ratio pushes target_w
    # well below 48 before clamping.
    crop = np.zeros((48, 4, 3), dtype=np.uint8)

    texts, scores = recognizer._call_recognition([crop])

    assert _FakeInferenceRequest.last_shape is not None
    assert _FakeInferenceRequest.last_shape[-1] >= _Engine.MIN_WIDTH
    assert texts == ['x']
    assert scores == [0.9]


def test_genuine_backend_failure_is_not_silently_scored_zero(model_module):
    """Any *other* backend failure (post-fix, e.g. an engine reload glitch)
    must still surface as a value distinguishable from a legitimate
    low-confidence recognition -- never a bare ('', 0.0).
    """

    def _always_fails(_shape):
        response = MagicMock()
        response.has_error.return_value = True
        err = MagicMock()
        err.message.return_value = 'simulated backend outage'
        response.error.return_value = err
        return response

    _FakeInferenceRequest.response_factory = _always_fails
    recognizer = _make_recognizer(model_module)

    crop = np.zeros((48, 200, 3), dtype=np.uint8)
    texts, scores = recognizer._call_recognition([crop])

    assert texts == ['']
    assert scores == [-1.0]
    # 0.0 is a real, legitimate CTC confidence value -- must not collide.
    assert scores[0] != 0.0
