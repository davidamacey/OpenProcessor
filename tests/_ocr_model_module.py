"""The ocr_pipeline Triton model imported in isolation behind a fake ``pb_utils``.

Registered as a plugin in ``tests/conftest.py`` (pytest ``pytest_plugins``,
https://docs.pytest.org/en/stable/how-to/writing_plugins.html) so the OCR tests
share the ``model_module`` fixture without importing it from each other.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


MODEL_PY = Path(__file__).resolve().parents[1] / 'models' / 'ocr_pipeline' / '1' / 'model.py'


class FakeTensor:
    def __init__(self, name: str, array: np.ndarray) -> None:
        self.name = name
        self.array = array

    def as_numpy(self) -> np.ndarray:
        return self.array


class FakeInferenceRequest:
    """Records the shape it was asked to run and returns a scripted response."""

    last_shape: tuple[int, ...] | None = None
    response_factory: Callable[[tuple[int, ...]], object] | None = None

    def __init__(self, model_name, requested_output_names, inputs):
        self.model_name = model_name
        self.requested_output_names = requested_output_names
        self.inputs = inputs
        FakeInferenceRequest.last_shape = inputs[0].array.shape

    def exec(self):
        assert FakeInferenceRequest.response_factory is not None
        return FakeInferenceRequest.response_factory(self.inputs[0].array.shape)


def make_fake_pb_utils() -> types.ModuleType:
    mod = types.ModuleType('triton_python_backend_utils')
    mod.Tensor = FakeTensor  # type: ignore[attr-defined]
    mod.InferenceRequest = FakeInferenceRequest  # type: ignore[attr-defined]
    mod.get_output_tensor_by_name = lambda response, _name: response  # type: ignore[attr-defined]
    mod.get_input_tensor_by_name = lambda _request, _name: None  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def model_module(monkeypatch: pytest.MonkeyPatch):
    """Import ocr_pipeline's model.py in isolation with a fake pb_utils."""
    fake_pb_utils = make_fake_pb_utils()
    monkeypatch.setitem(sys.modules, 'triton_python_backend_utils', fake_pb_utils)
    # cupy/torch are optional (HAS_GPU=False path) -- force that path so
    # this test doesn't need a GPU.
    monkeypatch.setitem(sys.modules, 'cupy', None)
    monkeypatch.setitem(sys.modules, 'torch', None)
    # Triton loads the version directory flat, so model.py imports its sibling by bare name.
    monkeypatch.syspath_prepend(str(MODEL_PY.parent))

    spec = importlib.util.spec_from_file_location('ocr_pipeline_model', MODEL_PY)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['ocr_pipeline_model'] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules['ocr_pipeline_model']
    sys.modules.pop('ocr_postprocess', None)
