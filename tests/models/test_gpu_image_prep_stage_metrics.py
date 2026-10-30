"""Stage timing inside ``gpu_image_prep`` (``models/gpu_image_prep/1/stage_metrics.py``, SP0).

The model runs here against stand-ins for CuPy, nvJPEG, the CUDA kernels and Triton's Python
backend: the control flow, the metric counters and the response tensors are the real code.
"""

from __future__ import annotations

import importlib
import io
import json
import sys
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image


MODEL_DIR = Path(__file__).resolve().parents[2] / 'models' / 'gpu_image_prep'
FULL_STAGES = {
    'queue',
    'decode',
    'letterbox',
    'detect',
    'embed_frame',
    'embed_crops',
    'embed_pe',
    'crop_copy',
    'crop_jpeg',
    'blur',
}
W, H = 96, 64


class _Dev(np.ndarray):
    """A host array that answers ``.get()`` like a CuPy array."""

    def get(self) -> np.ndarray:
        return np.asarray(self)


def _dev(a: Any) -> _Dev:
    return np.ascontiguousarray(a).view(_Dev)


def _fake_cupy() -> types.ModuleType:
    cp = types.ModuleType('cupy')

    class _Ctx:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def __enter__(self) -> _Ctx:
            return self

        def __exit__(self, *_a: Any) -> None:
            return None

        def use(self) -> None:
            return None

        def synchronize(self) -> None:
            cp.syncs += 1

    cp.syncs = 0  # type: ignore[attr-defined]
    cp.uint8 = np.uint8  # type: ignore[attr-defined]
    cp.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        Device=_Ctx, Stream=_Ctx, get_current_stream=_Ctx
    )
    cp.get_default_memory_pool = lambda: types.SimpleNamespace(set_limit=lambda **_k: None)  # type: ignore[attr-defined]
    cp.full = lambda shape, v, dt: _dev(np.full(shape, v, dt))  # type: ignore[attr-defined]
    cp.empty = lambda shape, dt: _dev(np.zeros(shape, dt))  # type: ignore[attr-defined]
    cp.ascontiguousarray = _dev  # type: ignore[attr-defined]
    cp.asnumpy = np.asarray  # type: ignore[attr-defined]
    return cp


class _Tensor:
    def __init__(self, name: str, arr: np.ndarray) -> None:
        self.name, self.arr = name, arr

    def as_numpy(self) -> np.ndarray:
        return self.arr

    def is_cpu(self) -> bool:
        return True

    @staticmethod
    def from_dlpack(name: str, arr: np.ndarray) -> _Tensor:
        return _Tensor(name, np.asarray(arr))


class _Response:
    def __init__(self, output_tensors: list[_Tensor], error: Any = None) -> None:
        self.output_tensors, self.error_obj = output_tensors, error

    def has_error(self) -> bool:
        return self.error_obj is not None


class _BlsRequest:
    def __init__(self, model_name: str, inputs: list[_Tensor], requested_output_names: list[str]):
        self.model, self.inputs, self.outputs = model_name, inputs, requested_output_names

    def exec(self) -> _Response:
        n = self.inputs[0].arr.shape[0]
        if self.model == 'det':
            return _Response(
                [
                    _Tensor('num_dets', np.array([[2]], np.int32)),
                    _Tensor('det_boxes', np.array([[[100, 120, 300, 400], [320, 320, 600, 600]]])),
                    _Tensor('det_scores', np.array([[0.9, 0.8]], np.float32)),
                    _Tensor('det_classes', np.array([[1, 2]], np.int32)),
                ]
            )
        vec = np.tile(np.arange(8, dtype=np.float32), (n, 1)) + np.arange(n)[:, None]
        return _Response([_Tensor('image_embeddings', vec)])


def _fake_pb(counters: dict[tuple[str, str], list[float]]) -> types.ModuleType:
    pb = types.ModuleType('triton_python_backend_utils')

    class _Family:
        COUNTER = 'counter'

        def __init__(self, name: str, description: str, kind: str) -> None:
            self.name, self.description, self.kind = name, description, kind

        def Metric(self, labels: dict[str, str]) -> Any:  # noqa: N802  Triton's name
            series = counters.setdefault((self.name, labels['stage']), [])
            return types.SimpleNamespace(increment=series.append)

    pb.MetricFamily = _Family  # type: ignore[attr-defined]
    pb.Tensor = _Tensor  # type: ignore[attr-defined]
    pb.InferenceResponse = _Response  # type: ignore[attr-defined]
    pb.InferenceRequest = _BlsRequest  # type: ignore[attr-defined]
    pb.TritonError = lambda msg: msg  # type: ignore[attr-defined]
    pb.TritonModelException = RuntimeError  # type: ignore[attr-defined]
    pb.TRITONSERVER_RESPONSE_COMPLETE_FINAL = 1  # type: ignore[attr-defined]
    pb.get_input_tensor_by_name = lambda req, name: req.inputs.get(name)  # type: ignore[attr-defined]
    pb.get_output_tensor_by_name = lambda resp, name: next(  # type: ignore[attr-defined]
        t for t in resp.output_tensors if t.name == name
    )
    return pb


class _Lanes:
    def __init__(self, **_k: Any) -> None:
        pass

    def decode(self, data: bytes) -> tuple[_Dev, str]:
        rng = np.random.default_rng(len(data))
        return _dev(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)), 'fake'


class _Kernels:
    def resample_into(self, *_a: Any, **_k: Any) -> None:
        return None

    def lap_sums(self, _frame: Any, rects: list[tuple[int, int, int, int]]) -> np.ndarray:
        return np.arange(2 * len(rects), dtype=np.int64).reshape(-1, 2)


class _Request:
    def __init__(self, data: bytes, boxes: np.ndarray | None = None) -> None:
        self.inputs = {'jpeg': _Tensor('jpeg', np.frombuffer(data, np.uint8))}
        if boxes is not None:
            self.inputs['boxes'] = _Tensor('boxes', boxes)
        self.done = threading.Event()
        self.response: _Response | None = None

    def get_response_sender(self) -> _Request:
        return self

    def send(self, response: _Response, flags: int) -> None:
        self.response, self.flags = response, flags
        self.done.set()


@pytest.fixture
def counters() -> dict[tuple[str, str], list[float]]:
    return {}


@pytest.fixture
def make_model(
    monkeypatch: pytest.MonkeyPatch, counters: dict[tuple[str, str], list[float]]
) -> Any:
    cp = _fake_cupy()
    lanes_mod = types.ModuleType('nvjpeg_lanes')
    lanes_mod.NvJpegLanes = _Lanes  # type: ignore[attr-defined]
    kernels_mod = types.ModuleType('prep_kernels')
    kernels_mod.PrepKernels = _Kernels  # type: ignore[attr-defined]
    for name, mod in {
        'cupy': cp,
        'triton_python_backend_utils': _fake_pb(counters),
        'nvjpeg_lanes': lanes_mod,
        'prep_kernels': kernels_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.syspath_prepend(str(MODEL_DIR / '1'))
    for name in ('model', 'prep_geometry', 'stage_metrics'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    model_mod = importlib.import_module('model')
    created: list[Any] = []

    def build(**params: str) -> Any:
        cfg = {
            'parameters': {
                k: {'string_value': v}
                for k, v in {
                    'DETECTOR_MODEL': 'det',
                    'PE_MODEL': 'pe',
                    'WORKERS': '2',
                    **params,
                }.items()
            }
        }
        m = model_mod.TritonPythonModel()
        m.initialize({'model_config': json.dumps(cfg), 'model_instance_device_id': '0'})
        created.append(m)
        return m

    build.cp = cp  # type: ignore[attr-defined]
    build.module = model_mod  # type: ignore[attr-defined]
    yield build
    for m in created:
        m.workers.shutdown(wait=True)
        m.encode_pool.shutdown(wait=True)


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (W, H), (10, 20, 30)).save(buf, 'JPEG')
    return buf.getvalue()


def _run(model: Any, request: _Request) -> dict[str, np.ndarray]:
    model.execute([request])
    assert request.done.wait(10)
    assert request.response is not None
    assert not request.response.has_error()
    return {t.name: t.arr for t in request.response.output_tensors}


def _seen(counters: dict[tuple[str, str], list[float]]) -> dict[str, int]:
    return {
        s: len(v) for (fam, s), v in counters.items() if fam == 'op_gpu_prep_stage_observations'
    }


def _total(counters: dict[tuple[str, str], list[float]], stage: str) -> float:
    return sum(counters[('op_gpu_prep_stage_seconds', stage)])


def test_every_stage_is_observed_once_per_image(make_model: Any, counters: dict) -> None:
    model = make_model()
    _run(model, _Request(_jpeg()))
    _run(model, _Request(_jpeg()))

    assert _seen(counters) == dict.fromkeys(FULL_STAGES, 2)
    for stage in FULL_STAGES:
        assert _total(counters, stage) >= 0.0
    assert _total(counters, 'crop_jpeg') > 0.0  # two detections were encoded


def test_outputs_are_identical_with_metrics_and_with_a_synchronising_clock(
    make_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = _run(make_model(), _Request(_jpeg()))

    cp = make_model.cp
    synced = _run(make_model(STAGE_SYNC='1'), _Request(_jpeg()))
    assert cp.syncs > 0  # the attribution mode waited on the stream

    monkeypatch.setattr(make_model.module.StageMetrics, 'publish', lambda *_a: None)
    unmetered = _run(make_model(), _Request(_jpeg()))

    for other in (synced, unmetered):
        assert other.keys() == plain.keys()
        for name, arr in plain.items():
            assert np.array_equal(arr, other[name]), name


def test_no_sync_by_default(make_model: Any) -> None:
    _run(make_model(), _Request(_jpeg()))
    assert make_model.cp.syncs == 0


def test_stored_box_mode_observes_only_the_stages_it_runs(make_model: Any, counters: dict) -> None:
    boxes = np.array([[0.1, 0.1, 0.5, 0.6]], np.float64)
    out = _run(make_model(), _Request(_jpeg(), boxes))
    assert out['crop_embeddings'].shape[0] == 1
    assert {s for s, n in _seen(counters).items() if n} == {
        'queue',
        'decode',
        'embed_frame',
        'embed_crops',
        'embed_pe',
    }


def test_a_cpu_fallback_image_observes_queue_and_decode(make_model: Any, counters: dict) -> None:
    out = _run(make_model(), _Request(b'not a jpeg at all'))
    assert out['status'][0] == 'not_jpeg'
    assert {s for s, n in _seen(counters).items() if n} == {'queue', 'decode'}


def test_the_queue_stage_counts_the_wait_for_a_free_slot(make_model: Any, counters: dict) -> None:
    model = make_model(MAX_IN_FLIGHT='1')
    first, second = _Request(_jpeg()), _Request(_jpeg())
    model.execute([first, second])
    assert first.done.wait(10)
    assert second.done.wait(10)
    assert _seen(counters)['queue'] == 2


def test_an_unknown_stage_is_refused() -> None:
    sys.path.insert(0, str(MODEL_DIR / '1'))
    try:
        sm = importlib.import_module('stage_metrics')
        clock = sm.StageClock()
        with pytest.raises(ValueError, match='unknown gpu_image_prep stage'):
            clock.add('typo', 1.0)
    finally:
        sys.path.remove(str(MODEL_DIR / '1'))
        sys.modules.pop('stage_metrics', None)


def test_the_config_declares_the_attribution_switch_off() -> None:
    text = (MODEL_DIR / 'config.pbtxt').read_text()
    assert 'key: "STAGE_SYNC" value: { string_value: "0" }' in text
