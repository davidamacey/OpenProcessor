"""Stage timing inside ``gpu_image_prep`` (``models/gpu_image_prep/1/stage_metrics.py``, SP0).

The model runs here against stand-ins for CuPy, nvJPEG, the CUDA kernels and Triton's Python
backend: the control flow, the metric counters and the response tensors are the real code.
"""

from __future__ import annotations

import gc
import importlib
import io
import json
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image


MODEL_DIR = Path(__file__).resolve().parents[2] / 'models' / 'gpu_image_prep'
FULL_STAGES = {
    'queue',
    'budget',
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
    out_of_memory = type('OutOfMemoryError', (RuntimeError,), {})
    cp.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        Device=_Ctx,
        Stream=_Ctx,
        get_current_stream=_Ctx,
        memory=types.SimpleNamespace(OutOfMemoryError=out_of_memory),
        runtime=types.SimpleNamespace(
            memGetInfo=lambda: (3 << 30, 8 << 30),
            CUDARuntimeError=type('CUDARuntimeError', (RuntimeError,), {}),
        ),
    )
    cp.get_default_memory_pool = lambda: types.SimpleNamespace(set_limit=lambda **_k: None)  # type: ignore[attr-defined]
    cp.full = lambda shape, v, dt: _dev(np.full(shape, v, dt))  # type: ignore[attr-defined]
    cp.alloc_limit = None  # type: ignore[attr-defined]
    cp.allocated = []  # type: ignore[attr-defined]

    def _empty(shape: Any, dt: Any) -> _Dev:
        n = int(np.prod(shape)) * np.dtype(dt).itemsize
        if cp.alloc_limit is not None and n > cp.alloc_limit:
            raise out_of_memory(f'Out of memory allocating {n} bytes')
        cp.allocated.append(n)
        return _dev(np.zeros(shape, dt))

    cp.empty = _empty  # type: ignore[attr-defined]
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

    violations: list[str] = []
    overlaps: list[int] = []
    gate = {'busy': False}

    class _Family:
        COUNTER = 'counter'

        def __init__(self, name: str, description: str, kind: str) -> None:
            self.name, self.description, self.kind = name, description, kind
            self.state = {'deleted': False, 'live': 0}

        def __del__(self) -> None:
            # Triton deletes a family when its Python object is collected: its metrics then
            # vanish from /metrics and increment() raises, and a family that dies before its
            # metrics crashes the Python backend (both seen on Triton 2.70).
            self.state['deleted'] = True
            if self.state['live']:
                violations.append(self.name)

        def Metric(self, labels: dict[str, str]) -> Any:  # noqa: N802  Triton's name
            series = counters.setdefault((self.name, labels['stage']), [])
            return _Metric(self.state, series)

    class _Metric:
        def __init__(self, state: dict[str, Any], series: list[float]) -> None:
            self.state, self.series = state, series
            state['live'] += 1

        def __del__(self) -> None:
            self.state['live'] -= 1

        def increment(self, value: float) -> None:
            if self.state['deleted']:
                raise RuntimeError('the MetricFamily has been deleted')
            # Triton 2.70 aborts the server when two threads increment at once
            if gate['busy']:
                overlaps.append(1)
            gate['busy'] = True
            time.sleep(0.0005)
            self.series.append(value)
            gate['busy'] = False

    pb.MetricFamily = _Family  # type: ignore[attr-defined]
    pb.family_order_violations = violations  # type: ignore[attr-defined]
    pb.increment_overlaps = overlaps  # type: ignore[attr-defined]
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

    fail_with: Exception | None = None
    seen_streams: list[Any] = []

    def decode(self, data: bytes, alloc_stream: Any = None) -> tuple[_Dev, str]:
        _Lanes.seen_streams.append(alloc_stream)
        if _Lanes.fail_with is not None:
            raise _Lanes.fail_with
        rng = np.random.default_rng(len(data))
        return _dev(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)), 'fake'


class _Kernels:
    fail_on_call: tuple[int, Exception] | None = None
    calls = 0

    def resample_into(self, *_a: Any, **_k: Any) -> None:
        _Kernels.calls += 1
        if _Kernels.fail_on_call and _Kernels.calls == _Kernels.fail_on_call[0]:
            raise _Kernels.fail_on_call[1]

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
    _Lanes.fail_with = None
    _Lanes.seen_streams = []
    _Kernels.fail_on_call = None
    _Kernels.calls = 0
    for name in ('model', 'prep_geometry', 'stage_metrics', 'prep_budget'):
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
    model.metrics.flush()
    return {t.name: t.arr for t in request.response.output_tensors}


def _seen(counters: dict[tuple[str, str], list[float]]) -> dict[str, int]:
    return {
        s: int(sum(v))
        for (fam, s), v in counters.items()
        if fam == 'op_gpu_prep_stage_observations'
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


def test_only_one_thread_increments_at_a_time(
    make_model: Any, counters: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(make_model.module.StageMetrics, 'FLUSH_SECONDS', 0.001)
    model = make_model(WORKERS='8')
    requests = [_Request(_jpeg()) for _ in range(24)]
    model.execute(requests)
    assert all(r.done.wait(20) for r in requests)
    model.metrics.flush()

    assert sys.modules['triton_python_backend_utils'].increment_overlaps == []
    assert _seen(counters)['queue'] == 24
    assert _seen(counters)['detect'] == 24


def test_finalize_drops_the_metrics_before_their_families(make_model: Any) -> None:
    model = make_model()
    _run(model, _Request(_jpeg()))
    model.finalize()
    del model
    gc.collect()
    assert sys.modules['triton_python_backend_utils'].family_order_violations == []


def test_no_sync_by_default(make_model: Any) -> None:
    _run(make_model(), _Request(_jpeg()))
    assert make_model.cp.syncs == 0


def test_stored_box_mode_observes_only_the_stages_it_runs(make_model: Any, counters: dict) -> None:
    boxes = np.array([[0.1, 0.1, 0.5, 0.6]], np.float64)
    out = _run(make_model(), _Request(_jpeg(), boxes))
    assert out['crop_embeddings'].shape[0] == 1
    assert {s for s, n in _seen(counters).items() if n} == {
        'queue',
        'budget',
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
    model.metrics.flush()
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


# ------------------------------------------------------------------ device-memory budget


def _status(model: Any, request: _Request) -> str:
    model.execute([request])
    assert request.done.wait(10)
    assert request.response is not None
    assert not request.response.has_error(), 'a memory failure is a status, not an error'
    model.metrics.flush()
    tensors = {t.name: t.arr for t in request.response.output_tensors}
    return str(tensors['status'][0])


def _mem_error(make_model: Any) -> Exception:
    return make_model.cp.cuda.memory.OutOfMemoryError('Out of memory allocating 63 MB')


def test_a_failed_decode_allocation_answers_gpu_memory_not_decoder(make_model: Any) -> None:
    model = make_model()
    _Lanes.fail_with = _mem_error(make_model)
    assert _status(model, _Request(_jpeg())) == 'gpu_memory'
    assert model.budget.in_use == 0


def test_a_decode_failure_that_is_not_memory_stays_decoder(make_model: Any) -> None:
    model = make_model()
    _Lanes.fail_with = ValueError('bad stream')
    assert _status(model, _Request(_jpeg())) == 'decoder'
    assert model.budget.in_use == 0


@pytest.mark.parametrize('call', [1, 2, 3, 4])
def test_a_failed_allocation_at_any_kernel_stage_answers_gpu_memory(
    make_model: Any, call: int
) -> None:
    """Calls 1 to 4: the letterbox, the frame square and the two crop squares."""
    model = make_model()
    _Kernels.fail_on_call = (call, _mem_error(make_model))
    assert _status(model, _Request(_jpeg())) == 'gpu_memory'
    assert model.budget.in_use == 0
    # the next image is unaffected
    _Kernels.fail_on_call = None
    assert _status(model, _Request(_jpeg())) == 'ok'


def test_a_failed_allocation_in_stored_box_mode_answers_gpu_memory(make_model: Any) -> None:
    model = make_model()
    _Kernels.fail_on_call = (1, _mem_error(make_model))
    boxes = np.array([[0.1, 0.1, 0.5, 0.6]], np.float64)
    assert _status(model, _Request(_jpeg(), boxes)) == 'gpu_memory'
    assert model.budget.in_use == 0


def test_another_exception_is_still_an_error_response(make_model: Any) -> None:
    model = make_model()
    _Kernels.fail_on_call = (1, ValueError('kernel bug'))
    request = _Request(_jpeg())
    model.execute([request])
    assert request.done.wait(10)
    assert request.response is not None
    assert request.response.has_error()
    assert model.budget.in_use == 0


def test_a_photo_larger_than_the_whole_budget_answers_over_budget(make_model: Any) -> None:
    model = make_model(DEVICE_BUDGET_MIB='0.01')  # 10,485 bytes against a 96 x 64 x 3 frame
    assert _status(model, _Request(_jpeg())) == 'over_budget'
    assert _Lanes.seen_streams == []  # never decoded


def test_requests_wait_for_the_budget_and_never_exceed_it(make_model: Any) -> None:
    """Budget for one 96 x 64 frame at a time: four concurrent images queue, all succeed."""
    model = make_model(WORKERS='4', DEVICE_BUDGET_MIB=str(20000 / 2**20))
    peak = [0]
    real = _Lanes.decode

    def watching(self: Any, data: bytes, alloc_stream: Any = None) -> Any:
        peak[0] = max(peak[0], model.budget.in_use)
        time.sleep(0.02)
        return real(self, data, alloc_stream)

    _Lanes.decode = watching  # type: ignore[method-assign]
    try:
        requests = [_Request(_jpeg()) for _ in range(4)]
        model.execute(requests)
        assert all(r.done.wait(20) for r in requests)
    finally:
        _Lanes.decode = real  # type: ignore[method-assign]
    statuses = [
        str({t.name: t.arr for t in r.response.output_tensors}['status'][0])
        for r in requests
        if r.response
    ]
    assert statuses == ['ok'] * 4
    assert 0 < peak[0] <= model.budget.total
    assert model.budget.in_use == 0


def test_a_wait_that_outlasts_its_limit_answers_gpu_memory(make_model: Any) -> None:
    model = make_model(WORKERS='2', DEVICE_BUDGET_MIB=str(20000 / 2**20), BUDGET_WAIT_SECONDS='0.2')
    gate = threading.Event()
    real = _Lanes.decode

    def stuck(self: Any, data: bytes, alloc_stream: Any = None) -> Any:
        gate.wait(5)
        return real(self, data, alloc_stream)

    _Lanes.decode = stuck  # type: ignore[method-assign]
    try:
        first, second = _Request(_jpeg()), _Request(_jpeg())
        model.execute([first])
        time.sleep(0.05)
        model.execute([second])
        assert second.done.wait(5)
        gate.set()
        assert first.done.wait(5)
    finally:
        gate.set()
        _Lanes.decode = real  # type: ignore[method-assign]
    assert second.response is not None
    assert {t.name: t.arr for t in second.response.output_tensors}['status'][0] == 'gpu_memory'
    assert first.response is not None
    assert {t.name: t.arr for t in first.response.output_tensors}['status'][0] == 'ok'
    assert model.budget.in_use == 0


def test_the_budget_changes_no_output(make_model: Any) -> None:
    wide = make_model()
    tight = make_model(WORKERS='4', DEVICE_BUDGET_MIB=str(20000 / 2**20))
    for _ in range(3):
        a = _run(wide, _Request(_jpeg()))
        b = _run(tight, _Request(_jpeg()))
        assert a.keys() == b.keys()
        for name in a:
            assert np.array_equal(a[name], b[name]), name


def test_frames_are_allocated_on_the_shared_arena_stream(make_model: Any) -> None:
    model = make_model()
    _run(model, _Request(_jpeg()))
    assert _Lanes.seen_streams == [model.alloc_stream]


def test_the_budget_is_reserved_at_load(make_model: Any) -> None:
    make_model(DEVICE_BUDGET_MIB='2048')
    assert 2048 * 2**20 in make_model.cp.allocated


def test_an_instance_that_cannot_reserve_its_budget_fails_to_load(make_model: Any) -> None:
    make_model.cp.alloc_limit = 1 << 20
    with pytest.raises(make_model.module.pb_utils.TritonModelException) as err:
        make_model(DEVICE_BUDGET_MIB='2048')
    text = str(err.value)
    assert 'DEVICE_BUDGET_MIB' in text
    assert '2048' in text


def test_a_budget_above_the_pool_limit_fails_to_load(make_model: Any) -> None:
    with pytest.raises(make_model.module.pb_utils.TritonModelException, match='POOL_MIB'):
        make_model(DEVICE_BUDGET_MIB='5000', POOL_MIB='4096')


def test_the_config_declares_the_budget_parameters() -> None:
    text = (MODEL_DIR / 'config.pbtxt').read_text()
    assert 'key: "DEVICE_BUDGET_MIB"' in text
    assert 'key: "BUDGET_WAIT_SECONDS"' in text
