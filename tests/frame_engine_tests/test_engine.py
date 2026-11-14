"""The frame engine's per-photo pipeline (``frame_engine/engine.py``) driven by its plan.

The engine runs here against stand-ins for CuPy, nvJPEG, the CUDA kernels and the transport to
Triton: the control flow, the plan handling, the budget, the stage clock and the outputs are the
real code. Ported from the Triton prep model's tests; the skip-proof tests are new: a branch the
plan leaves out issues no model call and produces no output.
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
import time
import types
from typing import Any

import numpy as np
import pytest
from _image_helpers import make_jpeg
from prometheus_client import CollectorRegistry

from src.domain import frame_engine_wire as wire


W, H = 96, 64
MIB = 1024 * 1024
FULL_STAGES = {
    'queue', 'budget', 'decode', 'letterbox', 'detect', 'embed_frame', 'embed_crops',
    'embed_pe', 'crop_copy', 'crop_jpeg', 'blur',
}  # fmt: skip
DETECTOR = {'model': 'det', 'input_size': 640, 'letterbox_fill': 114}
EMBED = {'model': 'pe', 'size': 16, 'batch': 32}


class _Dev(np.ndarray):
    """A host array that answers ``.get()`` like a CuPy array."""

    def get(self) -> np.ndarray:
        return np.asarray(self)


def _dev(a: Any) -> _Dev:
    return np.ascontiguousarray(a).view(_Dev)


def _fake_cupy() -> types.ModuleType:
    cp: Any = types.ModuleType('cupy')

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

    cp.syncs = 0
    cp.uint8 = np.uint8
    out_of_memory = type('OutOfMemoryError', (RuntimeError,), {})
    cp.cuda = types.SimpleNamespace(
        Device=_Ctx,
        Stream=_Ctx,
        get_current_stream=_Ctx,
        memory=types.SimpleNamespace(OutOfMemoryError=out_of_memory),
        runtime=types.SimpleNamespace(
            CUDARuntimeError=type('CUDARuntimeError', (RuntimeError,), {}),
        ),
    )
    cp.get_default_memory_pool = lambda: types.SimpleNamespace(set_limit=lambda **_k: None)
    cp.alloc_limit = None
    cp.allocated = []

    def _empty(shape: Any, dt: Any) -> _Dev:
        n = int(np.prod(shape)) * np.dtype(dt).itemsize
        if cp.alloc_limit is not None and n > cp.alloc_limit:
            raise out_of_memory(f'Out of memory allocating {n} bytes')
        cp.allocated.append(n)
        return _dev(np.zeros(shape, dt))

    cp.empty = _empty
    cp.ascontiguousarray = _dev
    cp.asnumpy = np.asarray
    return cp


class _Lanes:
    fail_with: Exception | None = None
    seen_streams: list[Any] = []

    def __init__(self, **_k: Any) -> None:
        pass

    def decode(self, data: bytes, alloc_stream: Any = None) -> tuple[_Dev, str]:
        _Lanes.seen_streams.append(alloc_stream)
        if _Lanes.fail_with is not None:
            raise _Lanes.fail_with
        rng = np.random.default_rng(len(data))
        return _dev(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)), 'fake'


class _Kernels:
    fail_on_call: tuple[int, Exception] | None = None
    calls = 0

    def resample_into(self, frame: Any, roi: Any, size: Any, dst: Any, *_a: Any, **_k: Any) -> None:
        _Kernels.calls += 1
        if _Kernels.fail_on_call and _Kernels.calls == _Kernels.fail_on_call[0]:
            raise _Kernels.fail_on_call[1]
        dst[...] = (roi[0] + roi[2]) % 251  # what the square holds depends on its rectangle

    def lap_sums(self, _frame: Any, rects: list[tuple[int, int, int, int]]) -> np.ndarray:
        return np.arange(2 * len(rects), dtype=np.int64).reshape(-1, 2)


class _Transport:
    """Records every model call and the order of commits, sends, collections and cancels; the
    detector finds two objects of classes 1 and 2. Every call is answered when it is sent."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[int]]] = []
        self.events: list[tuple[str, Any]] = []
        self.deadlines: list[float] = []
        self.staged = 0
        self.fail: dict[str, Exception] = {}
        self.fail_call: dict[int, Exception] = {}  # by the call's place in send order
        self.stage_fails: Exception | None = None

    def stage(self, nbytes: int) -> Any:
        from frame_engine.transport import Staged

        if self.stage_fails is not None:
            raise self.stage_fails
        self.staged += 1
        return Staged(device=_dev(np.zeros(nbytes, np.uint8)), nbytes=nbytes)

    def commit(self, staged: list[Any], sync: Any) -> None:
        sync()
        self.events.append(('commit', len(staged)))
        for s in staged:
            s.host = np.asarray(s.device).copy()
            s.committed = True

    def submit(self, model: str, staged: Any, tensor: Any, outputs: Any, *,
               deadline: float) -> Any:  # fmt: skip
        assert staged.host is not None, 'submit before commit'
        shape = list(tensor.shape)
        self.calls.append((model, shape))
        self.events.append(('submit', model))
        self.deadlines.append(deadline)
        return types.SimpleNamespace(model=model, host=staged.host, shape=shape, outputs=outputs,
                                     index=len(self.calls) - 1)  # fmt: skip

    def collect(self, p: Any) -> dict[str, Any]:
        self.events.append(('collect', p.model))
        if p.model in self.fail:
            raise self.fail[p.model]
        if p.index in self.fail_call:
            raise self.fail_call[p.index]
        if p.model == 'det':
            fused = np.zeros((1, 300, 6), np.float32)
            fused[0, :2] = [[100, 120, 300, 400, 0.9, 1], [320, 320, 600, 600, 0.8, 2]]
            return {p.outputs[0]: fused}
        n = p.shape[0]
        flat = p.host[: int(np.prod(p.shape))].reshape(n, -1)
        return {p.outputs[0]: flat[:, :8].astype(np.float32) + np.arange(8, dtype=np.float32)}

    def cancel(self, p: Any) -> None:
        self.events.append(('cancel', p.model))

    def release(self, staged: Any, sync: Any) -> None:
        if not staged.committed:
            sync()
        self.staged -= 1

    def models(self) -> list[str]:
        return [m for m, _ in self.calls]


def m_needs_cupy(name: str) -> bool:
    """The engine modules that import CuPy at load (re-imported against the stand-in)."""
    return name in ('frame_engine.engine', 'frame_engine.nvjpeg_lanes',
                    'frame_engine.prep_kernels') or name.startswith('frame_engine.branches')  # fmt: skip


@pytest.fixture
def eng(monkeypatch: pytest.MonkeyPatch) -> Any:
    cp = _fake_cupy()
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    for name in [m for m in sys.modules if m.startswith('frame_engine.')]:
        if m_needs_cupy(name):
            monkeypatch.delitem(sys.modules, name, raising=False)
    mod = importlib.import_module('frame_engine.engine')
    monkeypatch.setattr(mod, 'NvJpegLanes', _Lanes)
    monkeypatch.setattr(mod, 'PrepKernels', _Kernels)
    _Lanes.fail_with = None
    _Lanes.seen_streams = []
    _Kernels.fail_on_call = None
    _Kernels.calls = 0
    from frame_engine.settings import EngineSettings
    from frame_engine.stage_metrics import EngineMetrics

    created: list[Any] = []

    def build(transport: Any = None, **over: Any) -> Any:
        settings = EngineSettings(**{'workers': 2, **over})
        transport = transport or _Transport()
        engine = mod.FrameEngine(settings, transport, EngineMetrics(CollectorRegistry()))
        created.append(engine)
        return engine

    build.cp = cp  # type: ignore[attr-defined]
    build.module = mod  # type: ignore[attr-defined]
    yield build
    for e in created:
        e.close()


def _plan(branches: list[str] | None = None, *, boxes: bool = False, **over: Any) -> Any:
    body: dict[str, Any] = {
        'version': 1,
        'branches': branches or ['detect', 'embed_frame', 'embed_crops'],
        'detector': DETECTOR,
        'embed': EMBED,
        'crops': 'inline',
    }
    body.update(over)
    return wire.parse_plan(json.dumps({k: v for k, v in body.items() if v is not None}),
                           has_boxes=boxes)  # fmt: skip


def _jpeg() -> bytes:
    return make_jpeg(W, H, (10, 20, 30))


def _run(engine: Any, plan: Any = None, *, data: bytes | None = None, boxes: Any = None) -> Any:
    from frame_engine.messages import EngineRequest

    req = EngineRequest(data if data is not None else _jpeg(), plan or _plan(), boxes)
    return engine.submit(req).result(timeout=20)


def _seen(engine: Any) -> dict[str, int]:
    out = {}
    for metric in engine.metrics.stage_observations.collect():
        for s in metric.samples:
            if s.name.endswith('_total'):
                out[s.labels['stage']] = int(s.value)
    return out


def _mem_error(eng: Any) -> Exception:
    return eng.cp.cuda.memory.OutOfMemoryError('Out of memory allocating 63 MB')


# ------------------------------------------------------------------ the full plan


def test_the_full_plan_answers_every_output_and_observes_every_stage(eng: Any) -> None:
    engine = eng()
    r1 = _run(engine)
    # round 1: the detector and the frame square; round 2: one encoder call with both crops
    assert engine.transport.calls == [
        ('det', [1, 3, 640, 640]), ('pe', [1, 3, 16, 16]), ('pe', [2, 3, 16, 16]),
    ]  # fmt: skip
    _run(engine)
    assert r1.status == 'ok'
    assert r1.decode_route == 'gpu'
    assert r1.branches == dict.fromkeys(('detect', 'embed_frame', 'embed_crops'), ('ok', ''))
    assert set(r1.outputs) == {'size', *wire.expected_outputs(_plan())}
    assert r1.outputs['size'].tolist() == [W, H]
    assert r1.outputs['crop_embeddings'].shape == (2, 8)
    assert r1.outputs['crop_embedded'].tolist() == [True, True]
    assert r1.outputs['blur_sums'].shape == (3, 3)
    assert len(r1.outputs['crop_jpegs']) == 2
    assert all(j[:2] == b'\xff\xd8' for j in r1.outputs['crop_jpegs'])
    assert _seen(engine) == dict.fromkeys(FULL_STAGES, 2)
    assert engine.transport.staged == 0


def test_a_photo_without_detections_still_answers_every_output_of_its_plan(eng: Any) -> None:
    """``crop_jpegs`` is an output of ``crops: inline`` whatever the detector found, as the Triton
    prep model answered it (an empty list): the API reads it for every photo."""
    engine = eng()
    real = engine.transport.collect

    def nothing_found(p: Any) -> Any:
        out = real(p)
        return {k: np.zeros_like(v) for k, v in out.items()} if p.model == 'det' else out

    engine.transport.collect = nothing_found
    r = _run(engine)
    assert r.status == 'ok'
    assert set(r.outputs) == {'size', *wire.expected_outputs(_plan())}
    assert r.outputs['crop_jpegs'] == []
    assert r.outputs['det_boxes'].shape == (0, 4)
    assert r.outputs['crop_embeddings'].shape == (0, 8)


def test_encoder_chunks_follow_the_plan_batch(eng: Any) -> None:
    engine = eng()
    _run(engine, _plan(embed={**EMBED, 'batch': 1}))
    assert engine.transport.calls[1:] == [('pe', [1, 3, 16, 16])] * 3


# ------------------------------------------------------------------ skip-proof, per branch


def test_a_detect_only_plan_issues_no_encoder_call(eng: Any) -> None:
    engine = eng()
    r = _run(engine, _plan(['detect'], embed=None, crops='none'))
    assert engine.transport.models() == ['det']
    assert set(r.outputs) == {
        'size',
        'det_boxes',
        'det_scores',
        'det_classes',
        'letterbox',
        'blur_sums',
    }
    assert r.branches == {'detect': ('ok', '')}
    assert 'embed_pe' not in _seen(engine)
    assert 'crop_copy' not in _seen(engine)


def test_an_embed_frame_only_plan_issues_no_detector_call(eng: Any) -> None:
    engine = eng()
    r = _run(engine, _plan(['embed_frame'], detector=None, crops='none'))
    assert engine.transport.calls == [('pe', [1, 3, 16, 16])]
    assert set(r.outputs) == {'size', 'whole_embedding'}
    assert {s for s, n in _seen(engine).items() if n} == {
        'queue', 'budget', 'decode', 'embed_frame', 'embed_pe',
    }  # fmt: skip


def test_embed_crops_without_embed_frame_sends_no_frame_square(eng: Any) -> None:
    engine = eng()
    r = _run(engine, _plan(['detect', 'embed_crops'], crops='none'))
    assert engine.transport.calls == [('det', [1, 3, 640, 640]), ('pe', [2, 3, 16, 16])]
    assert 'whole_embedding' not in r.outputs
    assert r.outputs['crop_embeddings'].shape == (2, 8)


def test_embed_classes_embeds_only_the_wanted_detections(eng: Any) -> None:
    engine = eng()
    full = _run(engine, _plan())
    r = _run(engine, _plan(embed_classes=[2]))
    assert engine.transport.calls[-1] == ('pe', [1, 3, 16, 16])  # the one wanted crop
    assert r.outputs['crop_embedded'].tolist() == [False, True]
    assert r.outputs['crop_embeddings'].shape == (1, 8)
    np.testing.assert_array_equal(
        r.outputs['crop_embeddings'][0], full.outputs['crop_embeddings'][1]
    )
    assert len(r.outputs['crop_jpegs']) == 2  # crops are cut for every detection


def test_crops_none_encodes_no_jpeg(eng: Any) -> None:
    engine = eng()
    r = _run(engine, _plan(crops='none'))
    assert 'crop_jpegs' not in r.outputs
    assert 'crop_copy' not in _seen(engine)
    assert 'crop_jpeg' not in _seen(engine)


def test_stored_boxes_replace_the_detector(eng: Any) -> None:
    engine = eng()
    boxes = np.array([[0.1, 0.1, 0.5, 0.6], [0.0, 0.0, 1.0, 1.0]], np.float64)
    r = _run(engine, _plan(['embed_frame', 'embed_crops'], boxes=True, detector=None, crops='none'),
             boxes=boxes)  # fmt: skip
    assert engine.transport.calls == [('pe', [1, 3, 16, 16]), ('pe', [2, 3, 16, 16])]
    assert r.outputs['crop_embeddings'].shape == (2, 8)
    assert r.outputs['crop_embedded'].tolist() == [True, True]
    assert set(r.outputs) == {'size', 'whole_embedding', 'crop_embeddings', 'crop_embedded'}
    assert {s for s, n in _seen(engine).items() if n} == {
        'queue', 'budget', 'decode', 'embed_frame', 'embed_crops', 'embed_pe',
    }  # fmt: skip


def test_unbuilt_branches_answer_unsupported_and_the_rest_runs(eng: Any) -> None:
    engine = eng()
    plan = _plan(['detect', 'faces', 'ocr'], embed=None, crops='none',
                 derivatives=[{'name': 'thumb', 'side': 256}])  # fmt: skip
    r = _run(engine, plan)
    assert r.status == 'ok'
    assert r.branches == {
        'detect': ('ok', ''),
        'faces': ('error', 'unsupported_branch'),
        'ocr': ('error', 'unsupported_branch'),
        'derivatives': ('error', 'unsupported_branch'),
    }
    assert engine.transport.models() == ['det']
    assert not {'face_boxes', 'ocr_texts', 'derivative_jpegs'} & set(r.outputs)


def test_every_reported_name_has_exactly_one_branch_status_entry(eng: Any) -> None:
    engine = eng()
    for plan in (
        _plan(),
        _plan(['embed_frame'], detector=None, crops='none'),
        _plan(['detect', 'faces', 'face_embed'], embed=None, crops='none'),
    ):
        r = _run(engine, plan)
        assert list(r.branches) == sorted(r.branches, key=list(plan.reported()).index)
        assert set(r.branches) == set(plan.reported())
    failed = _run(engine, _plan(), data=b'not a jpeg')
    assert set(failed.branches) == set(_plan().reported())


# ------------------------------------------------------------------ model failures


def test_a_failed_detector_call_fails_detect_and_what_needs_it(eng: Any) -> None:
    from frame_engine.transport import ModelCallError

    engine = eng()
    engine.transport.fail['det'] = ModelCallError('det', 'StatusCode.INTERNAL', 'boom')
    r = _run(engine)
    assert r.status == 'ok'
    assert r.branches == {
        'detect': ('error', 'model_error'),
        'embed_frame': ('ok', ''),
        'embed_crops': ('error', 'dependency_failed'),
    }
    assert set(r.outputs) == {'size', 'whole_embedding'}
    assert engine.transport.staged == 0


def test_a_failed_encoder_call_fails_both_embed_branches(eng: Any) -> None:
    from frame_engine.transport import ModelCallError

    engine = eng()
    engine.transport.fail['pe'] = ModelCallError('pe', 'StatusCode.INTERNAL', 'boom')
    r = _run(engine)
    assert r.branches['detect'] == ('ok', '')
    assert r.branches['embed_frame'] == ('error', 'model_error')
    assert r.branches['embed_crops'] == ('error', 'model_error')
    assert 'whole_embedding' not in r.outputs
    assert len(r.outputs['crop_jpegs']) == 2


def test_an_unreachable_model_server_is_raised_not_answered(eng: Any) -> None:
    from frame_engine.transport import TritonUnavailableError

    engine = eng()
    engine.transport.fail['det'] = TritonUnavailableError('det: StatusCode.UNAVAILABLE')
    with pytest.raises(TritonUnavailableError):
        _run(engine)
    assert engine.budget.in_use == 0
    assert engine.transport.staged == 0


class _Region:
    raw_handle = b'aGFuZGxl'

    def __init__(self, nbytes: int) -> None:
        self.buf = np.zeros(nbytes, np.uint8)

    def view(self) -> Any:
        return self.buf

    def free(self) -> None:
        return None


class _Server:
    """A model server that takes every call and answers none until ``answering`` is set, or only
    the models in ``answers`` (the detector finds two objects, the encoder answers 8-wide
    vectors)."""

    def __init__(self) -> None:
        self.answering = False
        self.answers: set[str] = set()
        self.cancelled = 0
        self.regions: dict[str, int] = {}

    def is_server_ready(self, client_timeout: float | None = None) -> bool:
        return True

    def get_cuda_shared_memory_status(self, as_json: bool = False,
                                      client_timeout: float | None = None) -> Any:  # fmt: skip
        return {'regions': {name: {} for name in self.regions}}

    def register_cuda_shared_memory(self, name: str, *_a: Any, client_timeout: Any = None) -> None:
        self.regions[name] = 1

    def unregister_cuda_shared_memory(self, name: str = '', client_timeout: Any = None) -> None:
        self.regions.pop(name, None)

    @staticmethod
    def _result(model: str, inputs: Any) -> Any:
        n = int(inputs[0].shape()[0])
        if model == 'det':
            out = np.zeros((1, 300, 6), np.float32)
            out[0, :2] = [[100, 120, 300, 400, 0.9, 1], [320, 320, 600, 600, 0.8, 2]]
        else:
            out = np.ones((n, 8), np.float32)
        return types.SimpleNamespace(as_numpy=lambda _name: out)

    def async_infer(self, model_name: str, inputs: Any, callback: Any, *, outputs: Any = None,
                    client_timeout: Any = None) -> Any:  # fmt: skip
        if self.answering or model_name in self.answers:
            callback(result=self._result(model_name, inputs), error=None)
        return types.SimpleNamespace(cancel=self._cancel)

    def _cancel(self) -> None:
        self.cancelled += 1


def test_a_model_server_that_never_answers_fails_the_photo_and_frees_everything(
    eng: Any,
) -> None:
    """The model server restarted with calls in flight: each call ends at its deadline, the photo
    is answered (unavailable, retryable), its slice and its budget come back, and the next photo
    is served once the server answers again, without restarting the engine."""
    import tritonclient.grpc as grpcclient

    from frame_engine.stage_metrics import EngineMetrics
    from frame_engine.transport import CudaShmTransport, TritonUnavailableError

    server = _Server()
    transport = CudaShmTransport(
        [server],
        EngineMetrics(CollectorRegistry()),
        timeout=0.3,
        input_factory=grpcclient.InferInput,
        output_factory=grpcclient.InferRequestedOutput,
        region_factory=lambda _name, n: _Region(n),
        nbytes=8 << 20,
        card='card',
        instance='fe0',
        triton_device_id=0,
        slab_wait=0.05,
    )
    transport.start()
    engine = eng(transport=transport)
    start = time.monotonic()
    with pytest.raises(TritonUnavailableError, match='no answer'):
        _run(engine)
    assert time.monotonic() - start < 2
    assert server.cancelled == 2  # the detector's call at its deadline, the frame square's with it
    assert (engine.budget.in_use, transport.slab.in_use) == (0, 0)
    server.answering = True
    assert transport.check()
    r = _run(engine)
    assert r.status == 'ok'
    assert (engine.budget.in_use, transport.slab.in_use) == (0, 0)


def _shm(server: Any, nbytes: int) -> Any:
    """The real shared-memory transport over a fake server and a host buffer as the region."""
    import tritonclient.grpc as grpcclient

    from frame_engine.stage_metrics import EngineMetrics
    from frame_engine.transport import CudaShmTransport

    transport = CudaShmTransport(
        [server],
        EngineMetrics(CollectorRegistry()),
        timeout=5.0,
        input_factory=grpcclient.InferInput,
        output_factory=grpcclient.InferRequestedOutput,
        region_factory=lambda _name, n: _Region(n),
        nbytes=nbytes,
        card='card',
        instance='fe0',
        triton_device_id=0,
        slab_wait=0.05,
    )
    transport.start()
    return transport


STORED = np.array([[0.1, 0.1, 0.5, 0.6], [0.0, 0.0, 1.0, 1.0], [0.2, 0.3, 0.9, 0.8]], np.float64)


def _stored_plan(size: int, batch: int) -> Any:
    return _plan(['embed_frame', 'embed_crops'], boxes=True, detector=None, crops='none',
                 embed={**EMBED, 'size': size, 'batch': batch})  # fmt: skip


def test_the_squares_are_staged_one_encoder_call_at_a_time(eng: Any) -> None:
    """A region that holds one call's squares but not the photo's four serves the photo: the
    slab need is the plan's batch, not the number of boxes."""
    server = _Server()
    server.answering = True
    square = 3 * 64 * 64
    transport = _shm(server, 2 * square)  # two squares; the photo has four (frame + 3 boxes)
    engine = eng(transport=transport, region_mib=2 * square / MIB)
    r = _run(engine, _stored_plan(64, 2), boxes=STORED)
    assert r.status == 'ok'
    assert r.outputs['crop_embeddings'].shape == (3, 8)
    assert transport.slab.peak <= 2 * square
    assert (engine.budget.in_use, transport.slab.in_use) == (0, 0)


def test_staging_per_call_changes_no_output(eng: Any) -> None:
    whole = _run(eng(), _stored_plan(16, 32), boxes=STORED)
    chunked = _run(eng(), _stored_plan(16, 1), boxes=STORED)
    for name in ('whole_embedding', 'crop_embeddings', 'crop_embedded'):
        assert np.array_equal(whole.outputs[name], chunked.outputs[name]), name


def test_a_call_larger_than_the_whole_region_answers_over_budget_with_one_warning(
    eng: Any, caplog: pytest.LogCaptureFixture
) -> None:
    server = _Server()
    server.answering = True
    square = 3 * 64 * 64
    transport = _shm(server, 2 * square)
    engine = eng(transport=transport, region_mib=2 * square / MIB)
    for _ in range(2):
        r = _run(engine, _stored_plan(64, 4), boxes=STORED)  # three crop squares in one call
        assert (r.status, r.decode_route) == ('over_budget', 'none')
        assert (engine.budget.in_use, transport.slab.in_use) == (0, 0)
    warnings = [rec for rec in caplog.records if 'OP_FRAME_ENGINE_REGION_MIB' in rec.getMessage()]
    assert len(warnings) == 1  # the first, then every hundredth
    assert warnings[0].exc_info is None
    assert _run(engine, _stored_plan(64, 2), boxes=STORED).status == 'ok'


def _boxes(n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 0.6, (n, 2))
    return np.concatenate([xy, xy + rng.uniform(0.1, 0.4, (n, 2))], axis=1).clip(0.0, 1.0)


def _count_in_flight(transport: Any) -> list[int]:
    """Wraps the transport: [calls in flight now, most at once]."""
    seen = [0, 0]
    submit, collect = transport.submit, transport.collect

    def counted_submit(*a: Any, **k: Any) -> Any:
        p = submit(*a, **k)
        seen[0] += 1
        seen[1] = max(seen)
        return p

    def counted_collect(p: Any) -> Any:
        seen[0] -= 1
        return collect(p)

    transport.submit, transport.collect = counted_submit, counted_collect
    return seen


def test_round_one_sends_the_detector_and_the_frame_square_after_one_commit(eng: Any) -> None:
    """Both calls that need only the frame are in flight together; the crop squares wait for
    the detector's boxes, then go out before any encoder answer is collected."""
    engine = eng()
    _run(engine)
    assert engine.transport.events == [
        ('commit', 2), ('submit', 'det'), ('submit', 'pe'),  # round 1: one wait, two calls
        ('collect', 'det'),  # settle: the boxes
        ('commit', 1), ('submit', 'pe'),  # round 2: the crop squares
        ('collect', 'pe'), ('collect', 'pe'),  # finish
    ]  # fmt: skip


def test_round_two_submits_every_chunk_before_it_collects_any(eng: Any) -> None:
    engine = eng()
    _run(engine, _plan(embed={**EMBED, 'batch': 1}))
    events = engine.transport.events
    last_submit = max(i for i, e in enumerate(events) if e == ('submit', 'pe'))
    first_collect = min(i for i, e in enumerate(events) if e == ('collect', 'pe'))
    assert last_submit < first_collect
    assert events.count(('submit', 'pe')) == 3


def test_the_window_bounds_what_one_photo_holds_with_calls_in_flight(eng: Any) -> None:
    """300 rectangles in calls of 4 squares: many calls in flight at once, never more region
    than the photo's fair share (region over workers) plus one call."""
    server = _Server()
    server.answering = True
    square = 3 * 64 * 64
    region = 1 << 20
    transport = _shm(server, region)
    seen = _count_in_flight(transport)
    engine = eng(transport=transport, region_mib=region / MIB, workers=2)
    r = _run(engine, _stored_plan(64, 4), boxes=_boxes(300))
    assert r.status == 'ok'
    assert r.outputs['crop_embeddings'].shape == (300, 8)
    assert seen[1] >= 8  # the window holds ten calls of four squares
    assert transport.slab.peak <= engine.window + 4 * square
    assert (engine.budget.in_use, transport.slab.in_use) == (0, 0)


def test_a_photo_with_300_detections_is_served_by_a_16_mib_region(eng: Any) -> None:
    """The smallest region the settings allow, the encoder's real square (336) and batch (32),
    sixteen workers: the photo's 300 crop squares (97 MiB) go through it one window at a time."""
    server = _Server()
    server.answering = True
    region = 16 * MIB
    transport = _shm(server, region)
    engine = eng(transport=transport, region_mib=16, workers=16)
    r = _run(engine, _stored_plan(336, 32), boxes=_boxes(300))
    assert r.status == 'ok'
    assert r.outputs['crop_embeddings'].shape == (300, 8)
    assert transport.slab.peak <= region
    assert (engine.budget.in_use, transport.slab.in_use) == (0, 0)


def test_a_failed_crop_call_fails_embed_crops_only_and_cancels_its_other_calls(eng: Any) -> None:
    from frame_engine.transport import ModelCallError

    engine = eng()
    # sent in order: detector, frame square, crop square 1, crop square 2
    engine.transport.fail_call[2] = ModelCallError('pe', 'StatusCode.INTERNAL', 'boom')
    r = _run(engine, _plan(embed={**EMBED, 'batch': 1}))
    assert r.branches == {
        'detect': ('ok', ''),
        'embed_frame': ('ok', ''),
        'embed_crops': ('error', 'model_error'),
    }
    assert ('cancel', 'pe') in engine.transport.events  # crop square 2, still in flight
    assert set(r.outputs) == {'size', *wire.expected_outputs(_plan())} - {
        'crop_embeddings', 'crop_embedded',
    }  # fmt: skip
    assert engine.transport.staged == 0


def test_a_failed_frame_call_fails_embed_frame_only(eng: Any) -> None:
    from frame_engine.transport import ModelCallError

    engine = eng()
    engine.transport.fail_call[1] = ModelCallError('pe', 'StatusCode.INTERNAL', 'boom')
    r = _run(engine)
    assert r.branches['embed_frame'] == ('error', 'model_error')
    assert r.branches['embed_crops'] == ('ok', '')
    assert r.outputs['crop_embeddings'].shape == (2, 8)
    assert 'whole_embedding' not in r.outputs
    assert engine.transport.staged == 0


def test_an_encoder_that_never_answers_ends_the_photo_by_one_deadline(eng: Any) -> None:
    """The detector answers; the frame square and two crop calls are in flight and never
    answered: the photo is answered once, at about one call deadline, and every call is
    cancelled and every slice back."""
    from frame_engine.transport import TritonUnavailableError

    server = _Server()
    server.answers = {'det'}
    transport = _shm(server, 8 << 20)
    transport.timeout = 0.3
    engine = eng(transport=transport)
    start = time.monotonic()
    with pytest.raises(TritonUnavailableError, match='no answer'):
        _run(engine, _plan(embed={**EMBED, 'batch': 1}))
    assert time.monotonic() - start < 0.9
    assert server.cancelled == 3
    assert (engine.budget.in_use, transport.slab.in_use) == (0, 0)


def test_outputs_come_in_the_wire_order(eng: Any) -> None:
    r = _run(eng())
    assert list(r.outputs) == ['size', *wire.expected_outputs(_plan())]


def test_every_model_call_carries_the_request_deadline(eng: Any) -> None:
    from frame_engine.messages import EngineRequest

    engine = eng()
    request = EngineRequest(_jpeg(), _plan(), deadline=time.perf_counter() + 42)
    assert engine.submit(request).result(timeout=20).status == 'ok'
    assert engine.transport.deadlines == [request.deadline] * 3


def test_the_request_shows_the_stage_it_is_in(eng: Any) -> None:
    from frame_engine.messages import EngineRequest

    engine = eng()
    gate = threading.Event()
    real = engine.transport.collect

    def slow_detector(p: Any) -> Any:
        if p.model == 'det':
            gate.wait(5)
        return real(p)

    engine.transport.collect = slow_detector
    request = EngineRequest(_jpeg(), _plan())
    assert request.stage == 'queue'
    future = engine.submit(request)
    for _ in range(500):
        if request.stage == 'detect':
            break
        time.sleep(0.01)
    seen = request.stage
    gate.set()
    assert future.result(timeout=5).status == 'ok'
    assert seen == 'detect'


# ------------------------------------------------------------------ statuses and memory


def test_an_image_the_gpu_route_does_not_take_answers_its_reason(eng: Any) -> None:
    engine = eng()
    r = _run(engine, data=b'not a jpeg at all')
    assert (r.status, r.decode_route) == ('not_jpeg', 'none')
    assert r.outputs == {}
    assert set(r.branches.values()) == {('error', 'image_failed')}
    assert {s for s, n in _seen(engine).items() if n} == {'queue', 'decode'}
    assert engine.transport.calls == []


def test_a_failed_decode_allocation_answers_gpu_memory_not_decoder(eng: Any) -> None:
    engine = eng()
    _Lanes.fail_with = _mem_error(eng)
    assert _run(engine).status == 'gpu_memory'
    assert engine.budget.in_use == 0


def test_a_decode_failure_that_is_not_memory_stays_decoder(eng: Any) -> None:
    engine = eng()
    _Lanes.fail_with = eng.module.NvJpegError('DecodeJpegHost', 6)
    assert _run(engine).status == 'decoder'
    assert engine.budget.in_use == 0


@pytest.mark.parametrize('call', [1, 2, 3, 4])
def test_a_failed_allocation_at_any_kernel_stage_answers_gpu_memory(eng: Any, call: int) -> None:
    """Calls 1 to 4: the letterbox, the frame square and the two crop squares."""
    engine = eng()
    _Kernels.fail_on_call = (call, _mem_error(eng))
    assert _run(engine).status == 'gpu_memory'
    assert engine.budget.in_use == 0
    assert engine.transport.staged == 0
    _Kernels.fail_on_call = None
    assert _run(engine).status == 'ok'


def test_another_decode_exception_is_raised_not_called_a_decoder_failure(eng: Any) -> None:
    engine = eng()
    _Lanes.fail_with = ValueError('a bug in the lane code')
    with pytest.raises(ValueError, match='lane code'):
        _run(engine)
    assert engine.budget.in_use == 0


def test_another_exception_is_raised_and_releases_everything(eng: Any) -> None:
    engine = eng()
    _Kernels.fail_on_call = (1, ValueError('kernel bug'))
    with pytest.raises(ValueError, match='kernel bug'):
        _run(engine)
    assert engine.budget.in_use == 0
    assert engine.transport.staged == 0


def test_a_photo_larger_than_the_whole_budget_answers_over_budget(eng: Any) -> None:
    engine = eng(budget_mib=0.01)  # 10,485 bytes against a 96 x 64 x 3 frame
    assert _run(engine).status == 'over_budget'
    assert _Lanes.seen_streams == []  # never decoded


def test_requests_wait_for_the_budget_and_never_exceed_it(eng: Any) -> None:
    """Budget for one 96 x 64 frame at a time: four concurrent images queue, all succeed."""
    engine = eng(workers=4, budget_mib=20000 / 2**20)
    peak = [0]
    real = _Lanes.decode

    def watching(self: Any, data: bytes, alloc_stream: Any = None) -> Any:
        peak[0] = max(peak[0], engine.budget.in_use)
        time.sleep(0.02)
        return real(self, data, alloc_stream)

    _Lanes.decode = watching  # type: ignore[method-assign]
    try:
        from frame_engine.messages import EngineRequest

        futures = [engine.submit(EngineRequest(_jpeg(), _plan())) for _ in range(4)]
        statuses = [f.result(timeout=20).status for f in futures]
    finally:
        _Lanes.decode = real  # type: ignore[method-assign]
    assert statuses == ['ok'] * 4
    assert 0 < peak[0] <= engine.budget.total
    assert engine.budget.in_use == 0


def test_a_budget_wait_that_outlasts_its_limit_is_an_overload(eng: Any) -> None:
    from frame_engine.messages import EngineRequest
    from frame_engine.transport import OverloadError

    engine = eng(workers=2, budget_mib=20000 / 2**20, budget_wait_seconds=0.2)
    gate = threading.Event()
    real = _Lanes.decode

    def stuck(self: Any, data: bytes, alloc_stream: Any = None) -> Any:
        gate.wait(5)
        return real(self, data, alloc_stream)

    _Lanes.decode = stuck  # type: ignore[method-assign]
    try:
        first = engine.submit(EngineRequest(_jpeg(), _plan()))
        time.sleep(0.05)
        second = engine.submit(EngineRequest(_jpeg(), _plan()))
        with pytest.raises(OverloadError) as err:
            second.result(timeout=5)
        gate.set()
        assert first.result(timeout=5).status == 'ok'
    finally:
        gate.set()
        _Lanes.decode = real  # type: ignore[method-assign]
    assert err.value.cause == 'budget'
    assert engine.budget.in_use == 0


def test_a_full_region_is_an_overload_that_releases_the_budget(eng: Any) -> None:
    from frame_engine.transport import OverloadError

    engine = eng()
    engine.transport.stage_fails = OverloadError('slab', 'no room')
    with pytest.raises(OverloadError):
        _run(engine)
    assert engine.budget.in_use == 0


def test_the_budget_changes_no_output(eng: Any) -> None:
    wide = eng()
    tight = eng(workers=4, budget_mib=20000 / 2**20)
    for _ in range(3):
        a, b = _run(wide), _run(tight)
        assert a.outputs.keys() == b.outputs.keys()
        for name in a.outputs:
            if name == 'crop_jpegs':
                assert a.outputs[name] == b.outputs[name]
            else:
                assert np.array_equal(a.outputs[name], b.outputs[name]), name


def test_the_attribution_clock_changes_no_output(eng: Any) -> None:
    plain = _run(eng())
    synced = _run(eng(stage_sync=True))
    for name in plain.outputs:
        if name != 'crop_jpegs':
            assert np.array_equal(plain.outputs[name], synced.outputs[name]), name


def test_frames_are_allocated_on_the_shared_arena_stream(eng: Any) -> None:
    engine = eng()
    _run(engine)
    assert _Lanes.seen_streams == [engine.alloc_stream]


def test_the_budget_is_reserved_at_start(eng: Any) -> None:
    eng(budget_mib=2048)
    assert 2048 * 2**20 in eng.cp.allocated


def test_an_engine_that_cannot_reserve_its_budget_fails_to_start(eng: Any) -> None:
    eng.cp.alloc_limit = 1 << 20
    with pytest.raises(eng.cp.cuda.memory.OutOfMemoryError):
        eng(budget_mib=2048)


def test_an_unknown_stage_is_refused() -> None:
    from frame_engine.stage_metrics import StageClock

    with pytest.raises(ValueError, match='unknown frame engine stage'):
        StageClock().add('typo', 1.0)
