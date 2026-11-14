"""Batch ingest with the GPU decode (#217): photos at or above the size gate are decoded and
processed by the frame engine (one request per photo, protocol version 1); everything else, and
every image the engine declines, takes the CPU path. No GPU here: a fake pool on the engine's
address answers, and the real client (``src/clients/frame_engine.py``) runs in between."""

from __future__ import annotations

import dataclasses
import json
import struct
from typing import Any

import grpc
import numpy as np
import pytest
from tritonclient.utils import InferenceServerException

from curation.ingest_fakes import _jpeg_bytes, _make_service
from src.domain import frame_engine_wire as wire
from src.services.curation import ingest_gpu_decode
from src.services.curation.ingest_policy import EmbeddingPolicy, IngestPolicy


DIM = 3


class _Result:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self.outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray | None:
        return self.outputs.get(name)


def _text(value: str) -> np.ndarray:
    return np.array([value.encode()], dtype=np.object_)


def _one_bytes(inp: Any) -> bytes:
    """The one element of a BYTES input (KServe: a 4-byte length, then the bytes)."""
    raw = inp._raw_content
    (n,) = struct.unpack_from('<I', raw)
    return bytes(raw[4 : 4 + n])


def _engine_result(
    plan: wire.EnginePlan,
    width: int,
    height: int,
    status: str = 'ok',
    jpeg: bytes = b'\xff\xd8crop',
    failed_branch: str | None = None,
) -> _Result:
    if status != 'ok':
        failed = {b: {'status': 'error', 'reason': 'image_failed'} for b in plan.branches}
        return _Result(
            {
                'status': _text(status),
                'decode_route': _text('none'),
                'branch_status': _text(json.dumps(failed)),
            }
        )
    branches: dict[str, dict[str, str]] = {b: {'status': 'ok'} for b in plan.branches}
    if failed_branch is not None:
        branches[failed_branch] = {'status': 'error', 'reason': 'model_error'}
    scale = min(320 / width, 320 / height)
    new_w, new_h = round(width * scale), round(height * scale)
    out = {
        'status': _text('ok'),
        'decode_route': _text('gpu'),
        'branch_status': _text(json.dumps(branches)),
        'size': np.array([width, height], np.int32),
        'letterbox': np.array([scale, (320 - new_w) // 2, (320 - new_h) // 2], np.float64),
        'det_boxes': np.array([[16.0, 16.0, 192.0, 192.0]], np.float32),
        'det_scores': np.array([0.9], np.float32),
        'det_classes': np.array([1], np.int32),
        'whole_embedding': np.array([0.0, 0.0, 2.0], np.float32),
        # frame: n=100, sum=0, squares=500 -> 5.0 ; box: n=10, sum=10, squares=30 -> 2.0
        'blur_sums': np.array([[100, 0, 500], [10, 10, 30]], np.int64),
        'crop_jpegs': np.array([jpeg], dtype=np.object_),
    }
    if plan.runs('embed_crops'):
        out['crop_embeddings'] = np.array([[0.0, 3.0, 0.0]], np.float32)
        out['crop_embedded'] = np.array([True])
    return _Result(out)


@pytest.fixture(autouse=True)
def _fresh_ready_cache() -> Any:
    ingest_gpu_decode._READY.clear()
    yield
    ingest_gpu_decode._READY.clear()


class _Engine:
    """The pool on the frame engine's address: answers each photo as the engine would."""

    def __init__(self, status: str, sizes: dict[bytes, tuple[int, int]]) -> None:
        self.status = status
        self.sizes = sizes
        self.calls: list[dict[str, Any]] = []
        self.ready = True
        self.readiness_probes = 0
        self.failed_branch: str | None = None
        self.raises: Exception | None = None

    async def is_model_ready(self, model: str) -> bool:
        assert model == wire.MODEL_NAME
        self.readiness_probes += 1
        return self.ready

    async def infer(self, model: str, inputs: list, outputs: list, **_k: Any) -> _Result:
        assert model == wire.MODEL_NAME
        data = _one_bytes(inputs[0])
        plan = wire.parse_plan(_one_bytes(inputs[1]), has_boxes=len(inputs) > 2)
        self.calls.append({'nbytes': len(data), 'plan': plan})
        if self.raises is not None:
            raise self.raises
        w, h = self.sizes[data]
        return _engine_result(plan, w, h, self.status, failed_branch=self.failed_branch)


def _gpu_service(tmp_path: Any, *, status: str = 'ok', gate: float = 0.1):
    svc, os_fake, triton = _make_service()
    svc.config = dataclasses.replace(
        svc.config,
        ingest_gpu_decode=True,
        ingest_gpu_min_megapixels=gate,
        crop_cache_dir=tmp_path / 'crops',
    )
    sizes: dict[bytes, tuple[int, int]] = {}
    engine = _Engine(status, sizes)
    svc.frame_engine = engine
    triton.loaded = []

    async def is_model_ready(model: str) -> bool:
        return model in triton.loaded

    async def load_model(model: str) -> None:
        triton.loaded.append(model)

    triton.is_model_ready = is_model_ready
    triton.load_model = load_model
    return svc, os_fake, triton, engine, sizes


def _register(sizes: dict[bytes, tuple[int, int]], data: bytes, size: tuple[int, int]) -> bytes:
    sizes[data] = size
    return data


def _route_sample(route: str) -> float:
    from prometheus_client import REGISTRY

    return (
        REGISTRY.get_sample_value('op_decode_route_total', {'route': route, 'path': 'ingest'})
        or 0.0
    )


@pytest.mark.asyncio
async def test_a_photo_above_the_gate_is_ingested_from_the_gpu_decode(tmp_path) -> None:
    svc, os_fake, triton, engine, sizes = _gpu_service(tmp_path, gate=0.2)
    big = _register(sizes, _jpeg_bytes((640, 480), seed=1), (640, 480))
    small = _jpeg_bytes((320, 240), seed=2)
    result = await svc.ingest_batch([big, small], ['/x/big.jpg', '/x/small.jpg'])
    assert result.summary.successful == 2
    assert len(engine.calls) == 1  # only the photo at or above 0.2 MP went to the GPU decode
    docs = {d['image_path']: d for d in os_fake.images.values()}
    assert docs['/x/big.jpg']['pe_embedding'] == pytest.approx([0.0, 0.0, 1.0])  # normalized
    assert docs['/x/small.jpg']['pe_embedding'] == pytest.approx([0.0, 1.0, 0.0])  # CPU fake
    [(crop_id, item)] = [
        (k, d) for k, d in os_fake.items.items() if d['image_path'] == '/x/big.jpg'
    ]
    assert item['pe_embedding'] == pytest.approx([0.0, 1.0, 0.0])
    assert item['blur_full_var'] == 5.0
    assert item['blur_lap_var'] == 2.0
    assert item['class_name'] == 'widget'
    assert (tmp_path / 'crops' / 'default' / f'{crop_id}.jpg').read_bytes() == b'\xff\xd8crop'
    assert triton.calls.count('primary_end2end') == 1  # the CPU detector ran for the small one only
    # the models the engine calls are loaded on first use, once
    assert triton.loaded == ['primary_end2end', 'pe_image_encoder']


@pytest.mark.asyncio
async def test_the_plan_names_the_profile_detector_the_pe_encoder_and_inline_crops(
    tmp_path,
) -> None:
    svc, _os, _triton, engine, sizes = _gpu_service(tmp_path)
    big = _register(sizes, _jpeg_bytes((640, 480), seed=21), (640, 480))
    await svc.ingest_batch([big], ['/x/p.jpg'])
    [call] = engine.calls
    plan = call['plan']
    assert plan.branches == ('detect', 'embed_frame', 'embed_crops')
    assert plan.detector == wire.DetectorSpec('primary_end2end', 320, 114)
    assert plan.embed == wire.EmbedSpec('pe_image_encoder', 336, 32)
    assert plan.crops == 'inline'
    assert plan.embed_classes is None  # every detection gets a crop vector, as before


@pytest.mark.asyncio
async def test_a_declined_image_takes_the_cpu_path(tmp_path) -> None:
    svc, os_fake, triton, engine, sizes = _gpu_service(tmp_path, status='progressive')
    big = _register(sizes, _jpeg_bytes((640, 480), seed=3), (640, 480))
    result = await svc.ingest_batch([big], ['/x/p.jpg'])
    assert result.summary.successful == 1
    assert len(engine.calls) == 1
    [doc] = list(os_fake.images.values())
    assert doc['pe_embedding'] == pytest.approx([0.0, 1.0, 0.0])  # the CPU fake's vector
    assert triton.calls == ['primary_end2end']


@pytest.mark.parametrize('status', ['gpu_memory', 'over_budget'])
@pytest.mark.asyncio
async def test_a_memory_status_is_counted_and_takes_the_cpu_path(tmp_path, status: str) -> None:
    svc, os_fake, triton, engine, sizes = _gpu_service(tmp_path, status=status)
    big = _register(sizes, _jpeg_bytes((640, 480), seed=4), (640, 480))
    before = _route_sample(f'cpu_{status}')
    result = await svc.ingest_batch([big], ['/x/m.jpg'])
    assert result.summary.successful == 1
    assert len(engine.calls) == 1
    [doc] = list(os_fake.images.values())
    assert doc['pe_embedding'] == pytest.approx([0.0, 1.0, 0.0])  # the CPU fake's vector
    assert triton.calls == ['primary_end2end']
    assert _route_sample(f'cpu_{status}') == before + 1


@pytest.mark.asyncio
async def test_an_engine_still_overloaded_after_the_retries_is_cpu_gpu_memory(tmp_path) -> None:
    """The engine answers overload (RESOURCE_EXHAUSTED) instead of waiting past its bounded
    budget wait; once the client's retries are spent the photo takes the CPU path under the
    route the prep model's own memory timeout had."""
    svc, os_fake, triton, engine, sizes = _gpu_service(tmp_path)
    engine.raises = InferenceServerException(
        msg='overload: budget', status=str(grpc.StatusCode.RESOURCE_EXHAUSTED)
    )
    big = _register(sizes, _jpeg_bytes((640, 480), seed=22), (640, 480))
    result = await svc.ingest_batch([big], ['/x/o.jpg'])
    assert result.summary.successful == 1
    assert triton.calls == ['primary_end2end']  # the CPU path ran
    assert _routes(result, os_fake) == {'/x/o.jpg': 'cpu_gpu_memory'}


@pytest.mark.parametrize(
    'error',
    [
        InferenceServerException(msg='boom', status=str(grpc.StatusCode.INTERNAL)),
        OSError('connection refused'),
        RuntimeError('stream closed without a response'),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_engine_call_falls_back_and_says_cpu_call_failed(
    tmp_path, error: Exception
) -> None:
    svc, os_fake, triton, engine, sizes = _gpu_service(tmp_path)
    engine.raises = error
    big = _register(sizes, _jpeg_bytes((640, 480), seed=16), (640, 480))
    result = await svc.ingest_batch([big], ['/x/f.jpg'])
    assert result.summary.successful == 1
    assert triton.calls == ['primary_end2end']  # the CPU path ran
    assert _routes(result, os_fake) == {'/x/f.jpg': 'cpu_call_failed'}


@pytest.mark.parametrize('branch', ['detect', 'embed_frame', 'embed_crops'])
@pytest.mark.asyncio
async def test_a_branch_that_errored_takes_the_cpu_path_as_cpu_call_failed(
    tmp_path, branch: str
) -> None:
    svc, os_fake, triton, engine, sizes = _gpu_service(tmp_path)
    engine.failed_branch = branch
    big = _register(sizes, _jpeg_bytes((640, 480), seed=23), (640, 480))
    result = await svc.ingest_batch([big], ['/x/b.jpg'])
    assert result.summary.successful == 1
    assert triton.calls == ['primary_end2end']
    assert _routes(result, os_fake) == {'/x/b.jpg': 'cpu_call_failed'}


@pytest.mark.asyncio
async def test_a_letterbox_fill_of_three_values_keeps_the_cpu_path(tmp_path) -> None:
    """The engine fills its canvas with one grey; another fill would change the detector input."""
    svc, _os, _triton, engine, sizes = _gpu_service(tmp_path)
    svc.profile = dataclasses.replace(svc.profile, letterbox_fill=(114, 114, 0))
    big = _register(sizes, _jpeg_bytes((640, 480), seed=4), (640, 480))
    result = await svc.ingest_batch([big], ['/x/q.jpg'])
    assert result.summary.successful == 1
    assert engine.calls == []


@pytest.mark.asyncio
async def test_lazy_embedding_asks_for_no_crop_vectors(tmp_path) -> None:
    svc, os_fake, _triton, engine, sizes = _gpu_service(tmp_path)
    svc.policy = IngestPolicy(embedding=EmbeddingPolicy(mode='lazy'))
    big = _register(sizes, _jpeg_bytes((640, 480), seed=5), (640, 480))
    await svc.ingest_batch([big], ['/x/l.jpg'])
    [call] = engine.calls
    assert call['plan'].branches == ('detect', 'embed_frame')
    [item] = list(os_fake.items.values())
    assert 'pe_embedding' not in item


@pytest.mark.asyncio
async def test_gpu_decode_off_never_calls_the_engine(tmp_path) -> None:
    svc, _os, triton, engine, sizes = _gpu_service(tmp_path)
    svc.config = dataclasses.replace(svc.config, ingest_gpu_decode=False)
    big = _register(sizes, _jpeg_bytes((640, 480), seed=6), (640, 480))
    await svc.ingest_batch([big], ['/x/o.jpg'])
    assert engine.calls == []
    assert engine.readiness_probes == 0
    assert triton.calls == ['primary_end2end']
    assert triton.loaded == []


@pytest.mark.asyncio
async def test_a_service_without_an_engine_client_takes_the_cpu_path(tmp_path) -> None:
    svc, os_fake, _triton, _engine, sizes = _gpu_service(tmp_path)
    svc.frame_engine = None
    big = _register(sizes, _jpeg_bytes((640, 480), seed=24), (640, 480))
    result = await svc.ingest_batch([big], ['/x/n.jpg'])
    assert _routes(result, os_fake) == {'/x/n.jpg': 'cpu_gpu_off'}


@pytest.mark.asyncio
async def test_an_engine_that_is_not_ready_keeps_the_batch_on_the_cpu_path_and_is_asked_again(
    tmp_path,
) -> None:
    svc, os_fake, _triton, engine, sizes = _gpu_service(tmp_path)
    engine.ready = False
    big = _register(sizes, _jpeg_bytes((640, 480), seed=25), (640, 480))
    result = await svc.ingest_batch([big], ['/x/r.jpg'])
    assert engine.calls == []
    assert _routes(result, os_fake) == {'/x/r.jpg': 'cpu_gpu_off'}
    engine.ready = True
    assert await ingest_gpu_decode.gpu_decode_ready(svc) is True


def test_the_gpu_decode_is_on_by_default_with_the_three_megapixel_gate() -> None:
    from src.config.curation import CurationConfig

    cfg = CurationConfig()
    assert cfg.ingest_gpu_decode is True
    assert cfg.ingest_gpu_min_megapixels == 3.0


@pytest.mark.asyncio
async def test_a_model_that_fails_to_load_leaves_ingest_on_the_cpu_path(tmp_path) -> None:
    """Triton out of memory for the detector or the encoder the engine calls: the load raises,
    every photo is ingested on the CPU and the detector is used as before. A failure is a
    transient, not a verdict, so the next batch asks again."""
    svc, os_fake, triton, engine, sizes = _gpu_service(tmp_path)
    attempts: list[str] = []

    async def load_model(model: str) -> None:
        attempts.append(model)
        raise InferenceServerException('failed to load: CUDA out of memory')

    triton.load_model = load_model
    for name, seed in (('a', 7), ('b', 8)):
        big = _register(sizes, _jpeg_bytes((640, 480), seed=seed), (640, 480))
        result = await svc.ingest_batch([big], [f'/x/{name}.jpg'])
        assert result.summary.successful == 1
    assert engine.calls == []  # the GPU decode was never called
    assert attempts == ['primary_end2end'] * 2
    assert triton.calls == ['primary_end2end', 'primary_end2end']
    assert len(os_fake.images) == 2


@pytest.mark.asyncio
async def test_a_transient_probe_failure_is_not_cached(tmp_path) -> None:
    svc, _os, triton, _engine, _sizes = _gpu_service(tmp_path)
    real_load = triton.load_model
    failures = [True]

    async def flaky_load(model: str) -> None:
        if failures:
            failures.pop()
            raise InferenceServerException('triton restarting')
        await real_load(model)

    triton.load_model = flaky_load
    assert await ingest_gpu_decode.gpu_decode_ready(svc) is False
    assert await ingest_gpu_decode.gpu_decode_ready(svc) is True


@pytest.mark.asyncio
async def test_a_verdict_is_cached_across_lookups(tmp_path) -> None:
    svc, _os, _triton, engine, _sizes = _gpu_service(tmp_path)
    assert await ingest_gpu_decode.gpu_decode_ready(svc) is True
    assert await ingest_gpu_decode.gpu_decode_ready(svc) is True
    assert engine.readiness_probes == 1


def _routes(result: Any, os_fake: Any) -> dict[str, str | None]:
    """Route by image path from the results, checked against the stored images docs."""
    stored = {d['image_path']: d.get('decode_route') for d in os_fake.images.values()}
    by_path = {r.image_path: r.decode_route for r in result.results}
    assert {p: r for p, r in by_path.items() if r is not None} == stored
    return by_path


@pytest.mark.asyncio
async def test_every_route_decision_is_on_the_result_and_the_image_doc(tmp_path) -> None:
    svc, os_fake, _triton, _engine, sizes = _gpu_service(tmp_path, gate=0.2)
    big = _register(sizes, _jpeg_bytes((640, 480), seed=11), (640, 480))
    small = _jpeg_bytes((320, 240), seed=12)
    not_image = b'not an image at all'
    result = await svc.ingest_batch(
        [big, small, not_image], ['/x/big.jpg', '/x/small.jpg', '/x/bad.bin']
    )
    by_path = _routes(result, os_fake)
    assert by_path == {'/x/big.jpg': 'gpu', '/x/small.jpg': 'cpu_small', '/x/bad.bin': None}
    assert result.summary.decode_routes == {'gpu': 1, 'cpu_small': 1}


@pytest.mark.asyncio
async def test_an_undecodable_file_has_no_route_and_no_doc(tmp_path) -> None:
    svc, os_fake, _triton, _engine, _sizes = _gpu_service(tmp_path)
    result = await svc.ingest_batch([b'garbage'], ['/x/g.bin'])
    assert result.results[0].status == 'failed'
    assert result.results[0].decode_route is None  # failed before decode: no route stored
    assert not os_fake.images


@pytest.mark.parametrize('status', ['progressive', 'not_jpeg', 'decoder', 'gpu_memory'])
@pytest.mark.asyncio
async def test_a_declined_status_is_the_route_on_result_doc_and_summary(
    tmp_path, status: str
) -> None:
    svc, os_fake, _triton, _engine, sizes = _gpu_service(tmp_path, status=status)
    big = _register(sizes, _jpeg_bytes((640, 480), seed=13), (640, 480))
    result = await svc.ingest_batch([big], ['/x/d.jpg'])
    assert _routes(result, os_fake) == {'/x/d.jpg': f'cpu_{status}'}
    assert result.summary.decode_routes == {f'cpu_{status}': 1}


@pytest.mark.asyncio
async def test_gpu_decode_off_is_cpu_gpu_off_on_every_image_and_is_counted(tmp_path) -> None:
    svc, os_fake, _triton, engine, sizes = _gpu_service(tmp_path)
    svc.config = dataclasses.replace(svc.config, ingest_gpu_decode=False)
    big = _register(sizes, _jpeg_bytes((640, 480), seed=14), (640, 480))
    small = _jpeg_bytes((320, 240), seed=15)
    before = _route_sample('cpu_gpu_off')
    result = await svc.ingest_batch([big, small], ['/x/a.jpg', '/x/b.jpg'])
    assert engine.calls == []
    assert _routes(result, os_fake) == {'/x/a.jpg': 'cpu_gpu_off', '/x/b.jpg': 'cpu_gpu_off'}
    assert result.summary.decode_routes == {'cpu_gpu_off': 2}
    assert _route_sample('cpu_gpu_off') == before + 2


@pytest.mark.asyncio
async def test_the_single_image_path_is_cpu_single_and_stored(tmp_path) -> None:
    svc, os_fake, _triton, engine, _sizes = _gpu_service(tmp_path)
    result = await svc.ingest_one(_jpeg_bytes((640, 480), seed=17), '/x/one.jpg')
    assert engine.calls == []
    assert result.decode_route == 'cpu_single'
    [doc] = list(os_fake.images.values())
    assert doc['decode_route'] == 'cpu_single'


@pytest.mark.asyncio
async def test_a_duplicate_has_no_route_and_the_doc_keeps_its_first_route(tmp_path) -> None:
    svc, os_fake, _triton, _engine, sizes = _gpu_service(tmp_path)
    big = _register(sizes, _jpeg_bytes((640, 480), seed=18), (640, 480))
    first = await svc.ingest_batch([big], ['/x/1.jpg'])
    assert first.results[0].decode_route == 'gpu'
    svc.config = dataclasses.replace(svc.config, ingest_gpu_decode=False)
    again = await svc.ingest_batch([big], ['/x/1.jpg'])
    assert again.results[0].status == 'duplicate'
    assert again.results[0].decode_route is None
    assert again.summary.decode_routes == {}
    [doc] = list(os_fake.images.values())
    assert doc['decode_route'] == 'gpu'
