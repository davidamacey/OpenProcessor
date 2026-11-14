"""Batch ingest with the GPU decode (#217): photos at or above the size gate are decoded and
processed by the ``gpu_image_prep`` Triton model; everything else, and every image the GPU decode
declines, takes the CPU path. No GPU here: a fake pool answers for the model."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from curation.ingest_fakes import _jpeg_bytes, _make_service
from src.services.curation import ingest_gpu_decode
from src.services.curation.ingest_policy import EmbeddingPolicy, IngestPolicy


DIM = 3


class _Result:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self.outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray:
        return self.outputs[name]


def _gpu_decode_result(
    width: int, height: int, status: str = 'ok', jpeg: bytes = b'\xff\xd8crop'
) -> _Result:
    if status != 'ok':
        return _Result({'status': np.array([status.encode()], dtype=np.object_)})
    scale = min(320 / width, 320 / height)
    new_w, new_h = round(width * scale), round(height * scale)
    return _Result(
        {
            'status': np.array([b'ok'], dtype=np.object_),
            'size': np.array([width, height], np.int32),
            'letterbox': np.array([scale, (320 - new_w) / 2, (320 - new_h) / 2], np.float64),
            'det_boxes': np.array([[16.0, 16.0, 192.0, 192.0]], np.float32),
            'det_scores': np.array([0.9], np.float32),
            'det_classes': np.array([1], np.int32),
            'whole_embedding': np.array([0.0, 0.0, 2.0], np.float32),
            'crop_embeddings': np.array([[0.0, 3.0, 0.0]], np.float32),
            # frame: n=100, sum=0, squares=500 -> 5.0 ; box: n=10, sum=10, squares=30 -> 2.0
            'blur_sums': np.array([[100, 0, 500], [10, 10, 30]], np.int64),
            'crop_jpegs': np.array([jpeg], dtype=np.object_),
        }
    )


@pytest.fixture(autouse=True)
def _fresh_ready_cache() -> Any:
    ingest_gpu_decode._READY.clear()
    yield
    ingest_gpu_decode._READY.clear()


def _gpu_service(
    tmp_path: Any, *, status: str = 'ok', params: dict[str, str] | None = None, gate: float = 0.1
):
    svc, os_fake, triton = _make_service()
    svc.config = dataclasses.replace(
        svc.config,
        ingest_gpu_decode=True,
        ingest_gpu_min_megapixels=gate,
        crop_cache_dir=tmp_path / 'crops',
    )
    calls: list[dict[str, Any]] = []
    sizes: dict[bytes, tuple[int, int]] = {}

    async def get_model_parameters(model: str) -> dict[str, str]:
        assert model == ingest_gpu_decode.GPU_IMAGE_PREP_MODEL
        return params or {
            'DETECTOR_MODEL': 'primary_end2end',
            'NET_SIZE': '320',
            'LETTERBOX_FILL': '114',
        }

    async def stream_one(model: str, inputs: list) -> _Result:
        assert model == ingest_gpu_decode.GPU_IMAGE_PREP_MODEL
        data, embed = inputs[0]._raw_content, inputs[1]._raw_content
        calls.append({'nbytes': len(data), 'embed': bool(np.frombuffer(embed, np.bool_)[0])})
        w, h = sizes[data]
        return _gpu_decode_result(w, h, status)

    triton.loaded = []

    async def is_model_ready(model: str) -> bool:
        return model in triton.loaded

    async def load_model(model: str) -> None:
        triton.loaded.append(model)

    triton.get_model_parameters = get_model_parameters
    triton.stream_one = stream_one
    triton.is_model_ready = is_model_ready
    triton.load_model = load_model
    return svc, os_fake, triton, calls, sizes


def _register(sizes: dict[bytes, tuple[int, int]], data: bytes, size: tuple[int, int]) -> bytes:
    sizes[data] = size
    return data


@pytest.mark.asyncio
async def test_a_photo_above_the_gate_is_ingested_from_the_gpu_decode(tmp_path) -> None:
    svc, os_fake, triton, calls, sizes = _gpu_service(tmp_path, gate=0.2)
    big = _register(sizes, _jpeg_bytes((640, 480), seed=1), (640, 480))
    small = _jpeg_bytes((320, 240), seed=2)
    result = await svc.ingest_batch([big, small], ['/x/big.jpg', '/x/small.jpg'])
    assert result.summary.successful == 2
    assert len(calls) == 1  # only the photo at or above 0.2 MP went to the GPU decode
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
    assert triton.loaded == [ingest_gpu_decode.GPU_IMAGE_PREP_MODEL]  # loaded on first use, once


@pytest.mark.asyncio
async def test_a_declined_image_takes_the_cpu_path(tmp_path) -> None:
    svc, os_fake, triton, calls, sizes = _gpu_service(tmp_path, status='progressive')
    big = _register(sizes, _jpeg_bytes((640, 480), seed=3), (640, 480))
    result = await svc.ingest_batch([big], ['/x/p.jpg'])
    assert result.summary.successful == 1
    assert len(calls) == 1
    [doc] = list(os_fake.images.values())
    assert doc['pe_embedding'] == pytest.approx([0.0, 1.0, 0.0])  # the CPU fake's vector
    assert triton.calls == ['primary_end2end']


@pytest.mark.parametrize('status', ['gpu_memory', 'over_budget'])
@pytest.mark.asyncio
async def test_a_memory_status_is_counted_and_takes_the_cpu_path(tmp_path, status: str) -> None:
    from prometheus_client import REGISTRY

    def sample() -> float:
        return (
            REGISTRY.get_sample_value(
                'op_decode_route_total', {'route': f'cpu_{status}', 'path': 'ingest'}
            )
            or 0.0
        )

    svc, os_fake, triton, calls, sizes = _gpu_service(tmp_path, status=status)
    big = _register(sizes, _jpeg_bytes((640, 480), seed=4), (640, 480))
    before = sample()
    result = await svc.ingest_batch([big], ['/x/m.jpg'])
    assert result.summary.successful == 1
    assert len(calls) == 1
    [doc] = list(os_fake.images.values())
    assert doc['pe_embedding'] == pytest.approx([0.0, 1.0, 0.0])  # the CPU fake's vector
    assert triton.calls == ['primary_end2end']
    assert sample() == before + 1


@pytest.mark.asyncio
async def test_a_gpu_decode_built_for_another_detector_is_not_used(tmp_path) -> None:
    params = {'DETECTOR_MODEL': 'other_model', 'NET_SIZE': '320', 'LETTERBOX_FILL': '114'}
    svc, _os, _triton, calls, sizes = _gpu_service(tmp_path, params=params)
    big = _register(sizes, _jpeg_bytes((640, 480), seed=4), (640, 480))
    result = await svc.ingest_batch([big], ['/x/q.jpg'])
    assert result.summary.successful == 1
    assert calls == []


@pytest.mark.asyncio
async def test_lazy_embedding_asks_for_no_crop_vectors(tmp_path) -> None:
    svc, os_fake, _triton, calls, sizes = _gpu_service(tmp_path)
    svc.policy = IngestPolicy(embedding=EmbeddingPolicy(mode='lazy'))
    big = _register(sizes, _jpeg_bytes((640, 480), seed=5), (640, 480))
    await svc.ingest_batch([big], ['/x/l.jpg'])
    [call] = calls
    assert call['embed'] is False
    [item] = list(os_fake.items.values())
    assert 'pe_embedding' not in item


@pytest.mark.asyncio
async def test_gpu_decode_off_never_calls_the_model(tmp_path) -> None:
    svc, _os, triton, calls, sizes = _gpu_service(tmp_path)
    svc.config = dataclasses.replace(svc.config, ingest_gpu_decode=False)
    big = _register(sizes, _jpeg_bytes((640, 480), seed=6), (640, 480))
    await svc.ingest_batch([big], ['/x/o.jpg'])
    assert calls == []
    assert triton.calls == ['primary_end2end']
    assert triton.loaded == []


def test_the_gpu_decode_is_on_by_default_with_the_three_megapixel_gate() -> None:
    from src.config.curation import CurationConfig

    cfg = CurationConfig()
    assert cfg.ingest_gpu_decode is True
    assert cfg.ingest_gpu_min_megapixels == 3.0


@pytest.mark.asyncio
async def test_a_model_that_fails_to_load_leaves_ingest_on_the_cpu_path(tmp_path) -> None:
    """Triton out of memory for gpu_image_prep: the load raises, every photo is ingested on the
    CPU and the detector is used as before. A failure is a transient, not a verdict, so the
    next batch asks again (see the recovery test below)."""
    svc, os_fake, triton, calls, sizes = _gpu_service(tmp_path)
    attempts: list[str] = []

    async def load_model(model: str) -> None:
        attempts.append(model)
        raise RuntimeError('failed to load: CUDA out of memory')

    triton.load_model = load_model
    for name, seed in (('a', 7), ('b', 8)):
        big = _register(sizes, _jpeg_bytes((640, 480), seed=seed), (640, 480))
        result = await svc.ingest_batch([big], [f'/x/{name}.jpg'])
        assert result.summary.successful == 1
    assert calls == []  # the GPU decode was never called
    assert attempts == [ingest_gpu_decode.GPU_IMAGE_PREP_MODEL] * 2
    assert triton.calls == ['primary_end2end', 'primary_end2end']
    assert len(os_fake.images) == 2


@pytest.mark.asyncio
async def test_a_transient_probe_failure_is_not_cached(tmp_path) -> None:
    svc, _os, triton, _calls, _sizes = _gpu_service(tmp_path)
    real_load = triton.load_model
    failures = [True]

    async def flaky_load(model: str) -> None:
        if failures:
            failures.pop()
            raise RuntimeError('triton restarting')
        await real_load(model)

    triton.load_model = flaky_load
    assert await ingest_gpu_decode.gpu_decode_ready(svc) is False
    assert await ingest_gpu_decode.gpu_decode_ready(svc) is True


@pytest.mark.asyncio
async def test_a_verdict_is_cached_across_lookups(tmp_path) -> None:
    svc, _os, triton, _calls, _sizes = _gpu_service(tmp_path)
    probes: list[str] = []
    real = triton.get_model_parameters

    async def counting(model: str) -> dict[str, str]:
        probes.append(model)
        return await real(model)

    triton.get_model_parameters = counting
    assert await ingest_gpu_decode.gpu_decode_ready(svc) is True
    assert await ingest_gpu_decode.gpu_decode_ready(svc) is True
    assert len(probes) == 1


def _routes(result: Any, os_fake: Any) -> dict[str, str | None]:
    """Route by image path from the results, checked against the stored images docs."""
    stored = {d['image_path']: d.get('decode_route') for d in os_fake.images.values()}
    by_path = {r.image_path: r.decode_route for r in result.results}
    assert {p: r for p, r in by_path.items() if r is not None} == stored
    return by_path


@pytest.mark.asyncio
async def test_every_route_decision_is_on_the_result_and_the_image_doc(tmp_path) -> None:
    svc, os_fake, _triton, _calls, sizes = _gpu_service(tmp_path, gate=0.2)
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
    svc, os_fake, _triton, _calls, _sizes = _gpu_service(tmp_path)
    result = await svc.ingest_batch([b'garbage'], ['/x/g.bin'])
    assert result.results[0].status == 'failed'
    assert result.results[0].decode_route is None  # failed before decode: no route stored
    assert not os_fake.images


@pytest.mark.parametrize('status', ['progressive', 'not_jpeg', 'decoder', 'gpu_memory'])
@pytest.mark.asyncio
async def test_a_declined_status_is_the_route_on_result_doc_and_summary(
    tmp_path, status: str
) -> None:
    svc, os_fake, _triton, _calls, sizes = _gpu_service(tmp_path, status=status)
    big = _register(sizes, _jpeg_bytes((640, 480), seed=13), (640, 480))
    result = await svc.ingest_batch([big], ['/x/d.jpg'])
    assert _routes(result, os_fake) == {'/x/d.jpg': f'cpu_{status}'}
    assert result.summary.decode_routes == {f'cpu_{status}': 1}


@pytest.mark.asyncio
async def test_gpu_decode_off_is_cpu_gpu_off_on_every_image_and_is_counted(tmp_path) -> None:
    from prometheus_client import REGISTRY

    def sample() -> float:
        return (
            REGISTRY.get_sample_value(
                'op_decode_route_total', {'route': 'cpu_gpu_off', 'path': 'ingest'}
            )
            or 0.0
        )

    svc, os_fake, _triton, calls, sizes = _gpu_service(tmp_path)
    svc.config = dataclasses.replace(svc.config, ingest_gpu_decode=False)
    big = _register(sizes, _jpeg_bytes((640, 480), seed=14), (640, 480))
    small = _jpeg_bytes((320, 240), seed=15)
    before = sample()
    result = await svc.ingest_batch([big, small], ['/x/a.jpg', '/x/b.jpg'])
    assert calls == []
    assert _routes(result, os_fake) == {'/x/a.jpg': 'cpu_gpu_off', '/x/b.jpg': 'cpu_gpu_off'}
    assert result.summary.decode_routes == {'cpu_gpu_off': 2}
    assert sample() == before + 2


@pytest.mark.asyncio
async def test_a_failed_gpu_call_falls_back_and_says_cpu_call_failed(tmp_path) -> None:
    svc, os_fake, triton, _calls, sizes = _gpu_service(tmp_path)

    async def broken(_model: str, _inputs: list) -> _Result:
        raise RuntimeError('stream closed without a response')

    triton.stream_one = broken
    big = _register(sizes, _jpeg_bytes((640, 480), seed=16), (640, 480))
    result = await svc.ingest_batch([big], ['/x/f.jpg'])
    assert result.summary.successful == 1
    assert triton.calls == ['primary_end2end']  # the CPU path ran
    assert _routes(result, os_fake) == {'/x/f.jpg': 'cpu_call_failed'}


@pytest.mark.asyncio
async def test_the_single_image_path_is_cpu_single_and_stored(tmp_path) -> None:
    svc, os_fake, _triton, calls, _sizes = _gpu_service(tmp_path)
    result = await svc.ingest_one(_jpeg_bytes((640, 480), seed=17), '/x/one.jpg')
    assert calls == []
    assert result.decode_route == 'cpu_single'
    [doc] = list(os_fake.images.values())
    assert doc['decode_route'] == 'cpu_single'


@pytest.mark.asyncio
async def test_a_duplicate_has_no_route_and_the_doc_keeps_its_first_route(tmp_path) -> None:
    svc, os_fake, _triton, _calls, sizes = _gpu_service(tmp_path)
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
