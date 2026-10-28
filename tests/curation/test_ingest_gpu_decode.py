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
            'det_boxes': np.array([[0.05, 0.05, 0.6, 0.6]], np.float32),
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
    CPU, the detector is used as before and the load is not retried for every batch."""
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
    assert attempts == [ingest_gpu_decode.GPU_IMAGE_PREP_MODEL]
    assert triton.calls == ['primary_end2end', 'primary_end2end']
    assert len(os_fake.images) == 2
