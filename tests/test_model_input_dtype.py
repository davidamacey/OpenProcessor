"""A detector request carries the ``images`` datatype the model declares.

Built-in engines declare UINT8 and scale in the graph; a promoted project model declares
FP32 (RGB in [0, 1]) and answered 500 when sent uint8 pixels.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from PIL import Image

from src.clients import triton_client as tc
from src.clients.triton_pool_core import AsyncTritonPool, images_input
from src.services.curation.ingest_detect import WholeImageDetector


if TYPE_CHECKING:
    from collections.abc import Callable


def _tensor_values(inp: Any) -> np.ndarray:
    dtype = np.uint8 if inp.datatype() == 'UINT8' else np.float32
    return np.frombuffer(inp._get_content(), dtype=dtype).reshape(inp.shape())


class TestImagesInput:
    PIXELS = np.random.default_rng(1).integers(0, 256, (2, 3, 8, 8), dtype=np.uint8)

    def test_uint8_is_unchanged(self) -> None:
        inp = images_input(self.PIXELS, 'UINT8')
        assert inp.datatype() == 'UINT8'
        assert np.array_equal(_tensor_values(inp), self.PIXELS)

    def test_fp32_is_the_same_pixels_over_255(self) -> None:
        inp = images_input(self.PIXELS, 'FP32')
        assert inp.datatype() == 'FP32'
        assert inp.shape() == [2, 3, 8, 8]
        assert np.allclose(_tensor_values(inp), self.PIXELS.astype(np.float32) / 255.0)

    @pytest.mark.parametrize('datatype', ['FP16', 'INT8', 'BF16', ''])
    def test_other_datatypes_fail_closed(self, datatype: str) -> None:
        with pytest.raises(ValueError, match=datatype or 'images'):
            images_input(self.PIXELS, datatype)


def _sync_client(datatype: str, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[Any]]:
    sent: list[Any] = []

    class _Adapter:
        requested_outputs = ('o',)

        def parse(self, _r: Any, batch_size: int) -> list[dict[str, Any]]:
            return [{'num_dets': 0} for _ in range(batch_size)]

    metadata = SimpleNamespace(
        inputs=[SimpleNamespace(name='images', shape=[-1, 3, 64, 64], datatype=datatype)],
        outputs=[],
    )
    client = tc.TritonClient.__new__(tc.TritonClient)
    client.input_size = 640
    client._model_info = {}
    client.client = SimpleNamespace(get_model_metadata=lambda _n: metadata)
    monkeypatch.setattr(tc, 'resolve_adapter', lambda _m: _Adapter())
    monkeypatch.setattr(tc, 'InferRequestedOutput', lambda name: name)
    monkeypatch.setattr(
        client, '_infer_with_retry', lambda _n, inputs, _o: sent.append(inputs[0]) or object()
    )
    return client, sent


class TestSyncClient:
    IMG = np.random.default_rng(2).integers(0, 256, (50, 100, 3), dtype=np.uint8)

    def test_uint8_model_gets_uint8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, sent = _sync_client('UINT8', monkeypatch)
        client.infer_yolo_end2end_batch([self.IMG], 'm')
        assert sent[0].datatype() == 'UINT8'

    def test_fp32_model_gets_the_same_letterbox_scaled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        u8_client, u8_sent = _sync_client('UINT8', monkeypatch)
        u8_client.infer_yolo_end2end_batch([self.IMG], 'm')
        fp_client, fp_sent = _sync_client('FP32', monkeypatch)
        fp_client.infer_yolo_end2end_batch([self.IMG], 'm')
        assert fp_sent[0].datatype() == 'FP32'
        assert np.allclose(
            _tensor_values(fp_sent[0]), _tensor_values(u8_sent[0]).astype(np.float32) / 255.0
        )

    def test_unknown_datatype_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, sent = _sync_client('FP16', monkeypatch)
        with pytest.raises(ValueError, match='FP16'):
            client.infer_yolo_end2end_batch([self.IMG], 'm')
        assert sent == []


class _MetaClient:
    def __init__(self, datatype: str) -> None:
        self.datatype = datatype
        self.calls = 0

    async def get_model_metadata(self, _name: str, **_kw: Any) -> dict[str, Any]:
        self.calls += 1
        return {'inputs': [{'name': 'images', 'datatype': self.datatype, 'shape': [-1, 3, 64, 64]}]}


def _pool(datatype: str, monkeypatch: pytest.MonkeyPatch) -> tuple[AsyncTritonPool, _MetaClient]:
    async def _loaded(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr('src.clients.triton_pool_core.ensure_loaded', _loaded)
    pool = AsyncTritonPool(pool_size=1)
    meta = _MetaClient(datatype)
    pool._clients = [meta]  # type: ignore[list-item]
    pool._lock = asyncio.Lock()
    pool._initialized = True
    return pool, meta


class TestPoolDatatype:
    def test_reads_images_datatype_and_caches_within_the_ttl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pool, meta = _pool('FP32', monkeypatch)

        async def run() -> list[str]:
            return [await pool.get_model_input_datatype('m') for _ in range(2)]

        assert asyncio.run(run()) == ['FP32', 'FP32']
        assert meta.calls == 1

    def test_reread_after_the_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pool, meta = _pool('UINT8', monkeypatch)
        now = [1000.0]
        monkeypatch.setattr('src.clients.triton_pool_core.time.monotonic', lambda: now[0])

        async def run() -> None:
            await pool.get_model_input_datatype('m')
            meta.datatype = 'FP32'
            now[0] += 31
            assert await pool.get_model_input_datatype('m') == 'FP32'

        asyncio.run(run())
        assert meta.calls == 2


class _IngestPool:
    """Fake pool: declares a datatype, records the request, answers an empty detection."""

    def __init__(self, datatype: str) -> None:
        self.datatype = datatype
        self.inputs: list[Any] = []

    async def get_model_input_datatype(self, _model: str) -> str:
        return self.datatype

    async def infer(self, _model: str, inputs: list[Any], **_kw: Any) -> Any:
        self.inputs.extend(inputs)
        n = inputs[0].shape()[0]
        outputs = {
            'output0': np.zeros((n, 10, 6), dtype=np.float32),
        }
        return SimpleNamespace(as_numpy=lambda name: outputs[name])


def _detector(pool: _IngestPool) -> WholeImageDetector:
    profile = SimpleNamespace(
        input_size=64,
        letterbox_fill=(114, 114, 114),
        batch_limit=2,
        detector_model='det',
        confidence_floor=0.5,
        assigns_class=False,
    )
    return WholeImageDetector(
        triton_pool=pool, registry=None, profile=profile, secondary_profile=profile
    )  # type: ignore[arg-type]


_IMGS = [Image.new('RGB', (80, 40), (10 * i + 10, 50, 200)) for i in range(3)]
_CALLS: dict[str, Callable[[WholeImageDetector], Any]] = {
    'primary': lambda d: d.run_primary(_IMGS[0]),
    'primary_batch': lambda d: d.run_primary_batch(_IMGS),
    'secondary': lambda d: d.run_secondary_raw(_IMGS[0]),
    'secondary_batch': lambda d: d.run_secondary_raw_batch(_IMGS),
}


@pytest.mark.parametrize('call', list(_CALLS))
class TestIngestDetector:
    def test_fp32_model_gets_fp32_equal_to_uint8_over_255(self, call: str) -> None:
        u8, fp = _IngestPool('UINT8'), _IngestPool('FP32')
        asyncio.run(_CALLS[call](_detector(u8)))
        asyncio.run(_CALLS[call](_detector(fp)))
        assert {i.datatype() for i in u8.inputs} == {'UINT8'}
        assert {i.datatype() for i in fp.inputs} == {'FP32'}
        for a, b in zip(u8.inputs, fp.inputs, strict=True):
            assert np.allclose(_tensor_values(b), _tensor_values(a).astype(np.float32) / 255.0)

    def test_unknown_datatype_raises(self, call: str) -> None:
        pool = _IngestPool('FP16')
        with pytest.raises(ValueError, match='FP16'):
            asyncio.run(_CALLS[call](_detector(pool)))
        assert pool.inputs == []
