"""The opt-in ingest load profile: models the profile does not preload load on first use."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from src.clients import model_residency as residency


class _AioClient:
    def __init__(self, ready: set[str] | None = None, fail: str | None = None) -> None:
        self.ready = set(ready or ())
        self.fail = fail
        self.loads: list[str] = []

    async def is_model_ready(self, name: str) -> bool:
        return name in self.ready

    async def load_model(self, name: str) -> None:
        await asyncio.sleep(0.01)
        self.loads.append(name)
        if self.fail == name:
            raise RuntimeError(f'failed to load {name}')
        self.ready.add(name)


class _SyncClient:
    def __init__(self, ready: set[str] | None = None, fail: str | None = None) -> None:
        self.ready = set(ready or ())
        self.fail = fail
        self.loads: list[str] = []
        self.lock = threading.Lock()

    def is_model_ready(self, name: str) -> bool:
        return name in self.ready

    def load_model(self, name: str) -> None:
        with self.lock:
            self.loads.append(name)
        if self.fail == name:
            raise RuntimeError(f'failed to load {name}')
        self.ready.add(name)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    residency._ready.clear()
    residency._sync_locks.clear()
    residency._async_locks.clear()
    monkeypatch.delenv('OP_LOAD_PROFILE', raising=False)


def test_the_default_profile_loads_nothing_and_asks_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _SyncClient()
    residency.ensure_loaded_sync(client, 'arcface_w600k_r50')
    assert client.loads == []


def test_an_unknown_profile_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_LOAD_PROFILE', 'ingset')
    with pytest.raises(ValueError, match='OP_LOAD_PROFILE'):
        residency.on_demand()


def test_a_missing_model_is_loaded_once_under_the_ingest_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_LOAD_PROFILE', 'ingest')
    client = _SyncClient()
    residency.ensure_loaded_sync(client, 'scrfd_10g_bnkps')
    residency.ensure_loaded_sync(client, 'scrfd_10g_bnkps')
    assert client.loads == ['scrfd_10g_bnkps']


def test_a_ready_model_is_not_reloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Triton reloads a loaded model on a second load request: never ask for it."""
    monkeypatch.setenv('OP_LOAD_PROFILE', 'ingest')
    client = _SyncClient(ready={'pe_image_encoder'})
    residency.ensure_loaded_sync(client, 'pe_image_encoder')
    assert client.loads == []


def test_a_failed_load_fails_the_call_and_is_retried_next_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_LOAD_PROFILE', 'ingest')
    client = _SyncClient(fail='ocr_pipeline')
    with pytest.raises(RuntimeError, match='ocr_pipeline'):
        residency.ensure_loaded_sync(client, 'ocr_pipeline')
    client.fail = None
    residency.ensure_loaded_sync(client, 'ocr_pipeline')
    assert client.loads == [
        'paddleocr_det_trt',
        'paddleocr_rec_trt',
        'ocr_pipeline',
        'ocr_pipeline',
    ]


def test_a_load_that_raced_another_process_counts_when_the_model_became_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_LOAD_PROFILE', 'ingest')

    class Racing(_SyncClient):
        def load_model(self, name: str) -> None:
            self.ready.add(name)  # another API worker's load finished first
            raise RuntimeError('model is being loaded')

    residency.ensure_loaded_sync(Racing(), 'mobileclip2_s2_image_encoder')


def test_threads_asking_together_load_a_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_LOAD_PROFILE', 'ingest')
    client = _SyncClient()
    threads = [
        threading.Thread(target=residency.ensure_loaded_sync, args=(client, 'arcface_w600k_r50'))
        for _ in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert client.loads == ['arcface_w600k_r50']


@pytest.mark.asyncio
async def test_async_callers_load_a_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_LOAD_PROFILE', 'ingest')
    client = _AioClient()
    await asyncio.gather(*(residency.ensure_loaded(client, 'paddleocr_det_trt') for _ in range(16)))
    assert client.loads == ['paddleocr_det_trt']


@pytest.mark.asyncio
async def test_async_default_profile_asks_nothing() -> None:
    client = _AioClient()
    await residency.ensure_loaded(client, 'paddleocr_det_trt')
    assert client.loads == []


@pytest.mark.asyncio
async def test_the_pool_loads_before_it_infers(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.clients.triton_pool_core import AsyncTritonPool

    monkeypatch.setenv('OP_LOAD_PROFILE', 'ingest')
    pool = AsyncTritonPool(url='unused:1', pool_size=1, max_concurrent=2)
    aio = _AioClient()
    order: list[str] = []

    class Client(_AioClient):
        async def load_model(self, name: str) -> None:
            order.append('load')
            await super().load_model(name)

        async def infer(self, *_a: Any, **_k: Any) -> str:
            order.append('infer')
            return 'result'

    client = Client()
    pool._clients = [client]  # type: ignore[list-item]
    pool._semaphore = asyncio.Semaphore(2)
    pool._lock = asyncio.Lock()
    pool._stats.requests_per_channel[0] = 0
    pool._initialized = True
    assert await pool.infer('scrfd_10g_bnkps', [], []) == 'result'
    assert order == ['load', 'infer']
    assert aio.loads == []


class _InferringClient(_SyncClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def load_model(self, name: str) -> None:
        self.calls.append(f'load {name}')
        super().load_model(name)

    def infer(self, model_name: str, *_a: Any, **_k: Any) -> str:
        assert model_name in self.ready, 'inference reached an unloaded model'
        self.calls.append(f'infer {model_name}')
        return 'result'


def test_the_core_client_loads_a_model_before_it_infers(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.clients.triton_client import TritonClient

    monkeypatch.setenv('OP_LOAD_PROFILE', 'ingest')
    tc = object.__new__(TritonClient)
    tc.client = _InferringClient()  # type: ignore[assignment]
    tc.max_retries, tc.retry_base_delay, tc.retry_max_delay = 0, 0.0, 0.0
    assert tc._infer_with_retry('mobileclip2_s2_image_encoder', [], []) == 'result'
    assert tc.client.calls == [  # type: ignore[attr-defined]
        'load mobileclip2_s2_image_encoder',
        'infer mobileclip2_s2_image_encoder',
    ]


def test_the_face_client_loads_both_face_models(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from src.clients.fast_face_client import FastFaceClient

    monkeypatch.setenv('OP_LOAD_PROFILE', 'ingest')
    ff = object.__new__(FastFaceClient)
    ff.client = _InferringClient()  # type: ignore[assignment]
    ff.scrfd_model, ff.arcface_model = 'scrfd_10g_bnkps', 'arcface_w600k_r50'

    class _Out:
        def as_numpy(self, _name: str) -> np.ndarray:
            return np.ones((1, 4), np.float32)

    ff.client.infer = lambda model_name, **_k: (  # type: ignore[method-assign]
        ff.client.calls.append(f'infer {model_name}'),  # type: ignore[attr-defined]
        _Out(),
    )[1]
    ff._call_arcface(np.zeros((1, 3, 112, 112), np.float32))
    assert ff.client.calls[:2] == ['load arcface_w600k_r50', 'infer arcface_w600k_r50']  # type: ignore[attr-defined]


def test_a_pipeline_model_loads_the_models_it_calls_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """ocr_pipeline calls the OCR detector and recognizer inside Triton: loading only the
    pipeline leaves every OCR request failing with the detector missing."""
    monkeypatch.setenv('OP_LOAD_PROFILE', 'ingest')
    client = _SyncClient()
    residency.ensure_loaded_sync(client, 'ocr_pipeline')
    assert client.loads == ['paddleocr_det_trt', 'paddleocr_rec_trt', 'ocr_pipeline']


@pytest.mark.asyncio
async def test_the_async_path_loads_pipeline_dependencies_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_LOAD_PROFILE', 'ingest')
    client = _AioClient()
    await residency.ensure_loaded(client, 'ocr_pipeline')
    assert client.loads == ['paddleocr_det_trt', 'paddleocr_rec_trt', 'ocr_pipeline']
