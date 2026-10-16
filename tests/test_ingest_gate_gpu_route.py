"""The ingest admission gate charges what an image holds in the API process.

A photo the GPU decode takes is never decoded on the host (only boxes, vectors and crop JPEGs
come back), so it holds its encoded bytes; charging its decoded size serialised 20 MP requests
inside one process while the GPU sat half idle.
"""

from __future__ import annotations

import asyncio
import io
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from PIL import Image

from src.services.curation import ingest_batch, ingest_gpu_decode
from src.services.curation.ingest_models import IngestResult
from src.utils.cpu_pool import InflightGate, mib


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import pytest


def _photo(shade: int = 90, width: int = 1500, height: int = 1200) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (width, height), (shade, 120, 150)).save(buf, 'JPEG')
    return buf.getvalue()


class _RecordingGate(InflightGate):
    def __init__(self) -> None:
        super().__init__(10_000)
        self.held: list[int] = []

    @asynccontextmanager
    async def hold(self, units: int) -> AsyncIterator[None]:
        self.held.append(units)
        async with super().hold(units):
            yield


class _Service:
    def __init__(self, *, gpu_min_mp: float) -> None:
        self.detector = SimpleNamespace(secondary_profile=None, run_primary_batch=self._primary)
        self.config = SimpleNamespace(ingest_gpu_decode=True, ingest_gpu_min_megapixels=gpu_min_mp)
        self.policy = SimpleNamespace(embedding=SimpleNamespace(mode='all'))

    async def _primary(self, imgs: list[Image.Image]) -> list[list]:
        return [[] for _ in imgs]

    async def _check_duplicates_msearch(self, hashes: list[str]) -> dict[str, str | None]:
        return dict.fromkeys(hashes)

    async def ingest_one(self, _bytes: bytes, path: str, **_kw: Any) -> IngestResult:
        return IngestResult(status='success', image_id=path, image_path=path)

    async def refresh_indexes(self) -> None:
        return None


def _units_held(
    monkeypatch: pytest.MonkeyPatch, *, gpu_ready: bool, gpu_min_mp: float
) -> list[int]:
    gate = _RecordingGate()
    monkeypatch.setattr(ingest_batch, 'ingest_gate', lambda: gate)

    async def ready(_service: Any) -> bool:
        return gpu_ready

    async def refuse(_service: Any, _data: bytes, **_kw: Any) -> str:
        return 'call_failed'  # the CPU path decodes it; only the charge is under test

    monkeypatch.setattr(ingest_gpu_decode, 'gpu_decode_ready', ready)
    monkeypatch.setattr(ingest_gpu_decode, 'run_gpu_decode', refuse)
    photos = [_photo(10), _photo(200)]
    asyncio.run(
        ingest_batch.run_ingest_batch(
            cast('Any', _Service(gpu_min_mp=gpu_min_mp)), photos, ['a', 'b']
        )
    )
    return gate.held


def test_a_gpu_decoded_photo_is_charged_its_encoded_size(monkeypatch: pytest.MonkeyPatch) -> None:
    encoded = mib(len(_photo()))
    assert _units_held(monkeypatch, gpu_ready=True, gpu_min_mp=1.0) == [2 * encoded]


def test_a_cpu_decoded_photo_is_charged_its_decoded_size(monkeypatch: pytest.MonkeyPatch) -> None:
    decoded = mib(1500 * 1200 * 3)
    assert _units_held(monkeypatch, gpu_ready=True, gpu_min_mp=10.0) == [2 * decoded]
    assert _units_held(monkeypatch, gpu_ready=False, gpu_min_mp=1.0) == [2 * decoded]


def _stage_count(stage: str) -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value('op_pipeline_stage_seconds_count', {'stage': stage}) or 0.0


def test_the_gpu_decode_call_is_timed_as_its_own_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    before = _stage_count('gpu_decode')
    _units_held(monkeypatch, gpu_ready=True, gpu_min_mp=1.0)
    assert _stage_count('gpu_decode') - before == 2
