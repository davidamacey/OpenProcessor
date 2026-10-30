"""Ingest keeps per-image pixel work off the event loop and bounds the images in flight (#241)."""

from __future__ import annotations

import asyncio
import io
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from src.services.curation import ingest_batch
from src.services.curation.ingest_models import IngestResult
from src.utils import cpu_pool
from src.utils.cpu_pool import InflightGate


if TYPE_CHECKING:
    import pytest


def _jpeg(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(buf, 'JPEG')
    return buf.getvalue()


class _Service:
    """Just enough of CurationIngestService for run_ingest_batch."""

    def __init__(self) -> None:
        self.in_detector = 0
        self.peak_in_detector = 0
        self.detector = SimpleNamespace(
            secondary_profile=None, run_primary_batch=self._run_primary_batch
        )
        self.config = SimpleNamespace(ingest_gpu_decode=False)

    async def _run_primary_batch(self, imgs: list[Image.Image]) -> list[list]:
        self.in_detector += len(imgs)
        self.peak_in_detector = max(self.peak_in_detector, self.in_detector)
        await asyncio.sleep(0.05)
        self.in_detector -= len(imgs)
        return [[] for _ in imgs]

    async def _check_duplicates_msearch(self, hashes: list[str]) -> dict[str, str | None]:
        return dict.fromkeys(hashes)

    async def ingest_one(self, _bytes: bytes, path: str, **_kw) -> IngestResult:
        return IngestResult(status='success', image_id=path, image_path=path)

    async def refresh_indexes(self) -> None:
        return None


def _ingest(service: _Service, images: list[bytes], paths: list[str]):
    return ingest_batch.run_ingest_batch(cast('Any', service), images, paths)


def _run_two_requests(monkeypatch: pytest.MonkeyPatch, capacity: int) -> int:
    gate = InflightGate(capacity)
    monkeypatch.setattr(ingest_batch, 'ingest_gate', lambda: gate)
    service = _Service()

    async def main() -> None:
        await asyncio.gather(
            _ingest(
                service,
                [_jpeg(1), _jpeg(2)],
                ['a', 'b'],
            ),
            _ingest(
                service,
                [_jpeg(3), _jpeg(4)],
                ['c', 'd'],
            ),
        )

    asyncio.run(main())
    return service.peak_in_detector


def test_requests_past_the_cap_queue_instead_of_decoding_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _run_two_requests(monkeypatch, capacity=2) == 2


def test_without_a_cap_every_request_decodes_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_two_requests(monkeypatch, capacity=8) == 4


def test_waiting_for_the_gate_is_recorded_as_a_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    from prometheus_client import REGISTRY

    def count() -> float:
        return (
            REGISTRY.get_sample_value(
                'op_pipeline_stage_seconds_count', {'stage': 'admission_wait'}
            )
            or 0.0
        )

    before = count()
    _run_two_requests(monkeypatch, capacity=2)
    assert count() - before == 2


def test_batch_decode_runs_off_the_loop_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation import ingest_helpers

    monkeypatch.setattr(cpu_pool, 'OFFLOAD_MIN_BYTES', 0)  # these frames are tiny
    threads: list[int] = []
    problems: list[str] = []
    loop_ran = threading.Event()
    real = ingest_helpers._decode_image

    def spy(data: bytes):
        threads.append(threading.get_ident())
        # Ordering, not elapsed time (#383): a decode on the loop thread would
        # block the heartbeat, so this wait would time out. The timeout bounds
        # a hang only; it is never the assertion.
        if not problems and not loop_ran.wait(timeout=10):
            problems.append('event loop did not run during a decode')
        return real(data)

    monkeypatch.setattr(ingest_helpers, '_decode_image', spy)
    service = _Service()

    async def main() -> int:
        async def heartbeat() -> None:
            for _ in range(3):
                await asyncio.sleep(0)
            loop_ran.set()

        beat = asyncio.create_task(heartbeat())
        await _ingest(
            service,
            [_jpeg(i) for i in range(4)],
            list('abcd'),
        )
        await beat
        return threading.get_ident()

    loop_thread = asyncio.run(main())
    assert threads
    assert loop_thread not in threads
    assert problems == []


def test_an_undecodable_image_is_left_for_ingest_one_to_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest_batch, 'ingest_gate', lambda: InflightGate(4))
    service = _Service()
    result = asyncio.run(
        _ingest(
            service,
            [b'not an image', _jpeg(5)],
            ['bad', 'good'],
        )
    )
    assert [r.status for r in result.results] == ['success', 'success']
    assert service.peak_in_detector == 1  # only the decodable image reached the detector


def _big_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (1500, 1200), (90, 120, 150)).save(buf, 'JPEG')  # 5.4 MiB decoded
    return buf.getvalue()


def _peak_for_two_big_images(monkeypatch: pytest.MonkeyPatch, capacity_mib: int) -> int:
    gate = InflightGate(capacity_mib)
    monkeypatch.setattr(ingest_batch, 'ingest_gate', lambda: gate)
    service = _Service()
    big = _big_jpeg()

    async def main() -> None:
        await asyncio.gather(
            _ingest(service, [big], ['a']),
            _ingest(service, [big], ['b']),
        )

    asyncio.run(main())
    return service.peak_in_detector


def test_the_gate_counts_decoded_megabytes_not_images(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _peak_for_two_big_images(monkeypatch, capacity_mib=8) == 1  # 6 + 6 does not fit
    assert _peak_for_two_big_images(monkeypatch, capacity_mib=12) == 2


def test_mib_rounds_up_and_an_unknown_size_counts_one() -> None:
    assert cpu_pool.mib(1) == 1
    assert cpu_pool.mib(cpu_pool.MIB) == 1
    assert cpu_pool.mib(cpu_pool.MIB + 1) == 2
    assert cpu_pool.mib(None) == 1
