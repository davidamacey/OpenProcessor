"""Ingest bounds the megabytes in flight (#241) and keeps per-image pixel work off the event loop."""

from __future__ import annotations

import asyncio
import io
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from curation.ingest_fakes import _make_service
from PIL import Image

from src.services.curation import ingest, ingest_batch
from src.services.curation.ingest_dedup import Resolved
from src.services.curation.ingest_models import IngestResult
from src.utils import cpu_pool
from src.utils.cpu_pool import InflightGate


if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest

    from src.services.curation.ingest_finish import PhotoJob


def _jpeg(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(buf, 'JPEG')
    return buf.getvalue()


class _Service:
    """Just enough of CurationIngestService for run_ingest_batch: the host stage (no engine);
    the photos inside ``ingest_photos`` are the ones the gate admitted."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak_in_flight = 0
        self.detector = SimpleNamespace(secondary_profile=None)
        self.config = SimpleNamespace(ingest_gpu_decode=False)

    async def resolve_contents(self, entries: list[tuple[str, bytes]]) -> list[Resolved]:
        return [Resolved(None, h) for h, _ in entries]

    async def ingest_photos(self, jobs: Sequence[PhotoJob]) -> list[IngestResult]:
        self.in_flight += len(jobs)
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        await asyncio.sleep(0.05)
        self.in_flight -= len(jobs)
        return [
            IngestResult(status='success', image_id=j.image_path, image_path=j.image_path)
            for j in jobs
        ]


def _ingest(service: _Service, images: list[bytes], paths: list[str]):
    return ingest_batch.run_ingest_batch(cast('Any', service), images, paths)


def _run_two_requests(monkeypatch: pytest.MonkeyPatch, capacity: int) -> int:
    gate = InflightGate(capacity)
    monkeypatch.setattr(ingest_batch, 'ingest_gate', lambda: gate)
    service = _Service()

    async def main() -> None:
        await asyncio.gather(
            _ingest(service, [_jpeg(1), _jpeg(2)], ['a', 'b']),
            _ingest(service, [_jpeg(3), _jpeg(4)], ['c', 'd']),
        )

    asyncio.run(main())
    return service.peak_in_flight


def test_requests_past_the_cap_queue_instead_of_working_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _run_two_requests(monkeypatch, capacity=2) == 2


def test_without_a_cap_every_request_works_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_host_stage_decode_runs_off_the_loop_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cpu_pool, 'OFFLOAD_MIN_BYTES', 0)  # these frames are tiny
    threads: list[int] = []
    problems: list[str] = []
    loop_ran = threading.Event()
    real = ingest._decode_image

    def spy(data: bytes):
        threads.append(threading.get_ident())
        # Ordering, not elapsed time (#383): a decode on the loop thread would
        # block the heartbeat, so this wait would time out. The timeout bounds
        # a hang only; it is never the assertion.
        if not problems and not loop_ran.wait(timeout=10):
            problems.append('event loop did not run during a decode')
        return real(data)

    monkeypatch.setattr(ingest, '_decode_image', spy)
    service, _os, _triton = _make_service()

    async def main() -> int:
        async def heartbeat() -> None:
            for _ in range(3):
                await asyncio.sleep(0)
            loop_ran.set()

        beat = asyncio.create_task(heartbeat())
        await service.ingest_batch([_jpeg(i) for i in range(4)], list('abcd'))
        await beat
        return threading.get_ident()

    loop_thread = asyncio.run(main())
    assert threads
    assert loop_thread not in threads
    assert problems == []


def test_an_undecodable_image_fails_alone_in_the_host_stage() -> None:
    service, os_fake, _triton = _make_service()
    result = asyncio.run(service.ingest_batch([b'not an image', _jpeg(5)], ['bad', 'good']))
    assert sorted((r.image_path, r.status) for r in result.results) == [
        ('bad', 'failed'),
        ('good', 'success'),
    ]
    bad = next(r for r in result.results if r.image_path == 'bad')
    assert bad.error_kind == 'decode_failed'
    assert len(os_fake.images) == 1


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
    return service.peak_in_flight


def test_the_gate_counts_decoded_megabytes_not_images_in_the_host_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _peak_for_two_big_images(monkeypatch, capacity_mib=8) == 1  # 6 + 6 does not fit
    assert _peak_for_two_big_images(monkeypatch, capacity_mib=12) == 2


def test_the_gate_counts_only_the_encoded_bytes_of_an_engine_stage_photo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def ready(_service: object) -> bool:
        return True

    monkeypatch.setattr(ingest_batch, 'engine_stage_applies', lambda _service: True)
    monkeypatch.setattr(ingest_batch, 'engine_ready', ready)
    # the API never decodes this photo, so 6 decoded MiB each do not count: 1 + 1 fit in 2
    assert _peak_for_two_big_images(monkeypatch, capacity_mib=2) == 2


def test_mib_rounds_up_and_an_unknown_size_counts_one() -> None:
    assert cpu_pool.mib(1) == 1
    assert cpu_pool.mib(cpu_pool.MIB) == 1
    assert cpu_pool.mib(cpu_pool.MIB + 1) == 2
    assert cpu_pool.mib(None) == 1
