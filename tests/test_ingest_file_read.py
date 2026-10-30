"""A batch's source files are read in parallel and off the event loop.

Reading 16 photos of 6.8 MB one after another on the loop thread from a network share held every
other request of the worker (and every awaited OpenSearch write) for the whole read.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from src.utils import file_read


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_files_come_back_in_order_and_a_missing_one_is_an_error_value(tmp_path: Path) -> None:
    (tmp_path / 'a.jpg').write_bytes(b'aa')
    (tmp_path / 'b.jpg').write_bytes(b'bbb')
    out = asyncio.run(
        file_read.read_files(
            [str(tmp_path / 'a.jpg'), str(tmp_path / 'gone.jpg'), str(tmp_path / 'b.jpg')]
        )
    )
    assert out[0] == b'aa'
    assert isinstance(out[1], OSError)
    assert out[2] == b'bbb'


def test_slow_reads_overlap_and_leave_the_loop_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ordering, not elapsed time (#383): eight reads must be in flight at once
    # (a barrier only opens when all eight threads arrive), and the event loop
    # must run while they are (the reads wait for a tick the loop makes). A
    # generous timeout only bounds a hang; it is never the assertion.
    monkeypatch.setenv('OP_INGEST_READ_THREADS', '8')
    file_read.shutdown_read_executor()
    threads: set[int] = set()
    problems: list[str] = []
    barrier = threading.Barrier(8)
    loop_ran = threading.Event()

    def slow(path: str) -> bytes:
        threads.add(threading.get_ident())
        if problems:
            return b'x'
        try:
            barrier.wait(timeout=10)
        except threading.BrokenBarrierError:
            problems.append('reads did not overlap')
        if not loop_ran.wait(timeout=10):
            problems.append('event loop did not run during the reads')
        return b'x'

    monkeypatch.setattr(file_read, '_read_one', slow)

    async def main() -> None:
        async def heartbeat() -> None:
            for _ in range(3):
                await asyncio.sleep(0)
            loop_ran.set()

        beat = asyncio.create_task(heartbeat())
        await file_read.read_files([str(tmp_path / f'{i}.jpg') for i in range(16)])
        await beat

    asyncio.run(main())
    file_read.shutdown_read_executor()
    assert problems == []
    assert threading.get_ident() not in threads
    assert len(threads) == 8


def test_the_read_is_timed_as_its_own_stage(tmp_path: Path) -> None:
    from prometheus_client import REGISTRY

    def count() -> float:
        return (
            REGISTRY.get_sample_value('op_pipeline_stage_seconds_count', {'stage': 'file_read'})
            or 0.0
        )

    (tmp_path / 'a.jpg').write_bytes(b'a')
    before = count()
    asyncio.run(file_read.read_files([str(tmp_path / 'a.jpg')]))
    file_read.shutdown_read_executor()
    assert count() - before == 1
