"""The detection worker's writer survives a failing region-embed stage (#374).

The writer snapshots and clears ``pending`` before the region-embed stage runs, and
calls ``out_q.task_done()`` only once the flush is settled. An exception escaping the
stage used to kill the writer coroutine with those items never marked done: a profile
swap's ``out_q.join()`` then waited forever, and the restarted worker met the same
source photo again. The stage is best-effort (its boxes go without a vector, the
backfill picks them up), so the flush must still write its items and settle them.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

import scripts.curation.worker.region_embed_stage as stage
from curation.test_region_embed_stage import _box, _FakePE, _make_task, _noise_photo
from scripts.curation.worker import flow
from scripts.curation.worker.no_verdict import NoVerdictCounter


if TYPE_CHECKING:
    from pathlib import Path

    from scripts.curation.worker.state import _ItemTask


def _ctx(pe: Any) -> SimpleNamespace:
    return SimpleNamespace(
        args=SimpleNamespace(batch_size=2),
        opensearch=object(),
        started_at=time.monotonic(),
        region_embed_pe=pe,
        out_q=asyncio.Queue(),
        in_flight=set(),
        in_flight_lock=asyncio.Lock(),
        released_at={},
        metrics={'total_processed': 0, 'total_written': 0},
        visible_no_verdict=NoVerdictCounter(3),
        combined_no_verdict=NoVerdictCounter(3),
    )


def _record_writes(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    written: list[list[str]] = []

    async def bulk_update(_opensearch: Any, tasks: list[_ItemTask]) -> tuple[int, int]:
        written.append([t.crop_id for t in tasks])
        return len(tasks), 0

    monkeypatch.setattr(flow, '_bulk_update', bulk_update)
    return written


async def _run_two_flushes(ctx: SimpleNamespace, first: _ItemTask, second: _ItemTask) -> None:
    """One flush on the interval, then one at shutdown, with the writer alive between."""
    writer = asyncio.create_task(flow.writer(ctx))  # type: ignore[arg-type]
    await ctx.out_q.put(first)
    await asyncio.wait_for(ctx.out_q.join(), timeout=5)
    assert not writer.done(), 'the writer died on the failing flush'
    await ctx.out_q.put(second)
    await ctx.out_q.put(None)
    await asyncio.wait_for(ctx.out_q.join(), timeout=5)
    await asyncio.wait_for(writer, timeout=5)


@pytest.mark.asyncio
async def test_a_raising_region_stage_still_writes_and_settles_the_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written = _record_writes(monkeypatch)
    calls = 0

    async def embed(tasks: list[_ItemTask], _pe: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            msg = 'region stage blew up'
            raise RuntimeError(msg)

    monkeypatch.setattr(flow, 'embed_written_regions', embed)
    ctx = _ctx(_FakePE())

    await _run_two_flushes(ctx, _make_task(crop_id='c1'), _make_task(crop_id='c2'))

    assert written == [['c1'], ['c2']]
    assert ctx.metrics['total_written'] == 2


@pytest.mark.asyncio
async def test_a_photo_the_decoder_cannot_read_never_stops_the_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written = _record_writes(monkeypatch)
    bad, good = _noise_photo(tmp_path, 'bad'), _noise_photo(tmp_path, 'good')
    real = stage.decode_source_rgb

    def decode(path: str) -> Any:
        if path == bad:
            msg = 'malformed EXIF'
            raise ValueError(msg)
        return real(path)

    monkeypatch.setattr(stage, 'decode_source_rgb', decode)
    first, second = _make_task(crop_id='c1', boxes=[_box('b1')]), _make_task(crop_id='c2')
    first.image_path, second.image_path = bad, good
    second.pending_boxes = [_box('b2')]
    ctx = _ctx(_FakePE())

    await _run_two_flushes(ctx, first, second)

    assert written == [['c1'], ['c2']]
    assert first.box_vectors == {}
    assert set(second.box_vectors) == {'b2'}
