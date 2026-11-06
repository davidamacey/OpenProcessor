"""Progress of an auto-label run: the stage names, the stages' elapsed tick, and the bridge to the job.

The stages report through a synchronous reporter (``start_stage``, ``advance``, ``update``). The
bridge keeps that state in memory and a sibling task flushes it to ``ctx.progress`` once a second
(the context throttles the database write itself), so no stage awaits a database round trip.
Cancel and pause are read from the context at a stage boundary or a VLM chunk, never mid-write.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

from src.services.curation.clustering.pool_coverage import PARTIAL


if TYPE_CHECKING:
    from collections.abc import Coroutine

    from src.tasks.runtime.context import JobContext


STAGES: tuple[str, ...] = (
    'embed_missing',
    'cluster_id_normalize',
    'cluster_residuals',
    'auto_promote',
    'vlm',
    'finalize',
)
FLUSH_INTERVAL_S = 1.0


class PauseRequestedError(Exception):
    """Raised at a stage boundary or VLM chunk once a pause was asked for."""


def failed_stages(result: Any) -> list[str]:
    """Names of the stages a finished run reported as ``status: error`` or ``partial``
    (a cluster pass that did not cover its pool); such a run must not end as a success."""
    stages = result.get('stages') if isinstance(result, dict) else None
    if not isinstance(stages, dict):
        return []
    return [
        name
        for name, s in stages.items()
        if isinstance(s, dict) and s.get('status') in ('error', PARTIAL)
    ]


class JobProgress:
    """The reporter the stages call, flushed to the job's progress snapshot."""

    def __init__(self, ctx: JobContext) -> None:
        self._ctx = ctx
        self.stage = ''
        self.processed = 0
        self.total = 0
        self.backend: str | None = None
        self.backend_detail: str | None = None
        self.free_vram_mb: int | None = None
        self.peak_vram_mb: int | None = None
        self.stage_durations: dict[str, float] = {}
        self._stage_started_at = 0.0

    def _fold_stage_duration(self) -> None:
        if self.stage and self._stage_started_at > 0:
            spent = max(0.0, time.time() - self._stage_started_at)
            self.stage_durations[self.stage] = round(
                self.stage_durations.get(self.stage, 0.0) + spent, 2
            )

    def start_stage(self, stage: str, total: int = 0) -> None:
        if stage not in STAGES:
            raise ValueError(f'unknown stage: {stage}')
        self._fold_stage_duration()
        self.stage, self.processed, self.total = stage, 0, int(total)
        self._stage_started_at = time.time()

    def advance(self, n: int = 1) -> None:
        self.processed = min(self.processed + int(n), self.total or self.processed + int(n))

    def update(self, processed: int, total: int | None = None) -> None:
        self.processed = max(0, int(processed))
        if total is not None:
            self.total = max(0, int(total))

    def set_backend(self, name: str, detail: str, free_vram_mb: int | None = None) -> None:
        self.backend, self.backend_detail, self.free_vram_mb = name, detail, free_vram_mb

    def record_peak_vram(self, mb: int | None) -> None:
        if mb is not None and mb > (self.peak_vram_mb or 0):
            self.peak_vram_mb = int(mb)

    def finalize(self) -> None:
        self._fold_stage_duration()
        self._stage_started_at = 0.0

    @property
    def cancelled(self) -> bool:
        return self._ctx.cancelled()

    def raise_if_cancelled(self) -> None:
        if self._ctx.cancelled():
            raise asyncio.CancelledError('auto_label run cancelled')

    @property
    def paused(self) -> bool:
        return self._ctx.paused()

    def raise_if_paused(self) -> None:
        if self._ctx.paused():
            raise PauseRequestedError

    async def flush(self) -> None:
        await self._ctx.progress(
            self.processed,
            self.total,
            self.stage or None,
            {
                'backend': self.backend,
                'backend_detail': self.backend_detail,
                'free_vram_mb': self.free_vram_mb,
                'peak_vram_mb': self.peak_vram_mb,
                'stage_durations': dict(self.stage_durations),
            },
        )

    async def flush_loop(self, stop: asyncio.Event) -> None:
        """Flush once a second until ``stop`` is set."""
        while not stop.is_set():
            await self.flush()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=FLUSH_INTERVAL_S)


async def with_elapsed_tick[T](
    progress: JobProgress | None, coro: Coroutine[Any, Any, T], *, interval: float = 5.0
) -> T:
    """Run ``coro`` while a sibling task ticks ``progress.update(elapsed_s)``, so a stage whose
    operation offers no callback shows a live counter instead of ``0/0``."""
    if progress is None:
        return await coro
    started = time.time()

    async def _tick() -> None:
        while True:
            progress.update(processed=int(time.time() - started), total=0)
            await asyncio.sleep(interval)

    task = asyncio.create_task(coro)
    ticker = asyncio.create_task(_tick())
    try:
        return await task
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker
