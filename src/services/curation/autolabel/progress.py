"""Progress of an auto-label run: the stage names and the bridge to the job.

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


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext


STAGES: tuple[str, ...] = (
    'embed_missing',
    'vlm',
    'finalize',
)
FLUSH_INTERVAL_S = 1.0


class PauseRequestedError(Exception):
    """Raised at a stage boundary or VLM chunk once a pause was asked for."""


def failed_stages(result: Any) -> list[str]:
    """Names of the stages a finished run reported as ``status: error``; such a run must not
    end as a success."""
    stages = result.get('stages') if isinstance(result, dict) else None
    if not isinstance(stages, dict):
        return []
    return [
        name for name, s in stages.items() if isinstance(s, dict) and s.get('status') == 'error'
    ]


class JobProgress:
    """The reporter the stages call, flushed to the job's progress snapshot."""

    def __init__(self, ctx: JobContext) -> None:
        self._ctx = ctx
        self.stage = ''
        self.processed = 0
        self.total = 0
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
                'stage_durations': dict(self.stage_durations),
            },
        )

    async def flush_loop(self, stop: asyncio.Event) -> None:
        """Flush once a second until ``stop`` is set."""
        while not stop.is_set():
            await self.flush()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=FLUSH_INTERVAL_S)
