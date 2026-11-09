"""The bulk-ingest runner: claim files, send them through the ingest route, settle the rows.

It runs inside the ``bulk_ingest_run`` job (``src/tasks/jobs_c.py``). The source's queue
(``ingest_source_files``) is the ledger: a file is ``claimed`` when a batch of it is in flight,
``done`` once the pipeline stored it, ``failed`` when it did not. A resume claims what is left,
so nothing stored is read again.

The runner is the orchestrator, not the pipeline. Each batch of paths goes to the API's own
``POST /ingest/batch`` over loopback (:class:`HttpBatchSender`): that route is the ingest
pipeline's one stable seam, so every data-path improvement behind it reaches bulk ingest
unchanged, and the batches spread over all ``api`` worker processes instead of the one process
this job runs in.

    claim 500 files (SKIP LOCKED) -> batches of ``batch_size`` paths, ``concurrency`` deep
      -> ``concurrency`` workers, each one batch in flight at a time
      -> rows settled, counters, progress and checkpoint

The runner holds paths, never image bytes, so its memory does not grow with the source or the
images. Re-running is idempotent: ``ingest_batch`` dedups by content hash.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from src.core.logging import get_logger
from src.domain.job_perf import RESULT_KEY, JobPerfSummary
from src.services.curation.bulk_ingest.limits import (
    CLAIM_BATCH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    MAX_CONSECUTIVE_FAILED_BATCHES,
    METRICS_SAMPLE_INTERVAL_S,
    PROGRESS_WRITE_INTERVAL_S,
    RATE_WINDOW_S,
)
from src.services.curation.bulk_ingest.sender import BatchRequestError


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.services.curation.bulk_ingest.sender import BatchSender


logger = get_logger(__name__)

ERROR_KIND_BATCH = 'batch_error'
COUNTER_KEYS = (
    'ingested',
    'duplicates',
    'crops_indexed',
    'n_embedded',
    'n_not_embedded',
    'n_embed_failed',
)


@dataclass(frozen=True)
class BulkOptions:
    source_tag: str = 'bulk_ingest'
    batch_size: int = DEFAULT_BATCH_SIZE
    concurrency: int = DEFAULT_CONCURRENCY

    def to_dict(self) -> dict[str, Any]:
        return {
            'source_tag': self.source_tag,
            'batch_size': self.batch_size,
            'concurrency': self.concurrency,
        }


class BulkRunAbortedError(RuntimeError):
    """The run gave up: the source could not be read, or batch after batch failed entirely."""


@dataclass(frozen=True)
class ClaimedPath:
    file_key: str
    path: str


class FileLedger(Protocol):
    """The queue rows of one source, as the runner needs them."""

    async def remaining(self) -> int:
        """Files a run would still send: pending, plus failed ones with attempts left."""
        ...

    async def claim(self, limit: int) -> list[ClaimedPath]: ...

    async def settle(
        self,
        done: Sequence[tuple[str, str | None]],
        failed: Sequence[tuple[str, str, str]],
    ) -> None:
        """``(file_key, image_id)`` rows done; ``(file_key, error, error_kind)`` rows failed."""
        ...

    async def release(self) -> None:
        """Hand back the files this run claimed and did not settle."""
        ...


class Reporter(Protocol):
    """What the runner needs of its job: the stop signal, progress and a resume cursor."""

    def cancelled(self) -> bool: ...

    async def progress(
        self,
        done: int,
        total: int | None = None,
        stage: str | None = None,
        counters: Mapping[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> None: ...

    async def checkpoint(self, cursor: Mapping[str, Any] | None) -> None: ...


class BulkIngestRun:
    def __init__(
        self,
        *,
        sender: BatchSender,
        ledger: FileLedger,
        reporter: Reporter,
        options: BulkOptions,
        run_id: str,
        prior: Mapping[str, Any] | None = None,
        servers: Sequence[str] = (),
    ) -> None:
        self.sender = sender
        self.ledger = ledger
        self.reporter = reporter
        self.options = options
        self.run_id = run_id
        self.servers = list(servers)
        self.discovered = 0
        self.discovery_complete = False
        self.failed = 0
        self.request_s = 0.0
        self.in_flight = 0
        # Totals continue from the earlier attempts of a resumed run (the checkpoint).
        self.totals = {key: int((prior or {}).get(key) or 0) for key in COUNTER_KEYS}
        self._attempt_done = 0
        self._consecutive_failed_batches = 0
        self._abort: str | None = None
        self._started = time.monotonic()
        self._last_flush = 0.0
        self._window: deque[tuple[float, int]] = deque()
        self._stage_base: dict[str, float] = {}
        self._stage_delta: dict[str, float] = {}
        self._last_stage_sample = 0.0

    # ------------------------------------------------------------------ control

    def _stopping(self) -> bool:
        return self._abort is not None or self.reporter.cancelled()

    # ----------------------------------------------------------------- progress

    @property
    def processed(self) -> int:
        return self._attempt_done + self.failed

    def _rates(self, now: float) -> tuple[float, float | None]:
        elapsed = max(now - self._started, 1e-6)
        overall = self.processed / elapsed
        while self._window and now - self._window[0][0] > RATE_WINDOW_S:
            self._window.popleft()
        recent: float | None = None
        if len(self._window) >= 2:
            span = self._window[-1][0] - self._window[0][0]
            if span > 0:
                recent = (self._window[-1][1] - self._window[0][1]) / span
        return overall, recent

    def counters(self, now: float) -> dict[str, Any]:
        overall, recent = self._rates(now)
        rate = recent or overall
        remaining = self.discovered - self.processed
        return {
            'discovered': self.discovered,
            'discovery_complete': self.discovery_complete,
            'failed': self.failed,
            **self.totals,
            'in_flight': self.in_flight,
            'images_per_s': round(overall, 3),
            'images_per_s_recent': round(recent, 3) if recent is not None else None,
            'eta_s': round(remaining / rate, 1)
            if self.discovery_complete and rate and remaining > 0
            else None,
            'elapsed_s': round(now - self._started, 1),
            'stage_s': {**self._stage_delta, 'batch_requests': round(self.request_s, 3)},
        }

    async def _sample_stages(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_stage_sample < METRICS_SAMPLE_INTERVAL_S:
            return
        self._last_stage_sample = now
        totals = await self.sender.stage_seconds()
        if not self._stage_base:
            self._stage_base = totals
            return
        self._stage_delta = {
            stage: round(total - self._stage_base.get(stage, 0.0), 3)
            for stage, total in totals.items()
            if total > self._stage_base.get(stage, 0.0)
        }

    async def _flush(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_flush < PROGRESS_WRITE_INTERVAL_S:
            return
        self._last_flush = now
        self._window.append((now, self.processed))
        await self._sample_stages(force=force)
        await self.reporter.progress(
            self.processed, self.discovered, 'ingest', self.counters(now), force=force
        )
        await self.reporter.checkpoint(dict(self.totals))

    # ----------------------------------------------------------------- producer

    async def _produce(self, queue: asyncio.Queue[list[ClaimedPath] | None]) -> None:
        size = self.options.batch_size
        while not self._stopping():
            claimed = await self.ledger.claim(CLAIM_BATCH)
            if not claimed:
                self.discovery_complete = True
                return
            for start in range(0, len(claimed), size):
                await queue.put(claimed[start : start + size])
            await self._flush()

    # ------------------------------------------------------------------ workers

    async def _work(self, queue: asyncio.Queue[list[ClaimedPath] | None]) -> None:
        while True:
            batch = await queue.get()
            if batch is None:
                return
            if self._stopping():
                continue  # the producer's release hands these back
            self.in_flight += len(batch)
            try:
                await self._process(batch)
            except Exception as exc:  # one bad batch never stops the run
                logger.error('bulk_ingest_batch_crashed', error=str(exc), n=len(batch))
                await self._record_failures(batch, str(exc), ERROR_KIND_BATCH)
            finally:
                self.in_flight -= len(batch)
            await self._flush()

    async def _record_failures(
        self, batch: Sequence[ClaimedPath], error: str, error_kind: str
    ) -> None:
        self.failed += len(batch)
        await self.ledger.settle([], [(c.file_key, error[:300], error_kind) for c in batch])

    async def _process(self, batch: list[ClaimedPath]) -> None:
        started = time.perf_counter()
        by_path: dict[str, list[ClaimedPath]] = {}
        for claimed in batch:
            by_path.setdefault(claimed.path, []).append(claimed)
        try:
            body = await self.sender.post_batch(
                list(by_path), source=self.options.source_tag, run_id=self.run_id
            )
        except BatchRequestError as exc:
            logger.error('bulk_ingest_batch_failed', error=str(exc), n=len(batch))
            await self._record_failures(batch, str(exc), ERROR_KIND_BATCH)
            self._note_batch(all_failed=True, reason=str(exc))
            return
        finally:
            self.request_s += time.perf_counter() - started

        done: list[tuple[str, str | None]] = []
        failed: list[tuple[str, str, str]] = []
        ingested = duplicates = 0
        for res in body.get('results', []):
            claimed_rows = by_path.pop(res.get('image_path'), [])
            if res.get('status') in ('success', 'duplicate'):
                ingested += res.get('status') == 'success'
                duplicates += res.get('status') == 'duplicate'
                done.extend((c.file_key, res.get('image_id')) for c in claimed_rows)
            else:
                failed.extend(
                    (c.file_key, res.get('error') or 'ingest failed', res.get('error_kind') or '')
                    for c in claimed_rows
                )
        # a result the route never returned is a failure, not a success
        failed.extend(
            (c.file_key, 'no result returned', ERROR_KIND_BATCH)
            for rows in by_path.values()
            for c in rows
        )
        await self.ledger.settle(done, failed)
        self.failed += len(failed)
        summary = body.get('summary', {})
        self.totals['ingested'] += ingested
        self.totals['duplicates'] += duplicates
        self._attempt_done += len(done)
        for key in ('crops_indexed', 'n_embedded', 'n_not_embedded', 'n_embed_failed'):
            self.totals[key] += summary.get(key, 0)
        first_error = next(
            (r.get('error') for r in body.get('results', []) if r.get('error')), None
        )
        self._note_batch(all_failed=not done, reason=first_error)

    def _note_batch(self, *, all_failed: bool, reason: str | None) -> None:
        if not all_failed:
            self._consecutive_failed_batches = 0
            return
        self._consecutive_failed_batches += 1
        if self._consecutive_failed_batches >= MAX_CONSECUTIVE_FAILED_BATCHES:
            self._abort = (
                f'{self._consecutive_failed_batches} batches in a row failed entirely'
                f' (last error: {reason})'
            )[:300]

    # ---------------------------------------------------------------------- run

    async def run(self) -> dict[str, Any]:
        """Drain the queue and return the job's result. Raises :class:`BulkRunAbortedError` when
        the run gave up; a stop request ends it early with the counts so far."""
        self.discovered = await self.ledger.remaining()
        await self._sample_stages(force=True)
        await self.reporter.progress(0, self.discovered, 'ingest', self.counters(time.monotonic()))
        queue: asyncio.Queue[list[ClaimedPath] | None] = asyncio.Queue(
            maxsize=self.options.concurrency
        )
        try:
            async with asyncio.TaskGroup() as group:
                for _ in range(self.options.concurrency):
                    group.create_task(self._work(queue))
                await self._produce(queue)
                for _ in range(self.options.concurrency):
                    await queue.put(None)
        finally:
            await self.ledger.release()
        await self._flush(force=True)
        if self._abort:
            raise BulkRunAbortedError(self._abort)
        return self._result()

    def _result(self) -> dict[str, Any]:
        now = time.monotonic()
        wall = round(now - self._started, 3)
        images = self.totals['ingested'] + self.totals['duplicates']
        return {
            **self.counters(now),
            'processed': self.processed,
            'cancelled': self.reporter.cancelled(),
            RESULT_KEY: JobPerfSummary(
                images=images,
                wall_s=wall,
                images_per_s=round(images / wall, 3) if wall else 0.0,
                decode_routes={},
                stage_seconds={k: float(v) for k, v in self._stage_delta.items()},
                pe_calls_per_image=0.0,
                servers=self.servers,
            ).model_dump(),
        }
