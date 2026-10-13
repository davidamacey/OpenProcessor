"""The bulk-ingest runner: claim, run, cancel, resume.

Runs in-process as an ``asyncio.Task`` in one ``api`` worker; every fact another
process needs lives on disk (:mod:`.store`), so status and cancel work from
whichever worker a request lands on. The ``scripts/curation/bulk_ingest.py`` CLI
starts and watches runs through the routes, so there is exactly one runner.

The runner is the orchestrator, not the pipeline. Each batch of paths goes to the
API's own ``POST /ingest/batch`` over loopback (:class:`HttpBatchSender`): that
route is the ingest pipeline's one stable seam, so every data-path improvement
behind it reaches bulk ingest unchanged, and the batches spread over all ``api``
worker processes instead of the single event loop this task lives on (one
process tops out near 10 images/s of host CPU; see ``docs/PERFORMANCE.md``).

    producer (walks the source off the loop, skips what the ledger holds)
      -> queue of path batches, ``concurrency`` deep
      -> ``concurrency`` workers, each one batch in flight at a time
      -> ledger append, counters, throttled progress write

The runner holds paths, never image bytes, so its memory does not grow with the
source or the images. Re-running is idempotent: ``ingest_batch`` dedups by content
hash, and the ledger lets a resume skip what this run already stored without
reading the file again.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from src.config.project_context import ensure_marked_dir
from src.core.logging import get_logger
from src.services.curation.bulk_ingest.limits import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    MAX_CONSECUTIVE_FAILED_BATCHES,
    METRICS_SAMPLE_INTERVAL_S,
    PROGRESS_WRITE_INTERVAL_S,
    RATE_WINDOW_S,
)
from src.services.curation.bulk_ingest.sender import BatchRequestError
from src.services.curation.bulk_ingest.source import BulkSource, iter_source_paths
from src.services.curation.bulk_ingest.store import (
    ACTIVE_STATUSES,
    OUTCOME_DUPLICATE,
    OUTCOME_INGESTED,
    RESUMABLE_STATUSES,
    BulkStore,
    active_run,
    list_stores,
    new_run_id,
    now_iso,
    runs_root,
)
from src.services.curation.file_job import heartbeat_ticker
from src.services.curation.job_lock import exclusive_start_lock


if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.services.curation.bulk_ingest.sender import BatchSender


class RunListener(Protocol):
    """Told what each batch settled, for a source that keeps its own record of it."""

    def on_done(self, refs: list[str]) -> None: ...

    def on_failed(self, rows: list[tuple[str, str, str]]) -> None:
        """``(ref, error, error_kind)`` per failed reference."""
        ...


logger = get_logger(__name__)

ERROR_KIND_BATCH = 'batch_error'
_tasks: dict[str, asyncio.Task[None]] = {}


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


class BulkBusyError(Exception):
    def __init__(self, run_id: str | None) -> None:
        super().__init__(f'a bulk ingest is already running: {run_id}')
        self.run_id = run_id


def _source_hash(source: BulkSource) -> str:
    return hashlib.sha256(source.key.encode()).hexdigest()


def claim_run(source: BulkSource, options: BulkOptions) -> tuple[BulkStore, bool]:
    """Atomically claim a run: a new one, or the interrupted/failed/cancelled/partly
    failed run of the same source (``resumed`` is True), which keeps its ledger.
    Raises :class:`BulkBusyError` while any run of the project is live."""
    root = ensure_marked_dir(runs_root())
    key = _source_hash(source)
    with exclusive_start_lock(root / 'start.lock') as acquired:
        if not acquired:
            raise BulkBusyError(None)
        live = active_run()
        if live is not None:
            raise BulkBusyError(live.run_id)
        prior = next(
            (
                s
                for s in list_stores()
                if s.job.read().get('source_key') == key
                and s.job.read().get('status') in RESUMABLE_STATUSES
            ),
            None,
        )
        store = prior or BulkStore(root / new_run_id())
        resumed = prior is not None
        store.write_request({'source': source.to_dict(), 'options': options.to_dict()})
        store.job.clear_signals()
        state = store.job.read()
        store.job.write(
            {
                **state,
                'run_id': store.run_id,
                'source_key': key,
                'source': source.to_dict(),
                'options': options.to_dict(),
                'status': 'queued',
                'error': None,
                'finished_at': None,
                'started_at': state.get('started_at') or now_iso(),
                'attempts': int(state.get('attempts') or 0) + 1,
                'updated_at': now_iso(),
            }
        )
        store.job.touch_heartbeat()
    return store, resumed


def cancel_run(store: BulkStore) -> None:
    """Ask a live run to stop after its in-flight batches. A run that is not live
    keeps its status (the flag is cleared by the next claim)."""
    if store.job.is_live(ACTIVE_STATUSES):
        store.job.request_cancel()


class BulkIngestRun:
    def __init__(
        self,
        *,
        sender: BatchSender,
        store: BulkStore,
        source: BulkSource,
        options: BulkOptions,
        listener: RunListener | None = None,
    ) -> None:
        self.listener = listener
        self.sender = sender
        self.store = store
        self.source = source
        self.options = options
        self.job = store.job
        self.discovered = 0
        self.discovery_complete = False
        self.skipped = 0
        self.ingested = 0
        self.duplicates = 0
        self.failed = 0
        self.crops_indexed = 0
        self.n_embedded = 0
        self.n_not_embedded = 0
        self.n_embed_failed = 0
        self.request_s = 0.0
        self.in_flight = 0
        self._attempt_done = 0
        self._consecutive_failed_batches = 0
        self._abort: str | None = None
        self._dropped = False
        self._started = time.monotonic()
        self._last_flush = 0.0
        self._window: deque[tuple[float, int]] = deque()
        self._stage_base: dict[str, float] = {}
        self._stage_delta: dict[str, float] = {}
        self._last_stage_sample = 0.0

    # ------------------------------------------------------------------ control

    def _stopping(self) -> bool:
        return self._abort is not None or self.job.cancel_requested()

    # ----------------------------------------------------------------- progress

    @property
    def processed(self) -> int:
        return self.skipped + self._attempt_done + self.failed

    def _rates(self, now: float) -> tuple[float, float | None]:
        elapsed = max(now - self._started, 1e-6)
        overall = (self._attempt_done + self.failed) / elapsed
        while self._window and now - self._window[0][0] > RATE_WINDOW_S:
            self._window.popleft()
        recent: float | None = None
        if len(self._window) >= 2:
            span = self._window[-1][0] - self._window[0][0]
            if span > 0:
                recent = (self._window[-1][1] - self._window[0][1]) / span
        return overall, recent

    def _snapshot(self, now: float) -> dict[str, Any]:
        overall, recent = self._rates(now)
        rate = recent or overall
        remaining = self.discovered - self.processed
        stage_s = {**self._stage_delta, 'batch_requests': round(self.request_s, 3)}
        return {
            'discovered': self.discovered,
            'discovery_complete': self.discovery_complete,
            'skipped': self.skipped,
            'ingested': self.ingested,
            'duplicates': self.duplicates,
            'failed': self.failed,
            'processed': self.processed,
            'crops_indexed': self.crops_indexed,
            'n_embedded': self.n_embedded,
            'n_not_embedded': self.n_not_embedded,
            'n_embed_failed': self.n_embed_failed,
            'in_flight': self.in_flight,
            'images_per_s': round(overall, 3),
            'images_per_s_recent': round(recent, 3) if recent is not None else None,
            'eta_s': round(remaining / rate, 1)
            if self.discovery_complete and rate and remaining > 0
            else None,
            'elapsed_s': round(now - self._started, 1),
            'stage_s': stage_s,
            'updated_at': now_iso(),
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
        self._window.append((now, self._attempt_done + self.failed))
        await self._sample_stages(force=force)
        self.job.update(**self._snapshot(now))

    # ----------------------------------------------------------------- producer

    async def _produce(self, queue: asyncio.Queue[list[str] | None], done: set[str]) -> None:
        iterator: Iterator[str] = iter_source_paths(self.source, should_stop=self._stopping)
        batch_size = self.options.batch_size

        def take() -> tuple[list[str], int, int, bool]:
            chunk: list[str] = []
            seen = skipped = 0
            for path in iterator:
                seen += 1
                if path in done:
                    skipped += 1
                    continue
                chunk.append(path)
                if len(chunk) >= batch_size:
                    return chunk, seen, skipped, False
            return chunk, seen, skipped, True

        try:
            while not self._stopping():
                chunk, seen, skipped, exhausted = await asyncio.to_thread(take)
                self.discovered += seen
                self.skipped += skipped
                if chunk:
                    await queue.put(chunk)
                if exhausted:
                    self.discovery_complete = True
                    break
                await self._flush()
        finally:
            close = getattr(iterator, 'close', None)
            if close is not None:
                await asyncio.to_thread(close)

    # ------------------------------------------------------------------ workers

    async def _work(self, queue: asyncio.Queue[list[str] | None]) -> None:
        while True:
            paths = await queue.get()
            if paths is None:
                return
            if self._stopping():
                self._dropped = True
                continue
            self.in_flight += len(paths)
            try:
                await self._process(paths)
            except Exception as exc:  # one bad batch never stops the run
                logger.error('bulk_ingest_batch_crashed', error=str(exc), n=len(paths))
                await self._record_failures(paths, str(exc), ERROR_KIND_BATCH)
            finally:
                self.in_flight -= len(paths)
            await self._flush()

    async def _record_failures(self, paths: list[str], error: str, error_kind: str) -> None:
        self.failed += len(paths)
        self.store.append_failures(
            [{'path': p, 'error': error[:300], 'error_kind': error_kind} for p in paths]
        )
        if self.listener is not None:
            await asyncio.to_thread(
                self.listener.on_failed, [(p, error[:300], error_kind) for p in paths]
            )

    async def _process(self, paths: list[str]) -> None:
        started = time.perf_counter()
        try:
            body = await self.sender.post_batch(
                paths, source=self.options.source_tag, run_id=self.store.run_id
            )
        except BatchRequestError as exc:
            logger.error('bulk_ingest_batch_failed', error=str(exc), n=len(paths))
            await self._record_failures(paths, str(exc), ERROR_KIND_BATCH)
            self._note_batch(all_failed=True, reason=str(exc))
            return
        finally:
            self.request_s += time.perf_counter() - started

        done_rows: list[tuple[str, str]] = []
        unaccounted = set(paths)
        for res in body.get('results', []):
            path = res.get('image_path')
            if path not in unaccounted:
                continue
            unaccounted.discard(path)
            if res.get('status') == 'success':
                done_rows.append((path, OUTCOME_INGESTED))
            elif res.get('status') == 'duplicate':
                done_rows.append((path, OUTCOME_DUPLICATE))
            else:
                await self._record_failures(
                    [path], res.get('error') or 'ingest failed', res.get('error_kind') or ''
                )
        if unaccounted:  # a result the route never returned is a failure, not a success
            await self._record_failures(sorted(unaccounted), 'no result returned', ERROR_KIND_BATCH)

        summary = body.get('summary', {})
        self.ingested += sum(1 for _, o in done_rows if o == OUTCOME_INGESTED)
        self.duplicates += sum(1 for _, o in done_rows if o == OUTCOME_DUPLICATE)
        self._attempt_done += len(done_rows)
        self.crops_indexed += summary.get('crops_indexed', 0)
        self.n_embedded += summary.get('n_embedded', 0)
        self.n_not_embedded += summary.get('n_not_embedded', 0)
        self.n_embed_failed += summary.get('n_embed_failed', 0)
        await asyncio.to_thread(self.store.append_done, done_rows)
        if self.listener is not None and done_rows:
            await asyncio.to_thread(self.listener.on_done, [p for p, _ in done_rows])
        first_error = next(
            (r.get('error') for r in body.get('results', []) if r.get('error')), None
        )
        self._note_batch(all_failed=not done_rows, reason=first_error)

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

    async def run(self) -> None:
        done, self.ingested, self.duplicates = await asyncio.to_thread(self.store.done)
        # Crop and vector totals continue from the earlier attempts of a resumed run.
        prior = self.job.read()
        self.crops_indexed = int(prior.get('crops_indexed') or 0)
        self.n_embedded = int(prior.get('n_embedded') or 0)
        self.n_not_embedded = int(prior.get('n_not_embedded') or 0)
        self.n_embed_failed = int(prior.get('n_embed_failed') or 0)
        await asyncio.to_thread(self.store.reset_failures)
        await self._sample_stages(force=True)
        self.job.update(status='running', **self._snapshot(time.monotonic()))
        self.job.touch_heartbeat()
        ticker = asyncio.create_task(heartbeat_ticker(self.job))
        queue: asyncio.Queue[list[str] | None] = asyncio.Queue(maxsize=self.options.concurrency)
        workers = [asyncio.create_task(self._work(queue)) for _ in range(self.options.concurrency)]
        try:
            try:
                await self._produce(queue, done)
            except Exception as exc:
                logger.error('bulk_ingest_source_failed', error=str(exc))
                self._abort = f'could not read the source: {exc}'[:300]
            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers)
        finally:
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker
            for worker in workers:
                worker.cancel()
        await self._flush(force=True)
        if self._abort:
            status = 'failed'
        elif self.job.cancel_requested() and (self._dropped or not self.discovery_complete):
            status = 'cancelled'
        elif self.failed:
            status = 'completed_with_errors'
        else:
            status = 'completed'
        self.job.update(status=status, error=self._abort, finished_at=now_iso())


def spawn(
    *,
    sender: BatchSender,
    store: BulkStore,
    source: BulkSource,
    options: BulkOptions,
    listener: RunListener | None = None,
) -> None:
    run = BulkIngestRun(
        sender=sender, store=store, source=source, options=options, listener=listener
    )

    async def runner() -> None:
        try:
            await run.run()
        except Exception as exc:
            logger.error('bulk_ingest_failed', run_id=store.run_id, error=str(exc))
            store.job.update(status='failed', error=str(exc)[:300], finished_at=now_iso())
        finally:
            aclose = getattr(sender, 'aclose', None)
            if aclose is not None:
                await aclose()

    task = asyncio.create_task(runner())
    _tasks[store.run_id] = task
    task.add_done_callback(lambda _t: _tasks.pop(store.run_id, None))
