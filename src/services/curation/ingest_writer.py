"""Coalesced ingest writes: many images' documents per OpenSearch bulk.

Every image used to cost three requests of its own (an images-index bulk, an ``mget`` for
the OCC pre-read and an items-index bulk), a few dozen documents each. This module
collects the documents of images that finish near each other in one API process, and
writes them as one images bulk, one ``mget`` and one items bulk of up to
:data:`MAX_BULK_DOCS` documents, with at most :data:`MAX_WRITERS` such flushes in flight.
Probe 5 in ``docs/PERFORMANCE.md`` measured the datastore side: one writer sustains about
1,000 documents/s, four about 3,200 and sixteen about 7,000.

The scheduling is group commit: a flush starts as soon as a writer slot is free, and waits
at most :data:`LINGER_S` for a thin batch to fill, so an idle process adds almost no
latency and a busy one batches by itself. A caller still gets an answer only after its own
documents were acknowledged, and it gets *its own* answer: one refused document fails
that image's result and nothing else in the flush.

Safety rules kept from the per-image path:

- Items go through :func:`~src.clients.occ_upsert.occ_upsert_bulk` (create, else
  read-merge-write), so the lock rule and human-label preservation are unchanged.
- Units are grouped by project index names and each group is flushed in the context its
  caller was in, so the project guard sees the bound project of the documents it writes.
- Two images that share an item id (the same image ingested twice at once) never share a
  flush, so the second sees the first as already created.
- An images doc is written before its items. A document the store refuses is reported in
  its own image's result (``refused``) and does not touch the others, as before.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
import weakref
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from src.clients.occ_upsert import DocOutcome, occ_upsert_bulk
from src.core.logging import get_logger


logger = get_logger(__name__)

MAX_BULK_DOCS = 1000
MAX_WRITERS = 4
LINGER_S = 0.02
# HTTP 429 (write queue full) is retried with a growing pause; anything else is final.
RETRY_429_PAUSES_S = (0.1, 0.25, 0.5, 1.0, 2.0)


@dataclass
class UnitResult:
    """What one image's documents did; ``refused`` lists the reasons for any the store refused."""

    images_indexed: int = 0
    created: int = 0
    updated: int = 0
    preserved_human: int = 0
    final_conflicts: int = 0
    filled_absent: int = 0
    created_ids: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)


@dataclass
class _Unit:
    image_doc: dict[str, Any] | None
    crop_docs: list[dict[str, Any]]
    images_index: str
    items_index: str
    guards: tuple[str, ...]
    fill_if_absent: tuple[str, ...]
    context: contextvars.Context
    future: asyncio.Future[UnitResult]
    queued_at: float = field(default_factory=time.monotonic)

    @property
    def n_docs(self) -> int:
        return len(self.crop_docs) + (1 if self.image_doc else 0)

    @cached_property
    def ids(self) -> set[str]:
        return {d['crop_id'] for d in self.crop_docs}

    @property
    def group(self) -> tuple[Any, ...]:
        return (self.images_index, self.items_index, self.guards, self.fill_if_absent)


def _is_429(exc: BaseException) -> bool:
    return getattr(exc, 'status_code', None) == 429


class IngestWriter:
    """The per-process queue in front of one OpenSearch client."""

    def __init__(
        self,
        client: Any,
        *,
        max_docs: int = MAX_BULK_DOCS,
        writers: int = MAX_WRITERS,
        linger_s: float = LINGER_S,
    ) -> None:
        self._client = client
        self.loop = asyncio.get_running_loop()
        self._max_docs = max_docs
        self._linger_s = linger_s
        self._slots = asyncio.Semaphore(writers)
        self._pending: list[_Unit] = []
        self._wake = asyncio.Event()
        self._drainer: asyncio.Task[None] | None = None
        self._flushes: set[asyncio.Task[None]] = set()
        self.bulk_sizes: list[int] = []  # docs per flush, for the tests and probes

    async def write(
        self,
        image_doc: dict[str, Any] | None,
        crop_docs: list[dict[str, Any]],
        *,
        images_index: str,
        items_index: str,
        guards: tuple[str, ...],
        fill_if_absent: tuple[str, ...] = (),
    ) -> UnitResult:
        """Queue one image's documents and wait for their own result."""
        unit = _Unit(
            image_doc,
            crop_docs,
            images_index,
            items_index,
            guards,
            fill_if_absent,
            contextvars.copy_context(),
            asyncio.get_running_loop().create_future(),
        )
        self._pending.append(unit)
        if self._drainer is None or self._drainer.done():
            self._drainer = asyncio.get_running_loop().create_task(self._drain())
        self._wake.set()
        return await unit.future

    async def _drain(self) -> None:
        try:
            await self._drain_loop()
        except Exception as exc:
            logger.error('ingest_writer_drain_failed', error=str(exc))
            for unit in self._pending:
                if not unit.future.done():
                    unit.future.set_exception(exc)
            self._pending = []

    async def _drain_loop(self) -> None:
        while True:
            self._pending = [u for u in self._pending if not u.future.done()]
            if not self._pending:
                self._wake.clear()
                await self._wake.wait()
                continue
            await self._slots.acquire()
            try:
                await self._fill_window()
                batch = self._take()
            except BaseException:
                self._slots.release()
                raise
            if not batch:
                self._slots.release()
                continue
            task = asyncio.get_running_loop().create_task(self._flush(batch))
            self._flushes.add(task)
            task.add_done_callback(self._flush_done)

    async def close(self) -> None:
        """Stop the queue once its writes are done (a bench or test ending its loop)."""
        while self._pending or self._flushes:
            await asyncio.sleep(0.005)
        if self._drainer is not None:
            self._drainer.cancel()
            await asyncio.gather(self._drainer, return_exceptions=True)

    def _flush_done(self, task: asyncio.Task[None]) -> None:
        self._flushes.discard(task)
        self._slots.release()

    async def _fill_window(self) -> None:
        """Wait, up to the linger, for a thin batch to fill."""
        while sum(u.n_docs for u in self._pending) < self._max_docs:
            left = (
                self._pending[0].queued_at + self._linger_s - time.monotonic()
                if self._pending
                else 0
            )
            if left <= 0:
                return
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), left)
            except TimeoutError:
                return

    def _take(self) -> list[_Unit]:
        batch: list[_Unit] = []
        rest: list[_Unit] = []
        ids: set[str] = set()
        docs = 0
        for unit in self._pending:
            fits = not batch or docs + unit.n_docs <= self._max_docs
            if unit.future.done():
                continue
            if fits and ids.isdisjoint(unit.ids):
                batch.append(unit)
                ids |= unit.ids
                docs += unit.n_docs
            else:
                rest.append(unit)
        self._pending = rest
        return batch

    async def _flush(self, batch: list[_Unit]) -> None:
        self.bulk_sizes.append(sum(u.n_docs for u in batch))
        groups: dict[tuple[Any, ...], list[_Unit]] = {}
        for unit in batch:
            groups.setdefault(unit.group, []).append(unit)
        await asyncio.gather(*(self._flush_group(units) for units in groups.values()))

    async def _flush_group(self, units: list[_Unit]) -> None:
        # The caller's context carries the bound project the guard checks.
        task = asyncio.get_running_loop().create_task(
            self._write_group(units), context=units[0].context.copy()
        )
        try:
            await task
        except BaseException as exc:  # a failed flush must still answer every caller
            if not isinstance(exc, Exception):
                raise
            logger.error('ingest_write_flush_failed', error=str(exc), n_images=len(units))
            for unit in units:
                if not unit.future.done():
                    unit.future.set_exception(exc)

    async def _write_group(self, units: list[_Unit]) -> None:
        results = {id(u): UnitResult() for u in units}
        await self._write_images(units, results)
        await self._write_items(units, results)
        for unit in units:
            if not unit.future.done():
                unit.future.set_result(results[id(unit)])

    async def _write_images(self, units: list[_Unit], results: dict[int, UnitResult]) -> None:
        todo = [u for u in units if u.image_doc]
        for pause in (*RETRY_429_PAUSES_S, None):
            if not todo:
                return
            body: list[dict[str, Any]] = []
            for u in todo:
                assert u.image_doc is not None
                body += [{'index': {'_index': u.images_index, '_id': u.image_doc['image_id']}}]
                body.append(u.image_doc)
            try:
                resp = await self._client.bulk(body=body, refresh=False)
            except Exception as exc:
                if _is_429(exc) and pause is not None:
                    await asyncio.sleep(pause)
                    continue
                raise
            if not resp.get('errors'):
                for unit in todo:
                    results[id(unit)].images_indexed = 1
                return
            retry: list[_Unit] = []
            for unit, item in zip(todo, resp.get('items', []), strict=False):
                op = item.get('index') or {}
                status = op.get('status', 500)
                if status < 300:
                    results[id(unit)].images_indexed = 1
                elif status == 429 and pause is not None:
                    retry.append(unit)
                else:
                    results[id(unit)].refused.append(f'images doc: {_reason(op)}')
            if not retry:
                return
            todo = retry
            await asyncio.sleep(pause or 0)

    async def _write_items(self, units: list[_Unit], results: dict[int, UnitResult]) -> None:
        remaining = [d for u in units for d in u.crop_docs]
        if not remaining:
            return
        first = units[0]
        outcomes: dict[str, DocOutcome] = {}
        for pause in (*RETRY_429_PAUSES_S, None):
            try:
                await occ_upsert_bulk(
                    self._client,
                    remaining,
                    index=first.items_index,
                    human_field_guards=list(first.guards),
                    writer_id='ingest',
                    fill_if_absent=first.fill_if_absent,
                    outcomes=outcomes,
                )
            except Exception as exc:
                if not (_is_429(exc) and pause is not None):
                    raise
                await asyncio.sleep(pause)
                continue
            refused = [d for d in remaining if _refused_429(outcomes.get(d['crop_id']))]
            if not refused or pause is None:
                break
            for doc in refused:
                outcomes.pop(doc['crop_id'])
            remaining = refused
            await asyncio.sleep(pause)
        for unit in units:
            self._settle(unit, results[id(unit)], outcomes)

    @staticmethod
    def _settle(unit: _Unit, result: UnitResult, outcomes: dict[str, DocOutcome]) -> None:
        for doc in unit.crop_docs:
            out = outcomes.get(doc['crop_id'])
            if out is None or out.status == 'error':
                result.refused.append(f'item {doc["crop_id"]}: {out and out.error}')
            elif out.status == 'created':
                result.created += 1
                result.created_ids.append(doc['crop_id'])
            elif out.status == 'updated':
                result.updated += 1
                result.preserved_human += out.preserved
                result.filled_absent += 1 if out.filled else 0
            elif out.status == 'conflict':
                result.final_conflicts += 1


def _refused_429(outcome: DocOutcome | None) -> bool:
    return (
        outcome is not None
        and outcome.status == 'error'
        and (outcome.error or '').startswith('429')
    )


def _reason(op: dict[str, Any]) -> str:
    error = op.get('error')
    if isinstance(error, dict):
        return f'{op.get("status")}: {error.get("type")}: {error.get("reason")}'
    return f'{op.get("status")}: {error}'


_WRITERS: weakref.WeakKeyDictionary[Any, IngestWriter] = weakref.WeakKeyDictionary()


def get_ingest_writer(client: Any) -> IngestWriter:
    """The writer of ``client`` in the running event loop (one per client and loop)."""
    writer = _WRITERS.get(client)
    loop = asyncio.get_running_loop()
    if writer is None or writer.loop is not loop:
        writer = IngestWriter(client)
        _WRITERS[client] = writer
    return writer
