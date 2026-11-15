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
- Two writes that share an item id (labels added to one image twice at once) never share a
  flush, so the second sees the first as already created.
- An images doc is written before its items, with ``create``: its id names the content
  (:func:`~src.services.near_duplicates.content_key.content_image_id`), so a second copy of the
  same content that got past the content lookup (the lookup cannot see an unrefreshed
  write) is answered 409. That unit is the ``duplicate`` and none of its items are written
  (#358). Each images doc carries a fresh write id (:data:`WRITE_ID_FIELD`): a 409
  whose stored doc holds this unit's own write id answers a create an earlier attempt of the
  same request applied (a transport retry after a lost response), so the image counts as
  written and its items are written (#375).
- Items are written only under an images doc this unit wrote, never under another copy's
  or a refused one.
- A document the store refuses is reported in its own image's result (``refused``) and does
  not touch the others, as before. The unit fails as a whole (:class:`IngestWriteError`): a
  photo is stored whole or it is not reported stored.
- A photo stored whole or not at all: the images doc lists its items (``item_ids``,
  :mod:`src.domain.image_completeness`) and is written before them, so a failure or a kill
  between the two steps leaves a doc the next ingest can recognise as incomplete. The unit that
  saw its own items step fail marks the doc (best effort); a unit that finds another copy's
  incomplete doc after a 409 repairs it when no live writer can still be writing it, and answers
  ``incomplete`` otherwise. A repair writes the items (the same OCC upsert, human guards
  included) and then sets the list in one OCC update; it never replaces the images doc.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
import uuid
import weakref
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from typing import Any

from opensearchpy.exceptions import OpenSearchException
from sqlalchemy.exc import SQLAlchemyError

from src.clients.occ_upsert import DocOutcome, occ_upsert_bulk
from src.core.logging import get_logger
from src.domain.image_completeness import (
    INCOMPLETE_MARK_FIELD,
    INDEXED_AT_FIELD,
    ITEM_IDS_FIELD,
    listed_item_ids,
    repairable,
)
from src.services.curation.image_completeness import mark_incomplete, missing_items, set_item_ids
from src.services.embedding_spaces.readers import item_write_fields, vector_source_excludes


logger = get_logger(__name__)

MAX_BULK_DOCS = 1000
MAX_WRITERS = 4
LINGER_S = 0.02
# HTTP 429 (write queue full) is retried with a growing pause; anything else is final.
RETRY_429_PAUSES_S = (0.1, 0.25, 0.5, 1.0, 2.0)
WRITE_ID_FIELD = 'ingest_write_id'

# How long after its images doc was written an incomplete photo may still have a live writer.
# The writer's longest life is its requests in a row, each at most as long as the OpenSearch
# client lets one run:
#   one request  = OPENSEARCH_ATTEMPTS x CLIENT_TIMEOUT_S. opensearch-py retries a connection
#                  error and a 502/503/504 up to max_retries (3, so 4 attempts) and does not retry
#                  a timeout (retry_on_timeout is False), so 4 x the timeout is the most one
#                  request takes;
#   CLIENT_TIMEOUT_S = 60, the slowest client that writes images (the job workers:
#                  src/tasks/jobs_c.py, src/tasks/b4_project.py; the API's is opensearch_timeout, 30);
#   the requests of one unit in a row, worst case: the images step 3 (create bulk, the 409's mget
#                  of the stored doc, the mget of its listed items), the items step 7 (mget,
#                  create bulk, conflict refetch mget, update mget + bulk, one retry round of mget
#                  + bulk), the list write 6 (an OCC bulk update with up to 2 retry rounds: mget +
#                  bulk each) and the failure mark 4 (one retry round);
#   plus the 429 pauses of the images step and of the items step (RETRY_429_PAUSES_S).
# tests/curation/test_ingest_repair_age.py pins each input against the real client and settings.
OPENSEARCH_ATTEMPTS = 4
CLIENT_TIMEOUT_S = 60
IMAGES_STEP_REQUESTS = 3
ITEMS_STEP_REQUESTS = 7
LIST_WRITE_REQUESTS = 6
MARK_WRITE_REQUESTS = 4
SEQUENTIAL_REQUESTS = (
    IMAGES_STEP_REQUESTS + ITEMS_STEP_REQUESTS + LIST_WRITE_REQUESTS + MARK_WRITE_REQUESTS
)
REPAIR_AFTER_S = SEQUENTIAL_REQUESTS * OPENSEARCH_ATTEMPTS * CLIENT_TIMEOUT_S + 2 * sum(
    RETRY_429_PAUSES_S
)


class IngestWriteError(Exception):
    """A unit's documents were not all stored. ``release_claim`` is True only when the store
    refused the images doc (it is known absent): then the content claim goes, so the content can
    be ingested again. Any other failure keeps the claim, which is what routes the next ingest of
    the same bytes to the repair of the doc that exists."""

    def __init__(self, message: str, *, release_claim: bool) -> None:
        super().__init__(message)
        self.release_claim = release_claim


@dataclass
class UnitResult:
    """What one image's documents did; ``refused`` lists the reasons for any the store refused.

    ``duplicate``: the images doc already existed, written by another copy of the content
    (its write id is not this unit's), and is whole, so nothing of this unit was written.
    ``incomplete``: the same, but the doc misses items and a writer may still be writing it, so
    nothing was written and the photo is not reported stored. ``repaired``: the doc existed
    incomplete with no live writer, and this unit wrote the items and the list (``stale_ids``
    are the items the old list named that this write no longer does). ``image_absent``: the
    store refused the images doc. ``failure``: a step raised."""

    images_indexed: int = 0
    duplicate: bool = False
    incomplete: bool = False
    repaired: bool = False
    image_absent: bool = False
    failure: str | None = None
    stale_ids: list[str] = field(default_factory=list)
    created: int = 0
    updated: int = 0
    preserved_human: int = 0
    final_conflicts: int = 0
    filled_absent: int = 0
    created_ids: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RepairTarget:
    """The stored image a unit completes: its id and the items its doc lists now."""

    image_id: str
    listed: tuple[str, ...]


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
    repair: RepairTarget | None = None
    follows: _Unit | None = None
    """The unit of this flush that created the images doc this unit's create found (409): its
    answer is decided once that unit's items are written."""
    queued_at: float = field(default_factory=time.monotonic)

    @property
    def image_id(self) -> str | None:
        if self.image_doc is not None:
            return str(self.image_doc['image_id'])
        return self.repair.image_id if self.repair is not None else None

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
        repair: RepairTarget | None = None,
    ) -> UnitResult:
        """Queue one image's documents and wait for their own result. ``repair`` completes an
        image already stored incomplete (``image_doc`` is then None): its items are written and
        its list is set."""
        unit = _Unit(
            {
                **image_doc,
                WRITE_ID_FIELD: uuid.uuid4().hex,
                ITEM_IDS_FIELD: sorted(d['crop_id'] for d in crop_docs),
            }
            if image_doc
            else None,
            crop_docs,
            images_index,
            items_index,
            guards,
            fill_if_absent,
            contextvars.copy_context(),
            asyncio.get_running_loop().create_future(),
            repair,
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
        try:
            await self._write_images(units, results)
            await self._write_items(units, results)
            await self._write_lists(units, results)
        except (OpenSearchException, SQLAlchemyError) as exc:
            # What each unit got done stays in its result, so the caller knows whether its images
            # doc exists; the units that had a step left fail.
            logger.error('ingest_write_flush_failed', error=str(exc), n_images=len(units))
            for unit in units:
                results[id(unit)].failure = f'{type(exc).__name__}: {exc}'
        for unit in units:
            result = results[id(unit)]
            if unit.follows is not None:
                won = results[id(unit.follows)]
                result.duplicate = not (won.failure or won.refused)
                result.incomplete = not result.duplicate
            if (result.failure or result.refused) and _stored_image(unit, result):
                await mark_incomplete(self._client, unit.images_index, str(unit.image_id))
            if not unit.future.done():
                unit.future.set_result(result)

    async def _write_images(self, units: list[_Unit], results: dict[int, UnitResult]) -> None:
        conflicted = await self._create_images([u for u in units if u.image_doc], results)
        if conflicted:
            await self._settle_conflicts(conflicted, results, units)

    async def _create_images(
        self, todo: list[_Unit], results: dict[int, UnitResult]
    ) -> list[_Unit]:
        """``create`` the images docs; the units answered 409 are returned unsettled."""
        conflicted: list[_Unit] = []
        for pause in (*RETRY_429_PAUSES_S, None):
            if not todo:
                break
            body: list[dict[str, Any]] = []
            for u in todo:
                assert u.image_doc is not None
                body += [{'create': {'_index': u.images_index, '_id': u.image_doc['image_id']}}]
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
                break
            retry: list[_Unit] = []
            for unit, item in zip(todo, resp.get('items', []), strict=False):
                op = item.get('create') or {}
                status = op.get('status', 500)
                if status < 300:
                    results[id(unit)].images_indexed = 1
                elif status == 409:
                    conflicted.append(unit)
                elif status == 429 and pause is not None:
                    retry.append(unit)
                else:
                    results[id(unit)].refused.append(f'images doc: {_reason(op)}')
                    results[id(unit)].image_absent = True
            todo = retry
            if todo:
                await asyncio.sleep(pause or 0)
        return conflicted

    async def _settle_conflicts(
        self, units: list[_Unit], results: dict[int, UnitResult], group: list[_Unit]
    ) -> None:
        """Tell a 409 that answers another copy of the content from one that answers this
        writer's own create. The transport re-sends a bulk whose response it lost (a
        connection error, a 502/503/504), so a create an earlier attempt applied comes back
        409; the stored doc then carries this unit's write id (#375). One realtime
        ``mget`` reads the stored docs; a missing doc (deleted since) is refused.

        Another copy's doc is a ``duplicate`` only when it is whole (every item its list names
        exists). When it is not, this unit repairs it if no live writer can be writing it
        (:func:`~src.domain.image_completeness.repairable`), and otherwise reports ``incomplete``
        and writes nothing."""
        resp = await self._client.mget(
            index=units[0].images_index,
            body={'ids': [u.image_doc['image_id'] for u in units if u.image_doc]},
            _source_includes=[
                WRITE_ID_FIELD,
                ITEM_IDS_FIELD,
                INDEXED_AT_FIELD,
                INCOMPLETE_MARK_FIELD,
            ],
        )
        stored = {d['_id']: d for d in resp.get('docs', [])}
        winners = {u.image_doc[WRITE_ID_FIELD]: u for u in group if u.image_doc}
        others: dict[str, tuple[_Unit, dict[str, Any]]] = {}
        for unit in units:
            assert unit.image_doc is not None
            image_id, own_id = unit.image_doc['image_id'], unit.image_doc[WRITE_ID_FIELD]
            doc = stored.get(image_id) or {}
            result = results[id(unit)]
            if not doc.get('found'):
                result.refused.append('images doc: 409, and no stored doc to compare')
                result.image_absent = True
            elif (doc.get('_source') or {}).get(WRITE_ID_FIELD) == own_id:
                logger.info('ingest_image_create_was_own', image_id=image_id)
                result.images_indexed = 1
            elif (
                winner := winners.get((doc.get('_source') or {}).get(WRITE_ID_FIELD))
            ) is not None:
                unit.follows = winner  # created by a unit of this flush, whose items follow
            else:
                others[image_id] = (unit, doc.get('_source') or {})
        listed = {i: ids for i, (_, src) in others.items() if (ids := listed_item_ids(src))}
        gaps = await missing_items(self._client, units[0].items_index, listed) if listed else {}
        now = datetime.now(UTC)
        for image_id, (unit, source) in others.items():
            result = results[id(unit)]
            if image_id not in gaps:
                result.duplicate = True
            elif repairable(source, now=now, repair_after_s=REPAIR_AFTER_S):
                unit.repair = RepairTarget(image_id, tuple(listed[image_id]))
                result.repaired = True
            else:
                result.incomplete = True

    async def _write_items(self, units: list[_Unit], results: dict[int, UnitResult]) -> None:
        # Items only under an images doc this unit wrote (or, items-only, one that exists):
        # never under another copy's doc, nor under one that was refused.
        units = [
            u
            for u in units
            if u.image_doc is None or results[id(u)].images_indexed or results[id(u)].repaired
        ]
        remaining = [d for u in units for d in u.crop_docs]
        if not remaining:
            return
        first = units[0]
        outcomes: dict[str, DocOutcome] = {}
        write_fields = await item_write_fields()
        excludes = await vector_source_excludes()
        for pause in (*RETRY_429_PAUSES_S, None):
            try:
                await occ_upsert_bulk(
                    self._client,
                    remaining,
                    index=first.items_index,
                    human_field_guards=list(first.guards),
                    vector_excludes=excludes,
                    vector_field=write_fields.vector_field,
                    state_field=write_fields.state_field,
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

    async def _write_lists(self, units: list[_Unit], results: dict[int, UnitResult]) -> None:
        """A repair's last write: the list names the items just written (and the failure mark
        goes). Only when every item of the unit was stored."""
        for unit in units:
            result = results[id(unit)]
            if unit.repair is None or result.refused:
                continue
            await set_item_ids(
                self._client, unit.images_index, unit.repair.image_id, sorted(unit.ids)
            )
            result.repaired = True
            result.stale_ids = [i for i in unit.repair.listed if i not in unit.ids]

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


def _stored_image(unit: _Unit, result: UnitResult) -> bool:
    """The unit's images doc exists, written or adopted by it (so a failure leaves it incomplete)."""
    return bool(result.images_indexed or result.repaired or unit.repair is not None)


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
