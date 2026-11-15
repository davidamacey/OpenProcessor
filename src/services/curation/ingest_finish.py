"""The finish phase of an ingest: the photos of one request, each moving on its own to its write.

:func:`finish_photos` runs one request's photos as a small pipeline instead of barrier steps:

::

    todo queue ──> K stage workers ──> inbox ──> one committer ──> commit groups (at most 3)
                   (one photo each:                 (classes, then the group transaction,
                    engine or host stage,            then the writes, one group at a time)
                    see ingest_stage)

* **Stage workers**, ``K = min(MAX_INGEST_CONCURRENCY, photos)``, pull photo indexes from an
  ``asyncio.Queue`` that is filled before they start; a worker ends when the queue is empty (there
  is no producer, so that is the end-of-input signal; ``Queue.shutdown`` is Python 3.13) and puts
  one end marker on the inbox. A photo ends in the inbox as a
  :class:`~src.services.curation.ingest_stage.StagedPhoto`, or as a result of its own (a
  duplicate, a failure): a photo failure is a value, never an exception that cancels the others.
* **The committer** takes the photos waiting in the inbox as one commit group when at least
  ``ceil(photos / 3)`` wait, or when every worker has ended. The third group waits for the last
  photo. Per group, in this order: the registry classes of the group's items are resolved (in
  memory; one class transaction only for a label new to the project), the group transaction places,
  claims and marks pending (:func:`~src.services.curation.ingest_index.commit_group`), the
  documents are built and the photos written together
  (:func:`~src.services.curation.ingest_index.write_group`). The next group starts when this one
  is written; the stage workers keep the GPU busy meanwhile.
* **The finish** (release the claims that must be released, announce every stored photo in one
  transaction) runs once per request in a ``finally`` and survives a cancellation of the request
  (:func:`~src.utils.task_group.finish_despite_cancel`): a stored photo never loses its events or
  its claim release.

Every task group goes through :func:`~src.utils.task_group.run_all`, which re-raises the first
failure unchanged, so a pool timeout still reaches the 503 handlers of ``src/core/db_session.py``.

What a photo ends with when a step fails:

::

    failure point             what the photo ends with
    ------------------------  -----------------------------------------------------------------
    engine call fails         ``failed``: ``engine_unavailable`` (down, overloaded past its
                              deadline), ``decode_failed`` (a file the engine cannot read),
                              ``detector_infer`` (a model branch errored); others go on
    class create fails        the photos that carry that class name ``failed``
                              (``detector_infer``); others of the group go on
    claim: content held       ``duplicate`` of the holder (``incomplete_write`` when the holder is
    by another image          not stored whole); no pending row, no write
    claim: key changed        ``failed`` (``bulk_index``); nothing stored
    or refused
    group 1: pool timeout     the request fails (503): nothing was stored
    or no connection
    group 2 or 3: the same    the stored photos keep ``success``; the photos of that group and
                              every photo not yet committed answer ``failed`` /
                              ``database_busy``; the stage workers stop taking photos
    OpenSearch write fails    ``failed`` (``bulk_index``); the claim is released only when the
                              store refused the images doc (the rules of ``write_prepared``)
    ``crop.created``          announced once, at the end, for every photo stored
    publish fails             the request fails; documents and claims stay

A photo of one content as another (byte-identical) answers ``duplicate`` from the first one's
result; two photos with one ``imohash`` and other bytes finish in separate waves.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy.exc import OperationalError, TimeoutError as PoolTimeoutError

from src.core.logging import get_logger
from src.domain.embedding_spaces import Target
from src.services.curation.ingest_gpu_decode import engine_ready, engine_stage_applies
from src.services.curation.ingest_helpers import MAX_INGEST_CONCURRENCY
from src.services.curation.ingest_index import (
    IndexOutcome,
    commit_group,
    doc_context,
    finish_commit,
    write_group,
)
from src.services.curation.ingest_models import (
    ERROR_KIND_DATABASE_BUSY,
    ERROR_KIND_DETECTOR_INFER,
    IngestResult,
)
from src.services.curation.ingest_stage import StagedPhoto, failed_result, stage_engine, stage_host
from src.services.curation.ops_metrics import record_ingest_result
from src.services.embedding_spaces.encoder_guard import require_encoder_for
from src.utils.task_group import finish_despite_cancel, run_all


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.ingest_dedup import Resolved


logger = get_logger(__name__)

#: A request's photos are committed in at most this many groups.
COMMIT_GROUPS = 3

#: Database failures the 503 handlers of ``src/core/db_session.py`` answer: no pool connection
#: within the wait, no connection at all.
_DATABASE_BUSY = (PoolTimeoutError, OperationalError)


@dataclass
class PhotoJob:
    """One photo of a request and what the request already knows about it."""

    image_bytes: bytes
    image_path: str
    source: str = 'unknown'
    source_identifier: str | None = None
    ingest_run_id: str | None = None
    resolved: Resolved | None = None
    """The photo's content looked up against the stored contents (the request's one lookup); a
    photo without it is looked up on its own."""


@dataclass
class _Flow:
    """What the workers and the committer of ONE call share; nothing outlives the call."""

    results: list[IngestResult | None]
    inbox: asyncio.Queue[tuple[int, StagedPhoto] | None] = field(default_factory=asyncio.Queue)
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    """Set when the database stopped taking commit groups: the workers take no more photos."""
    written: list[IndexOutcome] = field(default_factory=list)
    unclaimed: list[str] = field(default_factory=list)
    stored_groups: int = 0


async def _stage_worker(
    service: CurationIngestService,
    jobs: Sequence[PhotoJob],
    todo: asyncio.Queue[int],
    flow: _Flow,
    *,
    use_engine: bool,
    ready: bool,
) -> None:
    try:
        while not flow.stop.is_set():
            try:
                i = todo.get_nowait()
            except asyncio.QueueEmpty:
                break
            staged = await (
                stage_engine(service, jobs[i], engine_ready=ready)
                if use_engine
                else stage_host(service, jobs[i])
            )
            if isinstance(staged, IngestResult):
                flow.results[i] = staged
            else:
                flow.inbox.put_nowait((i, staged))
    finally:
        flow.inbox.put_nowait(None)  # this worker's end marker


def _finalize(result: IngestResult, staged: StagedPhoto) -> IngestResult:
    result.n_filtered = staged.n_filtered
    if result.status != 'duplicate':
        result.decode_route = staged.prepared.ctx.decode_route
    return result


def _database_busy(job: PhotoJob, staged: StagedPhoto) -> IngestResult:
    return failed_result(
        job,
        ERROR_KIND_DATABASE_BUSY,
        'the database is busy; retry later',
        staged.prepared.ctx.decode_route,
    )


async def _commit(
    service: CurationIngestService,
    jobs: Sequence[PhotoJob],
    flow: _Flow,
    group: list[tuple[int, StagedPhoto]],
) -> None:
    """Resolve the classes of ``group``, run its transaction, write its photos."""
    if flow.stop.is_set():
        for i, staged in group:
            flow.results[i] = _database_busy(jobs[i], staged)
        return
    class_errors = await service.resolve_classes([s.prepared.items for _, s in group])
    live: list[tuple[int, StagedPhoto]] = []
    for (i, staged), error in zip(group, class_errors, strict=True):
        if error is None:
            live.append((i, staged))
        else:
            flow.results[i] = failed_result(
                jobs[i], ERROR_KIND_DETECTOR_INFER, str(error), staged.prepared.ctx.decode_route
            )
    if not live:
        return
    prepared = [s.prepared for _, s in live]
    try:
        shared = await doc_context(service, prepared)
        claims = await commit_group(service, prepared)
    except _DATABASE_BUSY as exc:
        if flow.stored_groups == 0:
            raise  # nothing of this request is stored: the 503 handlers answer, the client resends
        logger.warning('ingest_group_database_busy', n_photos=len(live), error=type(exc).__name__)
        flow.stop.set()
        for i, staged in live:
            flow.results[i] = _database_busy(jobs[i], staged)
        return
    outcomes = await write_group(service, prepared, claims, shared, flow.written, flow.unclaimed)
    flow.stored_groups += 1
    for (i, staged), outcome in zip(live, outcomes, strict=True):
        flow.results[i] = _finalize(outcome.result, staged)


async def _committer(
    service: CurationIngestService,
    jobs: Sequence[PhotoJob],
    flow: _Flow,
    n_workers: int,
) -> None:
    per_group = math.ceil(len(jobs) / COMMIT_GROUPS)
    waiting: list[tuple[int, StagedPhoto]] = []
    ended = 0
    groups = 0
    while ended < n_workers or waiting:
        if ended < n_workers:
            item = await flow.inbox.get()
            items = [item]
            while not flow.inbox.empty():
                items.append(flow.inbox.get_nowait())
            for entry in items:
                if entry is None:
                    ended += 1
                else:
                    waiting.append(entry)
        all_staged = ended == n_workers
        early = groups < COMMIT_GROUPS - 1 and len(waiting) >= per_group
        if waiting and (all_staged or early):
            group, waiting = waiting, []
            await _commit(service, jobs, flow, group)
            groups += 1


async def finish_photos(
    service: CurationIngestService, jobs: Sequence[PhotoJob]
) -> list[IngestResult]:
    """Ingest ``jobs`` and answer one result per job, in order (see the module docstring)."""
    if not jobs:
        return []
    await require_encoder_for(Target.ITEM)
    use_engine = engine_stage_applies(service)
    ready = await engine_ready(service) if use_engine else False
    flow = _Flow(results=[None] * len(jobs))
    todo: asyncio.Queue[int] = asyncio.Queue()
    for i in range(len(jobs)):
        todo.put_nowait(i)
    n_workers = min(MAX_INGEST_CONCURRENCY, len(jobs))
    try:
        await run_all(
            [
                _stage_worker(service, jobs, todo, flow, use_engine=use_engine, ready=ready)
                for _ in range(n_workers)
            ]
            + [_committer(service, jobs, flow, n_workers)]
        )
    finally:
        if flow.written or flow.unclaimed:
            await finish_despite_cancel(finish_commit(service, flow.written, flow.unclaimed))
    results: list[IngestResult] = []
    for i, done in enumerate(flow.results):
        if done is None and not flow.stop.is_set():
            msg = f'photo {jobs[i].image_path} ended the flow without a result'
            raise RuntimeError(msg)
        results.append(
            record_ingest_result(
                done
                or failed_result(  # never taken: the database stopped taking groups
                    jobs[i], ERROR_KIND_DATABASE_BUSY, 'the database is busy; retry later'
                )
            )
        )
    return results


__all__ = ['COMMIT_GROUPS', 'PhotoJob', 'finish_photos']
