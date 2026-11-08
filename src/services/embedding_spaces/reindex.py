"""The re-index job's logic: time-sliced, resumable, no finished document encoded twice.

Embeddings plan 5.5. The handler in ``src/tasks/embedding_reindex.py`` binds this to the job
runtime; everything outside the process (the index, the encoder) is a port, so the logic runs
against fakes and the real adapters are swapped in by the worker image.

One batch is: read a page of document ids after the cursor, encode them, write the vectors with
a guard on the geometry read, count, record progress, then write the fenced checkpoint. The
checkpoint is the only thing a redelivery trusts, so a batch is repeated at most once and the
writes are idempotent (same vector, same field).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.db.repos import embedding_spaces as repo
from src.db.session import async_session_factory
from src.domain.embedding_spaces import EmbeddingSpaceError, Status, Target
from src.services.embedding_spaces import reindex_runs, spaces
from src.tasks.runtime.context import JobResult, Paused, Yield


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.tasks.runtime.context import JobContext

VERIFY_SAMPLE = 200
VERIFY_MIN_COSINE = 0.999
STATE_DEFERRED = 'deferred'
STATE_FAILED = 'failed'
Outcome = Literal['written', 'moved', 'missing']


class ReindexMode(StrEnum):
    BUILD = 'build'
    BACKFILL = 'backfill'
    CLEAR = 'clear'


class EmbeddingReindexParams(BaseModel):
    """The job's ``params`` (``space_key`` is also the exclusive scope)."""

    model_config = ConfigDict(extra='forbid')

    space_key: str
    target: Target
    mode: ReindexMode = ReindexMode.BUILD
    batch_size: int = Field(default=64, ge=1, le=512)
    scope: dict[str, Any] | None = None


class ReindexVerifyError(Exception):
    """The finished build did not pass the verify step; the space is marked ``failed``."""


@dataclass(frozen=True)
class DocRef:
    """A document to embed: its id and the geometry read with it (``None`` for an image)."""

    doc_id: str
    bbox_norm: tuple[float, ...] | None = None


@dataclass(frozen=True)
class VectorWrite:
    """``expected_bbox`` is what the page read; the write is skipped if the box moved since."""

    doc_id: str
    vector: Sequence[float]
    expected_bbox: tuple[float, ...] | None = None


class ReindexIndex(Protocol):
    """The OpenSearch side: PIT and ``search_after`` over ids only, OCC partial updates."""

    async def put_mapping(self, index_role: str, _properties: dict[str, Any]) -> None: ...

    async def count(self, space: repo.SpaceRow, mode: ReindexMode, scope: Mapping[str, Any]) -> int:
        """Documents the run will visit."""
        ...

    async def page(
        self,
        space: repo.SpaceRow,
        mode: ReindexMode,
        scope: Mapping[str, Any],
        after: str | None,
        size: int,
    ) -> list[DocRef]:
        """Up to ``size`` documents with ``_id`` greater than ``after`` (a fresh PIT per call
        sequence is the adapter's business: a redelivery resumes by id, not by PIT)."""
        ...

    async def write(
        self, space: repo.SpaceRow, writes: Sequence[VectorWrite]
    ) -> dict[str, Outcome]: ...

    async def mark(self, space: repo.SpaceRow, doc_ids: Sequence[str], state: str) -> None: ...

    async def count_with_vector(self, space: repo.SpaceRow) -> int: ...

    async def sample(self, space: repo.SpaceRow, size: int) -> list[DocRef]:
        """Random documents that hold a vector in this space."""
        ...

    async def read_vectors(
        self, space: repo.SpaceRow, doc_ids: Sequence[str]
    ) -> dict[str, Sequence[float]]: ...

    async def clear(self, space: repo.SpaceRow) -> None: ...


class Encoder(Protocol):
    """Decode each image once and encode the batch on Triton; a document that cannot be decoded
    or encoded is absent from the answer."""

    async def encode(
        self, space: repo.SpaceRow, docs: Sequence[DocRef]
    ) -> dict[str, Sequence[float]]: ...


@dataclass
class Cursor:
    after: str | None = None
    done: int = 0
    failed: int = 0
    skipped: int = 0
    total: int | None = None

    @classmethod
    def load(cls, saved: Any) -> Cursor:
        if not isinstance(saved, dict):
            return cls()
        return cls(
            after=saved.get('after'),
            done=int(saved.get('done', 0)),
            failed=int(saved.get('failed', 0)),
            skipped=int(saved.get('skipped', 0)),
            total=saved.get('total'),
        )

    def dump(self) -> dict[str, Any]:
        return {
            'after': self.after,
            'done': self.done,
            'failed': self.failed,
            'skipped': self.skipped,
            'total': self.total,
        }

    def counters(self) -> dict[str, int]:
        return {'failed': self.failed, 'skipped': self.skipped}


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


async def _space(ctx: JobContext, params: EmbeddingReindexParams) -> repo.SpaceRow:
    assert ctx.project_id is not None
    async with ctx.session() as session:
        row = await repo.get_space(session, ctx.project_id, params.target, params.space_key)
    if row is None:
        raise EmbeddingSpaceError(
            'embedding_space_not_found',
            f'no {params.target.value} space {params.space_key}',
            status=404,
        )
    return row


async def _begin(
    ctx: JobContext, params: EmbeddingReindexParams, row: repo.SpaceRow, index: ReindexIndex
) -> repo.SpaceRow:
    """``pending``/``failed`` -> ``building``; a ``building`` space is a redelivery or a restart,
    and a ``ready`` one is only valid for a ``backfill``."""
    assert ctx.project_id is not None
    if row.status in (Status.PENDING, Status.FAILED):
        async with async_session_factory()() as session:
            return await spaces.begin_build(
                session,
                ctx.project_id,
                target=params.target,
                key=params.space_key,
                put_mapping=index.put_mapping,
                job_id=ctx.job_id,
            )
    if row.status is Status.BUILDING or (
        row.status is Status.READY and params.mode is ReindexMode.BACKFILL
    ):
        return row
    raise EmbeddingSpaceError(
        'embedding_space_state', f'{row.space_key} is {row.status.value}; cannot re-index'
    )


async def _record(ctx: JobContext, cursor: Cursor) -> None:
    async with ctx.session() as session:
        await reindex_runs.record_counts(
            session,
            ctx.job_id,
            total=cursor.total,
            done=cursor.done,
            failed=cursor.failed,
            skipped=cursor.skipped,
        )
    await ctx.progress(cursor.done, cursor.total, 'encode', cursor.counters())
    await ctx.checkpoint(cursor.dump())


async def _batch(
    space: repo.SpaceRow,
    docs: Sequence[DocRef],
    index: ReindexIndex,
    encoder: Encoder,
    cursor: Cursor,
) -> None:
    vectors = await encoder.encode(space, docs)
    writes = [
        VectorWrite(d.doc_id, vectors[d.doc_id], d.bbox_norm) for d in docs if d.doc_id in vectors
    ]
    outcomes = await index.write(space, writes) if writes else {}
    deferred = [i for i, o in outcomes.items() if o == 'moved']
    failed = [d.doc_id for d in docs if d.doc_id not in vectors]
    if deferred:
        await index.mark(space, deferred, STATE_DEFERRED)
    if failed:
        await index.mark(space, failed, STATE_FAILED)
    cursor.done += sum(1 for o in outcomes.values() if o == 'written')
    cursor.skipped += len(deferred) + sum(1 for o in outcomes.values() if o == 'missing')
    cursor.failed += len(failed)
    cursor.after = docs[-1].doc_id


async def _verify(
    ctx: JobContext,
    space: repo.SpaceRow,
    index: ReindexIndex,
    encoder: Encoder,
    cursor: Cursor,
) -> None:
    await ctx.progress(cursor.done, cursor.total, 'verify', cursor.counters(), force=True)
    held = await index.count_with_vector(space)
    if held < cursor.done:
        raise ReindexVerifyError(f'{held} documents hold a vector, {cursor.done} were written')
    sample = await index.sample(space, VERIFY_SAMPLE)
    if not sample:
        return
    fresh = await encoder.encode(space, sample)
    stored = await index.read_vectors(space, [d.doc_id for d in sample])
    for doc in sample:
        if doc.doc_id not in fresh or doc.doc_id not in stored:
            raise ReindexVerifyError(f'{doc.doc_id} cannot be compared')
        if cosine(fresh[doc.doc_id], stored[doc.doc_id]) < VERIFY_MIN_COSINE:
            raise ReindexVerifyError(f'{doc.doc_id} re-encodes below cosine {VERIFY_MIN_COSINE}')


async def _clear(ctx: JobContext, params: EmbeddingReindexParams, index: ReindexIndex) -> Any:
    assert ctx.project_id is not None
    row = await _space(ctx, params)
    if row.status is not Status.RETIRING:
        raise EmbeddingSpaceError(
            'embedding_space_state', f'{row.space_key} is {row.status.value}; retire it first'
        )
    await index.clear(row)
    async with async_session_factory()() as session:
        await spaces.finish_retire(session, ctx.project_id, target=params.target, key=row.space_key)
    async with ctx.session() as session:
        await reindex_runs.finish_run(session, ctx.job_id)
    return {'mode': 'clear', 'space_key': row.space_key}


async def run_reindex(ctx: JobContext, index: ReindexIndex, encoder: Encoder) -> Any:
    """One delivery of the job: returns a result, :class:`Yield` at the slice, or :class:`Paused`.

    Cancel returns ``None`` at a batch boundary; the cancelled state is the wrapper's, and the
    space stays ``building`` with its partial field (re-startable as ``backfill``).
    """
    assert ctx.project_id is not None
    params = EmbeddingReindexParams.model_validate(dict(ctx.params))
    if params.mode is ReindexMode.CLEAR:
        return await _clear(ctx, params, index)
    space = await _begin(ctx, params, await _space(ctx, params), index)
    scope = params.scope or {}
    cursor = Cursor.load(ctx.resume_from)
    if cursor.total is None:
        cursor.total = await index.count(space, params.mode, scope)
    while True:
        if ctx.cancelled():
            return None
        if ctx.paused():
            return Paused()
        docs = await index.page(space, params.mode, scope, cursor.after, params.batch_size)
        if not docs:
            break
        await _batch(space, docs, index, encoder, cursor)
        await _record(ctx, cursor)
        if ctx.slice_due():
            return Yield()
    await _finish(ctx, params, space, index, encoder, cursor)
    return JobResult(
        {'done': cursor.done, 'failed': cursor.failed, 'skipped': cursor.skipped},
        with_errors=cursor.failed > 0,
    )


async def _finish(
    ctx: JobContext,
    params: EmbeddingReindexParams,
    space: repo.SpaceRow,
    index: ReindexIndex,
    encoder: Encoder,
    cursor: Cursor,
) -> None:
    assert ctx.project_id is not None
    try:
        await _verify(ctx, space, index, encoder, cursor)
    except ReindexVerifyError:
        async with async_session_factory()() as session:
            await spaces.mark_failed(
                session, ctx.project_id, target=params.target, key=space.space_key
            )
        raise
    if space.status is Status.BUILDING:
        async with async_session_factory()() as session:
            await spaces.mark_ready(
                session, ctx.project_id, target=params.target, key=space.space_key
            )
    async with ctx.session() as session:
        await reindex_runs.finish_run(session, ctx.job_id)


__all__ = [
    'DocRef',
    'EmbeddingReindexParams',
    'Encoder',
    'ReindexIndex',
    'ReindexMode',
    'ReindexVerifyError',
    'VectorWrite',
    'run_reindex',
]
