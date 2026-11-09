"""Undo the newest human edit of a kind (region write, VLM dismissal) on one or many crops.

A batch of up to ``OP_BATCH_SYNC_MAX`` ids is undone by the request; a larger one is an
``edit_undo_batch`` job running :func:`run_edit_undo_job` over the ids in ``job_items``. Both go
through :func:`~src.clients.occ_bulk.occ_bulk_rounds`: one ``_mget`` and one conditional ``_bulk``
per page of 500, each crop's outcome decided from its own history inside the loop, so a person's
write that races the undo wins and the undo is re-read, not forced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.occ import OCC_BULK_MGET_SOURCE_EXCLUDES, OCC_BULK_PAGE_SIZE
from src.clients.occ_bulk import occ_bulk_rounds
from src.config.curation_accessors import items_index
from src.db.repos import job_items
from src.services.curation.edit_history import (
    EDIT_HISTORY_FIELD,
    REGION_UNDO_WRITER,
    EditKind,
    find_edit_undo,
    record_edit,
    restore_edit_state,
)
from src.services.jobs.service import start_job
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repos.jobs import JobRow
    from src.tasks.runtime.context import JobContext

KIND = 'edit_undo_batch'
UNDO_MAX_RETRIES = 5
OUTCOMES = ('undone', 'nothing_to_undo', 'conflicts', 'not_found')


def edit_undo_update(current: dict[str, Any], kind: EditKind, writer: str) -> dict[str, Any] | None:
    """The update restoring the state before the latest un-undone ``kind`` edit of ``current``
    (and recording the replaced state, ``restorable=False``); ``None`` when there is none."""
    entry = find_edit_undo(current.get(EDIT_HISTORY_FIELD), kind)
    if entry is None:
        return None
    history = record_edit(current, kind=kind, writer=writer, restorable=False)
    return {
        **restore_edit_state(entry, kind, current=current),
        EDIT_HISTORY_FIELD: history,
        'updated_at': now_iso(),
    }


async def undo_edit_writes(
    opensearch: Any, crop_ids: list[str], kind: EditKind, writer: str
) -> dict[str, list[str]]:
    """Undo the newest ``kind`` edit of each crop, one conditional bulk per page.

    The ids per outcome: ``undone``, ``nothing_to_undo`` (no edit of the kind left on record),
    ``conflicts`` (a concurrent write outlasted the retries) and ``not_found``.
    """
    outcome: dict[str, str] = {}

    def prepare(doc_id: str, doc: dict[str, Any] | None) -> dict[str, Any] | None:
        if doc is None:
            outcome[doc_id] = 'not_found'
            return None
        update = edit_undo_update(doc.get('_source') or {}, kind, writer)
        outcome[doc_id] = 'nothing_to_undo' if update is None else 'conflicts'  # until written
        return update

    def on_written(doc_id: str) -> None:
        outcome[doc_id] = 'undone'

    def on_failed(doc_id: str, _doc: dict[str, Any] | None, _why: str) -> None:
        outcome[doc_id] = 'conflicts'

    await occ_bulk_rounds(
        opensearch,
        index=items_index(),
        ids=crop_ids,
        prepare=prepare,
        on_written=on_written,
        on_failed=on_failed,
        max_retries=UNDO_MAX_RETRIES,
        refresh='wait_for',
        source_excludes=OCC_BULK_MGET_SOURCE_EXCLUDES,
    )
    by: dict[str, list[str]] = {name: [] for name in OUTCOMES}
    for crop_id in crop_ids:
        by[outcome.get(crop_id, 'conflicts')].append(crop_id)
    return by


async def queue_edit_undo_job(
    session: AsyncSession, project: uuid.UUID, crop_ids: list[str]
) -> JobRow:
    """Queue :data:`KIND` over ``crop_ids`` (kept in ``job_items``); 409 ``job_busy`` while one is active."""

    async def rows(inner: AsyncSession, job: JobRow) -> None:
        await job_items.add_items(inner, job.id, crop_ids)

    started = await start_job(
        session,
        KIND,
        project,
        params={'total': len(crop_ids)},
        requested_by='api',
        with_rows=rows,
    )
    return started.job


async def run_edit_undo_job(ctx: JobContext, opensearch: Any) -> dict[str, Any]:
    """The ``edit_undo_batch`` job body: the region undo of the ids in ``job_items``, page by page.

    The checkpoint holds the position and the outcome so far. An undo is not idempotent (it steps
    back one more edit), so the kind never auto-resumes; the page is the unit a manual resume redoes.
    """
    total = int(ctx.params['total'])
    cursor = ctx.resume_from or {}
    done = int(cursor.get('done', 0))
    by: dict[str, list[str]] = {name: list(cursor.get(name, [])) for name in OUTCOMES}
    while done < total and not ctx.cancelled():
        async with ctx.session() as session:
            page = await job_items.read_items(
                session, ctx.job_id, start=done, limit=OCC_BULK_PAGE_SIZE
            )
        if not page:
            break
        part = await undo_edit_writes(opensearch, page, EditKind.REGION, REGION_UNDO_WRITER)
        for name in OUTCOMES:
            by[name] += part[name]
        done += len(page)
        await ctx.checkpoint({'done': done, **by})
        await ctx.progress(done, total, 'undoing', {'undone': len(by['undone'])})
    return {
        'undone': len(by['undone']),
        **{k: by[k] for k in OUTCOMES[1:]},
        'complete': done >= total,
    }
