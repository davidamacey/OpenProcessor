"""Resolve the pending VLM new-class proposals of a term: the write half of
``POST /review/new_class_proposals/resolve``.

The request writes a batch of up to ``OP_BATCH_SYNC_MAX`` ids itself; a larger one is a
``review_resolve_batch`` job that runs :func:`run_resolve_job` over the ids in ``job_items``. Both
write through :func:`~src.clients.occ_bulk.occ_bulk_rounds`: one ``_mget`` and one conditional
``_bulk`` per page of 500, a 409 re-read and retried, the proposal re-checked against the fresh
document on every attempt, so an item that someone resolved in the meantime is skipped, never
overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.clients.occ import OCC_BULK_MGET_SOURCE_EXCLUDES, OCC_BULK_PAGE_SIZE
from src.clients.occ_bulk import BulkFailure, occ_bulk_rounds
from src.config.curation_accessors import items_index
from src.core.logging import get_logger
from src.db.repos import job_items
from src.services.curation.class_label import ItemLabel, class_label_update
from src.services.curation.new_class_terms import is_open_proposal
from src.services.jobs.service import start_job


if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repos.jobs import JobRow
    from src.tasks.runtime.context import JobContext

logger = get_logger(__name__)

KIND = 'review_resolve_batch'
RESOLVE_WRITER = 'human:resolve_new_class'
RESOLVE_MAX_RETRIES = 5


@dataclass
class ResolveOutcome:
    """What a resolve did, per crop id."""

    updated_ids: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)


async def resolve_pending_proposals(
    opensearch: Any,
    crop_ids: list[str],
    *,
    label: str,
    class_id: int,
    class_name: str,
    label_source: str,
) -> ResolveOutcome:
    """Label every crop in ``crop_ids`` that is still an open proposal for ``label``.

    A crop that is gone or no longer pending this proposal is ``skipped`` and not written; one
    whose write kept conflicting (or failed) is a ``conflicts`` entry carrying its current
    ``class_source``. Every write records a restorable ``class_id_history`` snapshot.
    """
    out = ResolveOutcome()
    human = ItemLabel.human(
        class_id=class_id,
        class_name=class_name,
        label_source=label_source,
        writer=RESOLVE_WRITER,
    )

    def prepare(doc_id: str, doc: dict[str, Any] | None) -> dict[str, Any] | None:
        source = None if doc is None else doc.get('_source') or {}
        if source is None or not is_open_proposal(source, label):
            out.skipped.append(doc_id)
            return None
        return {**class_label_update(source, human), 'needs_new_class': False}

    def on_written(doc_id: str) -> None:
        out.updated_ids.append(doc_id)

    def on_failed(doc_id: str, doc: dict[str, Any] | None, _why: BulkFailure) -> None:
        source = (doc or {}).get('_source') or {}
        out.conflicts.append({'crop_id': doc_id, 'current_source': source.get('class_source')})

    await occ_bulk_rounds(
        opensearch,
        index=items_index(),
        ids=crop_ids,
        prepare=prepare,
        on_written=on_written,
        on_failed=on_failed,
        max_retries=RESOLVE_MAX_RETRIES,
        source_excludes=OCC_BULK_MGET_SOURCE_EXCLUDES,
    )
    return out


async def refresh_items(opensearch: Any) -> None:
    """Make the written labels visible to the queue and summary reads."""
    try:
        await opensearch.indices.refresh(index=items_index())
    except OpenSearchException as exc:
        logger.warning('resolve_new_class_refresh_failed', error=str(exc))


def response_body(params: dict[str, Any], matched: int, outcome: ResolveOutcome) -> dict[str, Any]:
    """The job result (and the part of the 200 body the write decides): counts and id lists."""
    return {
        'class_id': params['class_id'],
        'class_name': params['class_name'],
        'created': params['created'],
        'label': params['label'],
        'matched': matched,
        'updated': len(outcome.updated_ids),
        'updated_ids': outcome.updated_ids,
        'conflicts': outcome.conflicts,
        'skipped': outcome.skipped,
    }


async def queue_resolve_job(
    session: AsyncSession, project: uuid.UUID, params: dict[str, Any], crop_ids: list[str]
) -> JobRow:
    """Queue :data:`KIND` over ``crop_ids`` (kept in ``job_items``); 409 ``job_busy`` while one is active."""

    async def rows(inner: AsyncSession, job: JobRow) -> None:
        await job_items.add_items(inner, job.id, crop_ids)

    started = await start_job(
        session,
        KIND,
        project,
        params={**params, 'total': len(crop_ids)},
        requested_by='api',
        with_rows=rows,
    )
    return started.job


async def run_resolve_job(ctx: JobContext, opensearch: Any) -> dict[str, Any]:
    """The ``review_resolve_batch`` job body: the ids in ``job_items``, page by page.

    The checkpoint holds the position and the outcome so far, so a resume continues at the page
    where it stopped; a page already written reads as not pending and cannot be written twice.
    """
    params = dict(ctx.params)
    total = int(params['total'])
    cursor = ctx.resume_from or {}
    done = int(cursor.get('done', 0))
    out = ResolveOutcome(
        updated_ids=list(cursor.get('updated_ids', [])),
        skipped=list(cursor.get('skipped', [])),
        conflicts=list(cursor.get('conflicts', [])),
    )
    while done < total and not ctx.cancelled():
        async with ctx.session() as session:
            page = await job_items.read_items(
                session, ctx.job_id, start=done, limit=OCC_BULK_PAGE_SIZE
            )
        if not page:
            break
        part = await resolve_pending_proposals(
            opensearch,
            page,
            label=params['label'],
            class_id=params['class_id'],
            class_name=params['class_name'],
            label_source=params['label_source'],
        )
        out.updated_ids += part.updated_ids
        out.skipped += part.skipped
        out.conflicts += part.conflicts
        done += len(page)
        await ctx.checkpoint(
            {
                'done': done,
                'updated_ids': out.updated_ids,
                'skipped': out.skipped,
                'conflicts': out.conflicts,
            }
        )
        await ctx.progress(done, total, 'writing', {'updated': len(out.updated_ids)})
    if out.updated_ids:
        await refresh_items(opensearch)
    return {**response_body(params, total, out), 'complete': done >= total}
