"""Write a bulk label or move as a recorded operation, on the job queue.

The start route queues a ``label_op`` job; in the same transaction it writes the operation's
record (status ``writing``, owned by the job) and the target ids into ``job_items``. The worker
pages through the ids, writes each page with the OCC human relabel and records the page's
members with the state the write replaced, then checkpoints. A job that dies leaves the record
``writing`` with no active job (served ``interrupted``); the pages it wrote can still be undone,
and a resume continues at the checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.occ import OCC_BULK_PAGE_SIZE
from src.config.project_context import current_project_id
from src.core.logging import get_logger
from src.db.repos import job_items
from src.db.repos.records import new_label_operation_id
from src.db.session import async_session_factory, project_tx
from src.services.curation.class_label import ItemLabel, candidate_move_update, class_label_update
from src.services.curation.crop_bulk import occ_bulk_human_relabel
from src.services.curation.label_ops.store import (
    NewLabelOp,
    create_op_in,
    read_op,
    save_op,
    write_members,
)
from src.services.jobs.service import start_job
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repos.jobs import JobRow
    from src.tasks.runtime.context import JobContext

logger = get_logger(__name__)

KIND = 'label_op'
type Merger = Callable[[dict[str, Any]], dict[str, Any]]


def merger_for(spec: dict[str, Any], op_id: str) -> Merger:
    """The merge that tags each item with the operation; ``spec`` is the stored write recipe."""
    if spec['kind'] == 'label':

        def label(current: dict[str, Any]) -> dict[str, Any]:
            return class_label_update(
                current,
                ItemLabel.human(
                    class_id=spec['class_id'],
                    class_name=spec['class_name'],
                    label_source=spec['label_source'],
                    writer=spec['writer'],
                    op_id=op_id,
                ),
            )

        return label

    def move(current: dict[str, Any]) -> dict[str, Any]:
        return candidate_move_update(
            current,
            partition_id=spec['partition_id'],
            cluster_uid=spec['target_uid'],
            now=now_iso(),
            op_id=op_id,
        )

    return move


def _recording(merge_for_op: Merger, op_id: str, priors: dict[str, Any]) -> Merger:
    """The operation's merge, keeping the state it replaced (the history snapshot it appends)
    in ``priors`` so the page's members can be recorded."""

    def merge(source: dict[str, Any]) -> dict[str, Any]:
        merged = merge_for_op(source)
        history = merged.get('class_id_history') or []
        if history and history[-1].get('op_id') == op_id:
            priors[source['crop_id']] = history[-1]
        return merged

    return merge


async def recorded_relabel(
    opensearch: Any,
    ids: list[str],
    *,
    op: NewLabelOp,
    merger_for: Callable[[str], Merger],
    job_id: uuid.UUID,
) -> dict[str, Any]:
    """Write ``ids`` as one recorded operation inside a job that is already running (the
    auto-label loop), owned by ``job_id``: the batch relabel response plus ``op_id``.
    ``merger_for(op_id)`` builds the merge body that tags each item with the operation."""
    async with async_session_factory()() as session, project_tx(session, current_project_id()):
        op_id = await new_label_operation_id(session)
        record = await create_op_in(session, op, op_id, job_id)
    priors: dict[str, Any] = {}
    merge = _recording(merger_for(record.op_id), record.op_id, priors)
    updated_ids: list[str] = []
    conflicts: list[dict[str, Any]] = []
    for start in range(0, len(ids), OCC_BULK_PAGE_SIZE):
        result = await occ_bulk_human_relabel(
            opensearch, ids[start : start + OCC_BULK_PAGE_SIZE], merge, writer_id=op.writer
        )
        await write_members(record, [(i, priors[i]) for i in result['updated_ids'] if i in priors])
        priors.clear()
        updated_ids += result['updated_ids']
        conflicts += result['conflicts']
    record.status = 'written'
    record.count = len(updated_ids)
    record.conflicts = len(conflicts)
    record.finished_at = now_iso()
    await save_op(record)
    logger.info(
        'label_op_written', op_id=record.op_id, count=record.count, conflicts=record.conflicts
    )
    return {
        'updated': len(updated_ids),
        'updated_ids': updated_ids,
        'conflicts': conflicts,
        'op_id': record.op_id,
    }


async def start(
    session: AsyncSession,
    project: uuid.UUID,
    *,
    op: NewLabelOp,
    spec: dict[str, Any],
    ids: list[str],
) -> tuple[uuid.UUID, JobRow]:
    """Queue the relabel: the record, the target ids and the job in one transaction.
    Returns ``(op_id, job)``."""
    async with session.begin():
        op_id = await new_label_operation_id(session)

    async def rows(inner: AsyncSession, job: JobRow) -> None:
        await create_op_in(inner, op, op_id, job.id)
        await job_items.add_items(inner, job.id, ids)

    started = await start_job(
        session,
        KIND,
        project,
        params={'op_id': str(op_id), 'spec': spec, 'writer': op.writer, 'total': len(ids)},
        requested_by='api',
        with_rows=rows,
    )
    return op_id, started.job


async def run_relabel(ctx: JobContext, opensearch: Any) -> dict[str, Any]:
    """Execute the job: page by page, record members, checkpoint; resumes at the checkpoint."""
    params = ctx.params
    op_id = params['op_id']
    record = await read_op(op_id)
    merge_for_op = merger_for(params['spec'], op_id)
    priors: dict[str, Any] = {}

    merge = _recording(merge_for_op, op_id, priors)

    cursor = ctx.resume_from or {}
    done = int(cursor.get('done', 0))
    updated = int(cursor.get('updated', 0))
    conflicts = int(cursor.get('conflicts', 0))
    total = int(params['total'])
    while done < total:
        if ctx.cancelled():
            break
        async with ctx.session() as session:
            page = await job_items.read_items(
                session, ctx.job_id, start=done, limit=OCC_BULK_PAGE_SIZE
            )
        result = await occ_bulk_human_relabel(opensearch, page, merge, writer_id=params['writer'])
        await write_members(record, [(i, priors[i]) for i in result['updated_ids'] if i in priors])
        priors.clear()
        done += len(page)
        updated += len(result['updated_ids'])
        conflicts += len(result['conflicts'])
        await ctx.checkpoint({'done': done, 'updated': updated, 'conflicts': conflicts})
        await ctx.progress(done, total, 'writing', {'updated': updated, 'conflicts': conflicts})
    if done >= total:
        record.status = 'written'
        record.count = updated
        record.conflicts = conflicts
        record.finished_at = now_iso()
        await save_op(record)
        logger.info('label_op_written', op_id=op_id, count=updated, conflicts=conflicts)
    return {'op_id': op_id, 'updated': updated, 'conflicts': conflicts, 'complete': done >= total}


__all__ = ['KIND', 'merger_for', 'recorded_relabel', 'run_relabel', 'start']
