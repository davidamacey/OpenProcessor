"""Records of label operations: a ``label_operations`` row per operation and a
``label_operation_members`` row per item it wrote, in the bound project (``src/db/repos/records.py``).

The row holds the record (kind, status, writer, the fingerprint of what the operation wrote and
the undo passes in ``params``); the members are the ids the operation wrote. Liveness is the
``label_op:<op_id>`` lease while the operation writes and the ``label_undo`` lease while any undo of
the project runs. The id is matched against :data:`LABEL_OP_ID_RE` before any query is built.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

from src.config.project_context import current_project, current_project_id
from src.core.logging import get_logger
from src.db.repos import jobs as jobs_repo
from src.db.repos.records import (
    MEMBER_BATCH,
    LabelOperationMemberLoad,
    LabelOperationRow,
    add_label_operation_members,
    create_label_operation,
    get_label_operation,
    label_operation_members,
    list_label_operations,
    prune_label_operations,
    save_label_operation,
)
from src.db.session import async_session_factory, project_tx
from src.domain import job_states as js
from src.domain.label_ops import (
    LABEL_OP_ID_RE,
    LabelOpKind,
    OpFingerprint,
    label_fingerprint,
    move_fingerprint,
)
from src.services.rejections import rejection
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

logger = get_logger(__name__)

LABEL_OPS_KEEP = 200
"""Operations kept per project; the oldest are pruned when a new one is created."""

UNDO_PASSES_KEPT = 10

StoredStatus = Literal['writing', 'written', 'undoing', 'undone']
ServedStatus = Literal['writing', 'interrupted', 'written', 'undoing', 'undo_interrupted', 'undone']


class LabelOpRecord(BaseModel):
    """One operation's stored state (schema 1)."""

    schema_version: int = 1
    op_id: str
    kind: LabelOpKind
    writer: str
    at: str
    finished_at: str | None = None
    class_name: str | None = None
    class_id: int | None = None
    target_cluster_id: int | None = None
    target_kind: Literal['class', 'candidate'] | None = None
    label_source: str | None = None
    target: Literal['crop_ids', 'selection']
    selected: int
    count: int | None = None
    conflicts: int | None = None
    fingerprint: OpFingerprint
    status: StoredStatus = 'writing'
    undo_passes: list[dict[str, Any]] = []


class NewLabelOp(BaseModel):
    """What the write route knows before it writes."""

    kind: LabelOpKind
    writer: str
    class_name: str | None
    class_id: int | None
    target_cluster_id: int | None = None
    target_kind: Literal['class', 'candidate'] | None = None
    label_source: str | None = None
    target: Literal['crop_ids', 'selection']
    selected: int


def _not_found(op_id: str) -> Exception:
    return rejection(404, 'label_op_not_found', f'no label operation {op_id!r} in this project')


def parse_op_id(op_id: str) -> uuid.UUID:
    """The id as a uuid; ``404`` for an id of the wrong shape (no query is made)."""
    if not LABEL_OP_ID_RE.fullmatch(op_id):
        raise _not_found(op_id)
    return uuid.UUID(op_id)


def _params(record: LabelOpRecord) -> dict[str, Any]:
    return record.model_dump(mode='json', exclude={'op_id', 'kind', 'status', 'schema_version'})


def _record(row: LabelOperationRow) -> LabelOpRecord | None:
    try:
        return LabelOpRecord.model_validate(
            {
                **row.params,
                'op_id': str(row.id),
                'kind': row.kind,
                'status': row.status,
            }
        )
    except ValidationError as exc:
        logger.warning('label_op_record_unreadable', op_id=str(row.id), error=str(exc))
        return None


async def create_op_in(
    session: Any, new: NewLabelOp, op_id: uuid.UUID, job_id: uuid.UUID
) -> LabelOpRecord:
    """Write the record (status ``writing``, owned by ``job_id``) in the caller's transaction,
    before any item is written, and prune the oldest operations beyond :data:`LABEL_OPS_KEEP`."""
    if current_project().read_only:
        raise rejection(409, 'project_read_only', 'the project is read-only; nothing was written')
    project_id = current_project_id()
    at = now_iso()
    if new.kind == 'label':
        assert new.class_name is not None
        fingerprint = label_fingerprint(
            op_id=str(op_id), writer=new.writer, at=at, class_name=new.class_name
        )
    else:
        fingerprint = move_fingerprint(
            op_id=str(op_id),
            writer=new.writer,
            at=at,
            class_name=new.class_name if new.target_kind == 'class' else None,
            cluster_id=new.target_cluster_id,
        )
    record = LabelOpRecord(op_id=str(op_id), at=at, fingerprint=fingerprint, **new.model_dump())
    await create_label_operation(
        session,
        project_id,
        kind=record.kind,
        created_by=record.writer,
        params=_params(record),
        op_id=op_id,
        job_id=job_id,
    )
    await prune_label_operations(session, project_id, keep=LABEL_OPS_KEEP)
    logger.info('label_op_created', op_id=record.op_id, kind=record.kind, selected=record.selected)
    return record


async def save_op(record: LabelOpRecord) -> None:
    project_id = current_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        await save_label_operation(
            session,
            project_id,
            parse_op_id(record.op_id),
            status=record.status,
            params=_params(record),
            undone_at=datetime.now(UTC) if record.status == 'undone' else None,
        )


async def read_op(op_id: str) -> LabelOpRecord:
    """The stored record; ``404`` when it is unknown, unreadable or pruned."""
    key = parse_op_id(op_id)
    project_id = current_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        row = await get_label_operation(session, project_id, key)
    record = None if row is None else _record(row)
    if record is None:
        raise _not_found(op_id)
    return record


async def list_ops(*, limit: int, after: str | None = None) -> list[LabelOpRecord]:
    """Readable records, newest first (ids sort in creation order), older than
    ``after`` (exclusive) when given. An unreadable one is skipped and logged."""
    project_id = current_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        rows = await list_label_operations(
            session, project_id, limit=limit, after=None if after is None else parse_op_id(after)
        )
    return [r for r in map(_record, rows) if r is not None]


async def write_members(record: LabelOpRecord, members: Iterable[tuple[str, Any]]) -> None:
    """Record the items the operation wrote, each with its state before the write (``prior``)."""
    async with async_session_factory()() as session, project_tx(session, current_project_id()):
        await add_label_operation_members(
            session,
            parse_op_id(record.op_id),
            (LabelOperationMemberLoad(item_id=i, prior=p) for i, p in members),
        )


async def member_pages(record: LabelOpRecord) -> AsyncIterator[list[LabelOperationMemberLoad]]:
    """The operation's members with their prior state, in pages of :data:`MEMBER_BATCH`."""
    key = parse_op_id(record.op_id)
    after: str | None = None
    while True:
        async with async_session_factory()() as session, project_tx(session, current_project_id()):
            page = await label_operation_members(session, key, after=after, limit=MEMBER_BATCH)
        if not page:
            return
        yield page
        after = page[-1].item_id


async def served_status(record: LabelOpRecord) -> ServedStatus:
    """The stored status, with a dead writer or undoer shown as ``interrupted`` /
    ``undo_interrupted``: the record still says it is working but no active job owns the work."""
    if record.status not in ('writing', 'undoing'):
        return record.status
    project_id = current_project_id()
    key = parse_op_id(record.op_id)
    async with async_session_factory()() as session, project_tx(session, project_id):
        if record.status == 'writing':
            row = await get_label_operation(session, project_id, key)
            job_id = None if row is None else row.job_id
            job = (
                None
                if job_id is None
                else await session.run_sync(lambda sync: jobs_repo.get(sync, job_id))
            )
            alive = job is not None and job.status in js.ACTIVE
            return 'writing' if alive else 'interrupted'
        undoer = await session.run_sync(
            lambda sync: jobs_repo.active_job_id(sync, project_id, 'label_undo', '')
        )
    return 'undoing' if undoer is not None else 'undo_interrupted'


__all__ = [
    'LABEL_OPS_KEEP',
    'UNDO_PASSES_KEPT',
    'LabelOpRecord',
    'NewLabelOp',
    'ServedStatus',
    'StoredStatus',
    'create_op_in',
    'list_ops',
    'member_pages',
    'parse_op_id',
    'read_op',
    'save_op',
    'served_status',
    'write_members',
]
