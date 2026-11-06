"""Holdout freezes, label-operation records and project markers.

Three small record families that used to be files under the project's state directory:

- ``holdout_freezes``: the audit trail of every freeze, one row current per project. Membership
  stays the items' ``test_holdout`` flag in OpenSearch; the row records who froze how many.
- ``label_operations`` and ``label_operation_members``: the record of a bulk label or move and
  the item ids it wrote with the fields it replaced, so an undo reads its prior state here.
- ``project_state``: small per-project markers (``region_drain``, ``region_dependency_health``)
  written with optimistic concurrency on ``revision``.

Every function takes a caller-owned session and never commits; the caller's ``project_tx`` (or
``deployment_tx`` for the ``load_*`` functions of the 0.6.0 upgrade) owns the transaction.
Queries filter on the project even though row-level security already does, so a wrong scope
returns nothing instead of another project's rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, insert as pg_insert

from src.db.models import HoldoutFreeze, LabelOperation, LabelOperationMember, Project, ProjectState
from src.db.models.config import PROJECT_STATE_KEYS
from src.db.models.datasets import LABEL_OPERATION_STATUSES


if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


LABEL_OPERATION_KINDS = ('label', 'move', 'assist_loop_propagate', 'assist_loop_review')
"""The kinds of ``label_operations.kind`` (a free-text column; this tuple is the one vocabulary)."""

MEMBER_BATCH = 5_000
"""Members per INSERT: 5,000 rows x 3 bind values stays far below the 65,535-parameter limit."""

_PROJECT_STATE_RETRIES = 8


class RecordRevisionConflict(Exception):  # noqa: N818  # named for the condition, like the 409 it becomes
    """A write named a ``revision`` that is no longer current."""


class RecordNotFoundError(LookupError):
    """The record a write named does not exist in this scope."""


# --- holdout freezes --------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutFreezeRow:
    id: uuid.UUID
    project_id: uuid.UUID
    is_current: bool
    source: str
    item_count: int
    seed: int | None
    snapshot_storage_key: str | None
    frozen_at: dt.datetime
    frozen_by: str


@dataclass(frozen=True)
class HoldoutFreezeLoad:
    """One 0.6.0 freeze, ids and times kept. The history is oldest first; one may be current."""

    id: uuid.UUID
    is_current: bool
    source: str
    item_count: int
    frozen_at: dt.datetime
    frozen_by: str
    seed: int | None = None
    snapshot_storage_key: str | None = None


def _freeze_row(row: HoldoutFreeze) -> HoldoutFreezeRow:
    return HoldoutFreezeRow(
        id=row.id,
        project_id=row.project_id,
        is_current=row.is_current,
        source=row.source,
        item_count=row.item_count,
        seed=row.seed,
        snapshot_storage_key=row.snapshot_storage_key,
        frozen_at=row.frozen_at,
        frozen_by=row.frozen_by,
    )


async def freeze_holdout(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    source: str,
    item_count: int,
    frozen_by: str,
    seed: int | None = None,
    snapshot_storage_key: str | None = None,
    make_current: bool = True,
) -> HoldoutFreezeRow:
    """Record a freeze. A current one demotes the previous current row in the same transaction.

    Concurrent freezes of one project queue on the project row, so each sees the other's commit
    and the partial unique index ``uq_holdout_current`` never fires: both land in the history and
    exactly one stays current. ``make_current=False`` records a freeze that is not the project's
    standing split (an import's), leaving the current row alone.
    """
    if item_count < 0:
        msg = 'item_count must not be negative'
        raise ValueError(msg)
    locked = (
        await session.execute(
            select(Project.id).where(Project.id == project_id).with_for_update(key_share=True)
        )
    ).scalar_one_or_none()
    if locked is None:
        msg = f'no project {project_id}'
        raise RecordNotFoundError(msg)
    if make_current:
        await session.execute(
            update(HoldoutFreeze)
            .where(HoldoutFreeze.project_id == project_id, HoldoutFreeze.is_current)
            .values(is_current=False)
        )
    row = HoldoutFreeze(
        project_id=project_id,
        is_current=make_current,
        source=source,
        item_count=item_count,
        seed=seed,
        snapshot_storage_key=snapshot_storage_key,
        frozen_by=frozen_by,
    )
    session.add(row)
    await session.flush()
    await session.refresh(row)
    return _freeze_row(row)


async def current_holdout_freeze(
    session: AsyncSession, project_id: uuid.UUID
) -> HoldoutFreezeRow | None:
    row = (
        await session.execute(
            select(HoldoutFreeze).where(
                HoldoutFreeze.project_id == project_id, HoldoutFreeze.is_current
            )
        )
    ).scalar_one_or_none()
    return None if row is None else _freeze_row(row)


async def list_holdout_freezes(
    session: AsyncSession, project_id: uuid.UUID, *, limit: int = 50
) -> list[HoldoutFreezeRow]:
    """The history, newest first (ids are time-ordered uuidv7)."""
    rows = (
        (
            await session.execute(
                select(HoldoutFreeze)
                .where(HoldoutFreeze.project_id == project_id)
                .order_by(HoldoutFreeze.id.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [_freeze_row(r) for r in rows]


async def forget_holdout_freeze(
    session: AsyncSession, project_id: uuid.UUID, freeze_id: uuid.UUID
) -> bool:
    """Delete a freeze that is not current (an undone import's); the current one is refused."""
    result = await session.execute(
        delete(HoldoutFreeze).where(
            HoldoutFreeze.project_id == project_id,
            HoldoutFreeze.id == freeze_id,
            HoldoutFreeze.is_current.is_(False),
        )
    )
    return bool(result.rowcount)  # type: ignore[attr-defined]


async def load_holdout_freezes(
    session: AsyncSession, project_id: uuid.UUID, rows: Sequence[HoldoutFreezeLoad]
) -> None:
    """The 0.6.0 upgrade's write: ids and times kept, an existing id or a second current raises."""
    if sum(1 for r in rows if r.is_current) > 1:
        msg = 'a project has at most one current holdout freeze'
        raise ValueError(msg)
    session.add_all(
        HoldoutFreeze(
            id=r.id,
            project_id=project_id,
            is_current=r.is_current,
            source=r.source,
            item_count=r.item_count,
            seed=r.seed,
            snapshot_storage_key=r.snapshot_storage_key,
            frozen_at=r.frozen_at,
            frozen_by=r.frozen_by,
        )
        for r in rows
    )
    await session.flush()


# --- label operations -------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelOperationRow:
    id: uuid.UUID
    project_id: uuid.UUID
    kind: str
    status: str
    params: Any
    member_count: int
    job_id: uuid.UUID | None
    created_at: dt.datetime
    created_by: str
    undone_at: dt.datetime | None


@dataclass(frozen=True)
class LabelOperationMemberLoad:
    """An item an operation wrote and the fields it replaced (``prior``), for undo."""

    item_id: str
    prior: Any


@dataclass(frozen=True)
class LabelOperationLoad:
    id: uuid.UUID
    project_id: uuid.UUID
    kind: str
    status: str
    params: Any
    created_at: dt.datetime
    created_by: str
    member_count: int = 0
    job_id: uuid.UUID | None = None
    undone_at: dt.datetime | None = None


def _check_kind(kind: str) -> None:
    if kind not in LABEL_OPERATION_KINDS:
        msg = f'unknown label operation kind {kind!r}'
        raise ValueError(msg)


def _check_status(status: str) -> None:
    if status not in LABEL_OPERATION_STATUSES:
        msg = f'unknown label operation status {status!r}'
        raise ValueError(msg)


def _op_row(row: LabelOperation) -> LabelOperationRow:
    return LabelOperationRow(
        id=row.id,
        project_id=row.project_id,
        kind=row.kind,
        status=row.status,
        params=row.params,
        member_count=row.member_count,
        job_id=row.job_id,
        created_at=row.created_at,
        created_by=row.created_by,
        undone_at=row.undone_at,
    )


async def new_label_operation_id(session: AsyncSession) -> uuid.UUID:
    """A time-ordered id, taken before the row exists because the items carry it as their tag."""
    return (await session.execute(select(func.uuidv7(type_=PG_UUID(as_uuid=True))))).scalar_one()


async def create_label_operation(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    kind: str,
    created_by: str,
    params: Any,
    op_id: uuid.UUID | None = None,
    job_id: uuid.UUID | None = None,
) -> LabelOperationRow:
    """The record, status ``writing``, written before any item is."""
    _check_kind(kind)
    row = LabelOperation(
        project_id=project_id,
        kind=kind,
        status='writing',
        params=params,
        job_id=job_id,
        created_by=created_by,
    )
    if op_id is not None:
        row.id = op_id
    session.add(row)
    await session.flush()
    await session.refresh(row)
    return _op_row(row)


async def get_label_operation(
    session: AsyncSession, project_id: uuid.UUID, op_id: uuid.UUID
) -> LabelOperationRow | None:
    row = (
        await session.execute(
            select(LabelOperation).where(
                LabelOperation.project_id == project_id, LabelOperation.id == op_id
            )
        )
    ).scalar_one_or_none()
    return None if row is None else _op_row(row)


async def list_label_operations(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    limit: int,
    after: uuid.UUID | None = None,
) -> list[LabelOperationRow]:
    """Newest first, older than ``after`` (exclusive)."""
    query = select(LabelOperation).where(LabelOperation.project_id == project_id)
    if after is not None:
        query = query.where(LabelOperation.id < after)
    rows = (await session.execute(query.order_by(LabelOperation.id.desc()).limit(limit))).scalars()
    return [_op_row(r) for r in rows]


async def save_label_operation(
    session: AsyncSession,
    project_id: uuid.UUID,
    op_id: uuid.UUID,
    *,
    status: str,
    params: Any,
    undone_at: dt.datetime | None = None,
) -> None:
    """Replace status and params (the record is one writer's at a time: the operation's lease)."""
    _check_status(status)
    values: dict[str, Any] = {'status': status, 'params': params}
    if undone_at is not None:
        values['undone_at'] = undone_at
    result = await session.execute(
        update(LabelOperation)
        .where(LabelOperation.project_id == project_id, LabelOperation.id == op_id)
        .values(**values)
    )
    if result.rowcount != 1:  # type: ignore[attr-defined]
        msg = f'no label operation {op_id}'
        raise RecordNotFoundError(msg)


def _batches[T](items: Iterable[T], size: int) -> Iterator[list[T]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


async def add_label_operation_members(
    session: AsyncSession, op_id: uuid.UUID, members: Iterable[LabelOperationMemberLoad]
) -> int:
    """Record members in batches of :data:`MEMBER_BATCH`; a repeated item id keeps the latest prior.

    Idempotent, so a resumed job can write the same page again. Returns the operation's member
    count after the write and stores it on the operation.
    """
    for batch in _batches(members, MEMBER_BATCH):
        statement = pg_insert(LabelOperationMember).values(
            [{'op_id': op_id, 'item_id': m.item_id, 'prior': m.prior} for m in batch]
        )
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=['op_id', 'item_id'], set_={'prior': statement.excluded.prior}
            )
        )
    return await _recount_members(session, op_id)


async def _recount_members(session: AsyncSession, op_id: uuid.UUID) -> int:
    count = (
        await session.execute(
            select(func.count())
            .select_from(LabelOperationMember)
            .where(LabelOperationMember.op_id == op_id)
        )
    ).scalar_one()
    await session.execute(
        update(LabelOperation).where(LabelOperation.id == op_id).values(member_count=count)
    )
    return count


async def label_operation_members(
    session: AsyncSession, op_id: uuid.UUID, *, after: str | None = None, limit: int = MEMBER_BATCH
) -> list[LabelOperationMemberLoad]:
    """One page of members in item-id order; page with ``after`` set to the last item id."""
    query = select(LabelOperationMember).where(LabelOperationMember.op_id == op_id)
    if after is not None:
        query = query.where(LabelOperationMember.item_id > after)
    rows = (
        await session.execute(query.order_by(LabelOperationMember.item_id).limit(limit))
    ).scalars()
    return [LabelOperationMemberLoad(item_id=r.item_id, prior=r.prior) for r in rows]


async def prune_label_operations(session: AsyncSession, project_id: uuid.UUID, *, keep: int) -> int:
    """Delete the oldest operations beyond ``keep``; one writing or undoing is never deleted."""
    ids = list(
        (
            await session.execute(
                select(LabelOperation.id)
                .where(LabelOperation.project_id == project_id)
                .order_by(LabelOperation.id.desc())
                .offset(keep)
            )
        ).scalars()
    )
    if not ids:
        return 0
    result = await session.execute(
        delete(LabelOperation).where(
            LabelOperation.id.in_(ids), LabelOperation.status.notin_(('writing', 'undoing'))
        )
    )
    return result.rowcount  # type: ignore[attr-defined,no-any-return]


async def load_label_operation(
    session: AsyncSession, op: LabelOperationLoad, members: Iterable[LabelOperationMemberLoad]
) -> None:
    """The 0.6.0 upgrade's write: id and times kept, members in batches of 5,000 (trap T-14).

    ``member_count`` is recomputed from the rows written, so it always matches the table. An
    existing operation or member raises.
    """
    _check_kind(op.kind)
    _check_status(op.status)
    session.add(
        LabelOperation(
            id=op.id,
            project_id=op.project_id,
            kind=op.kind,
            status=op.status,
            params=op.params,
            member_count=0,
            job_id=op.job_id,
            created_at=op.created_at,
            created_by=op.created_by,
            undone_at=op.undone_at,
        )
    )
    await session.flush()
    for batch in _batches(members, MEMBER_BATCH):
        await session.execute(
            pg_insert(LabelOperationMember).values(
                [{'op_id': op.id, 'item_id': m.item_id, 'prior': m.prior} for m in batch]
            )
        )
    await _recount_members(session, op.id)


# --- project state ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectStateRow:
    key: str
    value: Any
    revision: int
    updated_at: dt.datetime


def _check_key(key: str) -> None:
    if key not in PROJECT_STATE_KEYS:
        msg = f'unknown project state key {key!r}'
        raise ValueError(msg)


async def get_project_state(
    session: AsyncSession, project_id: uuid.UUID, key: str
) -> ProjectStateRow | None:
    _check_key(key)
    row = (
        await session.execute(
            select(ProjectState).where(
                ProjectState.project_id == project_id, ProjectState.key == key
            )
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    return ProjectStateRow(
        key=row.key, value=row.value, revision=row.revision, updated_at=row.updated_at
    )


async def put_project_state(
    session: AsyncSession,
    project_id: uuid.UUID,
    key: str,
    value: Any,
    *,
    expected_revision: int,
) -> int:
    """Write the marker if its revision is ``expected_revision`` (0: it must not exist yet).

    Returns the new revision. A stale revision raises :class:`RecordRevisionConflict`; the check
    and the write are one statement, so two writers cannot both win.
    """
    _check_key(key)
    if expected_revision == 0:
        inserted = await session.execute(
            pg_insert(ProjectState)
            .values(project_id=project_id, key=key, value=value, revision=1)
            .on_conflict_do_nothing(index_elements=['project_id', 'key'])
            .returning(ProjectState.revision)
        )
        if inserted.scalar_one_or_none() is None:
            msg = f'{key} already exists'
            raise RecordRevisionConflict(msg)
        return 1
    updated = await session.execute(
        update(ProjectState)
        .where(
            ProjectState.project_id == project_id,
            ProjectState.key == key,
            ProjectState.revision == expected_revision,
        )
        .values(value=value, revision=ProjectState.revision + 1, updated_at=func.now())
        .returning(ProjectState.revision)
    )
    revision = updated.scalar_one_or_none()
    if revision is None:
        msg = f'{key} is not at revision {expected_revision}'
        raise RecordRevisionConflict(msg)
    return revision


async def modify_project_state(
    session: AsyncSession,
    project_id: uuid.UUID,
    key: str,
    change: Callable[[Any | None], Any],
) -> Any:
    """Read, change and write a marker, retrying when another writer got in between.

    ``change`` receives the current value (``None`` when absent) and returns the new one; it runs
    again on a retry, so it must be pure. Each attempt is its own statement pair under the
    caller's transaction (READ COMMITTED), so a retry sees the other writer's commit.
    """
    for _ in range(_PROJECT_STATE_RETRIES):
        current = await get_project_state(session, project_id, key)
        value = change(None if current is None else current.value)
        try:
            await put_project_state(
                session,
                project_id,
                key,
                value,
                expected_revision=0 if current is None else current.revision,
            )
        except RecordRevisionConflict:
            continue
        return value
    msg = f'{key} kept changing; gave up after {_PROJECT_STATE_RETRIES} attempts'
    raise RecordRevisionConflict(msg)


async def load_project_state(
    session: AsyncSession, project_id: uuid.UUID, key: str, body: Any, *, revision: int
) -> None:
    """The 0.6.0 upgrade's write: the marker at the given revision; an existing one raises."""
    _check_key(key)
    if revision < 1:
        msg = 'revision starts at 1'
        raise ValueError(msg)
    session.add(ProjectState(project_id=project_id, key=key, value=body, revision=revision))
    await session.flush()


__all__ = [
    'LABEL_OPERATION_KINDS',
    'MEMBER_BATCH',
    'HoldoutFreezeLoad',
    'HoldoutFreezeRow',
    'LabelOperationLoad',
    'LabelOperationMemberLoad',
    'LabelOperationRow',
    'ProjectStateRow',
    'RecordNotFoundError',
    'RecordRevisionConflict',
    'add_label_operation_members',
    'create_label_operation',
    'current_holdout_freeze',
    'forget_holdout_freeze',
    'freeze_holdout',
    'get_label_operation',
    'get_project_state',
    'label_operation_members',
    'list_holdout_freezes',
    'list_label_operations',
    'load_holdout_freezes',
    'load_label_operation',
    'load_project_state',
    'modify_project_state',
    'new_label_operation_id',
    'prune_label_operations',
    'put_project_state',
    'save_label_operation',
]
