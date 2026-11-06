"""Label-operation records and members in Postgres: batches of 5,000, undo reads the prior state."""

from __future__ import annotations

import datetime as dt
import re
import uuid

import pytest
from _seed import make_project
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError

from src.db.repos.records import (
    MEMBER_BATCH,
    LabelOperationLoad,
    LabelOperationMemberLoad,
    RecordNotFoundError,
    add_label_operation_members,
    create_label_operation,
    get_label_operation,
    label_operation_members,
    list_label_operations,
    load_label_operation,
    new_label_operation_id,
    prune_label_operations,
    save_label_operation,
)
from src.db.session import async_session_factory, deployment_tx, project_tx


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]

_MEMBER_INSERT = re.compile(r'INSERT INTO label_operation_members', re.IGNORECASE)


def _members(n: int) -> list[LabelOperationMemberLoad]:
    return [
        LabelOperationMemberLoad(item_id=f'crop-{i:06d}', prior={'class_name': f'c{i % 3}'})
        for i in range(n)
    ]


class _Inserts:
    """Counts the INSERT statements into the member table (one per batch)."""

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def __call__(self, _conn, _cursor, statement, parameters, _context, _many) -> None:
        if _MEMBER_INSERT.search(statement):
            self.sizes.append(len(parameters) // 3)


@pytest.fixture
def inserts(bound_async_engine):
    counter = _Inserts()
    sync_engine = bound_async_engine.sync_engine
    event.listen(sync_engine, 'before_cursor_execute', counter)
    yield counter
    event.remove(sync_engine, 'before_cursor_execute', counter)


async def _create(project: uuid.UUID, *, kind: str = 'label', status: str | None = None):
    async with async_session_factory()() as session, project_tx(session, project):
        op = await create_label_operation(
            session, project, kind=kind, created_by='alice', params={'class_name': 'car'}
        )
        if status is not None:
            await save_label_operation(session, project, op.id, status=status, params=op.params)
        return op


@pytest.mark.asyncio
async def test_a_ten_thousand_member_op_writes_in_batches_of_five_thousand(pg_env, inserts) -> None:
    project = make_project(pg_env, 'ops-batches')
    op = await _create(project)
    inserts.sizes.clear()
    async with async_session_factory()() as session, project_tx(session, project):
        count = await add_label_operation_members(session, op.id, iter(_members(10_000)))
    assert count == 10_000
    assert inserts.sizes == [MEMBER_BATCH, MEMBER_BATCH]
    async with async_session_factory()() as session, project_tx(session, project):
        row = await get_label_operation(session, project, op.id)
    assert row is not None
    assert row.member_count == 10_000


@pytest.mark.asyncio
async def test_undo_reads_the_prior_state_of_each_member_from_the_table(pg_env) -> None:
    project = make_project(pg_env, 'ops-prior')
    op = await _create(project)
    async with async_session_factory()() as session, project_tx(session, project):
        await add_label_operation_members(session, op.id, _members(7))
    async with async_session_factory()() as session, project_tx(session, project):
        first = await label_operation_members(session, op.id, limit=5)
        rest = await label_operation_members(session, op.id, after=first[-1].item_id, limit=5)
    assert [m.item_id for m in first + rest] == [f'crop-{i:06d}' for i in range(7)]
    assert first[1].prior == {'class_name': 'c1'}
    assert len(rest) == 2


@pytest.mark.asyncio
async def test_writing_the_same_members_again_is_idempotent(pg_env) -> None:
    project = make_project(pg_env, 'ops-idem')
    op = await _create(project)
    async with async_session_factory()() as session, project_tx(session, project):
        await add_label_operation_members(session, op.id, _members(4))
        assert await add_label_operation_members(session, op.id, _members(6)) == 6


@pytest.mark.asyncio
async def test_status_and_params_round_trip_and_an_unknown_op_is_refused(pg_env) -> None:
    project = make_project(pg_env, 'ops-status')
    op = await _create(project)
    undone = dt.datetime(2026, 3, 4, tzinfo=dt.UTC)
    async with async_session_factory()() as session, project_tx(session, project):
        await save_label_operation(
            session, project, op.id, status='undone', params={'undo_passes': [1]}, undone_at=undone
        )
        row = await get_label_operation(session, project, op.id)
    assert row is not None
    assert (row.status, row.params, row.undone_at) == ('undone', {'undo_passes': [1]}, undone)
    async with async_session_factory()() as session:
        with pytest.raises(RecordNotFoundError):
            async with project_tx(session, project):
                await save_label_operation(
                    session, project, uuid.uuid4(), status='written', params={}
                )
        with pytest.raises(ValueError, match='unknown label operation status'):
            await save_label_operation(session, project, op.id, status='gone', params={})


@pytest.mark.asyncio
async def test_listing_is_newest_first_with_a_cursor_and_scoped_to_the_project(pg_env) -> None:
    project, other = make_project(pg_env, 'ops-list'), make_project(pg_env, 'ops-list-other')
    ids = [(await _create(project)).id for _ in range(5)]
    await _create(other)
    async with async_session_factory()() as session, project_tx(session, project):
        page = await list_label_operations(session, project, limit=3)
        rest = await list_label_operations(session, project, limit=3, after=page[-1].id)
    assert [o.id for o in page + rest] == list(reversed(ids))
    assert {o.project_id for o in page + rest} == {project}


@pytest.mark.asyncio
async def test_pruning_keeps_the_newest_and_never_deletes_a_writing_op(pg_env) -> None:
    project = make_project(pg_env, 'ops-prune')
    writing = await _create(project)
    done = [(await _create(project, status='written')).id for _ in range(4)]
    async with async_session_factory()() as session, project_tx(session, project):
        deleted = await prune_label_operations(session, project, keep=2)
        left = {o.id for o in await list_label_operations(session, project, limit=10)}
    assert deleted == 2
    assert left == {writing.id, *done[2:]}


@pytest.mark.asyncio
async def test_the_upgrade_load_keeps_id_and_time_writes_batches_and_reads_back(
    pg_env, inserts
) -> None:
    project = make_project(pg_env, 'ops-load')
    op_id = uuid.uuid4()
    created = dt.datetime(2026, 1, 5, 6, 7, 8, tzinfo=dt.UTC)
    op = LabelOperationLoad(
        id=op_id, project_id=project, kind='move', status='written',
        params={'legacy_op_id': 'lop_20260105T060708000000_abcdef'},
        created_at=created, created_by='bob',
    )  # fmt: skip
    inserts.sizes.clear()
    async with async_session_factory()() as session, deployment_tx(session, 'test load'):
        await load_label_operation(session, op, iter(_members(10_001)))
    assert inserts.sizes == [MEMBER_BATCH, MEMBER_BATCH, 1]
    async with async_session_factory()() as session, project_tx(session, project):
        row = await get_label_operation(session, project, op_id)
        members = await label_operation_members(session, op_id, limit=3)
    assert row is not None
    assert (row.created_at, row.member_count, row.kind) == (created, 10_001, 'move')
    assert row.params['legacy_op_id'].startswith('lop_')
    assert members[0].prior == {'class_name': 'c0'}

    async with async_session_factory()() as session:
        with pytest.raises(IntegrityError):
            async with deployment_tx(session, 'again'):
                await load_label_operation(session, op, [])


@pytest.mark.asyncio
async def test_a_preallocated_id_is_the_id_of_the_created_op(pg_env) -> None:
    project = make_project(pg_env, 'ops-preid')
    async with async_session_factory()() as session, project_tx(session, project):
        op_id = await new_label_operation_id(session)
        op = await create_label_operation(
            session, project, kind='label', created_by='a', params={}, op_id=op_id
        )
    assert op.id == op_id
