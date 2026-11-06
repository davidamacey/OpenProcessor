"""The assist loop's propagation and review are label operations: record, members, undo state."""

from __future__ import annotations

import pytest
from _seed import make_project

from src.db.repos.records import (
    LABEL_OPERATION_KINDS,
    LabelOperationMemberLoad,
    add_label_operation_members,
    create_label_operation,
    get_label_operation,
    label_operation_members,
    save_label_operation,
)
from src.db.session import async_session_factory, project_tx


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['assist_loop_propagate', 'assist_loop_review'])
async def test_an_op_of_each_loop_kind_records_members_and_undoes_them(pg_env, kind) -> None:
    project = make_project(pg_env, f'kinds-{kind.rsplit("_", 1)[1]}')
    members = [
        LabelOperationMemberLoad(item_id=f'crop-{i}', prior={'class_name': 'unknown'})
        for i in range(3)
    ]
    async with async_session_factory()() as session, project_tx(session, project):
        op = await create_label_operation(
            session, project, kind=kind, created_by='loop', params={'node': 7}
        )
        await add_label_operation_members(session, op.id, members)
        await save_label_operation(session, project, op.id, status='written', params=op.params)
    async with async_session_factory()() as session, project_tx(session, project):
        stored = await get_label_operation(session, project, op.id)
        assert stored is not None
        assert (stored.kind, stored.status, stored.member_count) == (kind, 'written', 3)
        priors = {m.item_id: m.prior for m in await label_operation_members(session, op.id)}
        assert priors == {m.item_id: m.prior for m in members}
        await save_label_operation(session, project, op.id, status='undoing', params=op.params)
        await save_label_operation(session, project, op.id, status='undone', params=op.params)
        undone = await get_label_operation(session, project, op.id)
    assert undone is not None
    assert undone.status == 'undone'


@pytest.mark.asyncio
async def test_the_existing_kinds_still_work_and_an_unknown_kind_is_refused(pg_env) -> None:
    project = make_project(pg_env, 'kinds-existing')
    async with async_session_factory()() as session, project_tx(session, project):
        for kind in ('label', 'move'):
            op = await create_label_operation(
                session, project, kind=kind, created_by='a', params={}
            )
            assert op.kind == kind
        with pytest.raises(ValueError, match='unknown label operation kind'):
            await create_label_operation(session, project, kind='bogus', created_by='a', params={})
    assert set(LABEL_OPERATION_KINDS) == {
        'label', 'move', 'assist_loop_propagate', 'assist_loop_review',
    }  # fmt: skip
