"""project_tx and deployment_tx: what they set, what they refuse, and that scope is per transaction."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.session import (
    ScopeError,
    ScopeNotAllowedError,
    async_session_factory,
    deployment_tx,
    project_tx,
    sync_project_tx,
    sync_session_factory,
)


pytestmark = pytest.mark.db

PROJECT = uuid.uuid4()
OTHER = uuid.uuid4()
_SHOW = text("SELECT current_setting('op.scope', true), current_setting('op.project_id', true)")


def _call_from(module_name: str, filename: str, session):
    """Run ``deployment_tx(session, 'probe')`` as if it were called from ``module_name``."""
    code = compile('result = deployment_tx(session, "probe")', filename, 'exec')
    namespace = {'__name__': module_name, 'deployment_tx': deployment_tx, 'session': session}
    exec(code, namespace)  # the test needs a caller frame with a chosen module name
    return namespace['result']


@pytest.mark.asyncio
async def test_project_tx_binds_the_project_for_the_transaction(bound_async_engine) -> None:
    async with async_session_factory()() as session, project_tx(session, PROJECT):
        row = (await session.execute(_SHOW)).one()
    assert tuple(row) == ('project', str(PROJECT))


@pytest.mark.asyncio
async def test_the_scope_survives_a_commit_and_the_auto_begin_after_it(bound_async_engine) -> None:
    async with async_session_factory()() as session, project_tx(session, PROJECT):
        await session.execute(text('SELECT 1'))
        await session.commit()
        row = (await session.execute(_SHOW)).one()
    assert tuple(row) == ('project', str(PROJECT))


@pytest.mark.asyncio
async def test_a_transaction_outside_the_helper_has_no_scope_on_a_reused_connection(
    pg_env, monkeypatch
) -> None:
    monkeypatch.setenv('OP_DB_API_CONNECTIONS_TOTAL', '4')
    monkeypatch.setenv('WEB_CONCURRENCY', '4')  # pool of 1: the next request reuses the connection
    factory = async_session_factory()
    async with factory() as first, project_tx(first, PROJECT):
        await first.execute(text('SELECT 1'))
    async with factory() as second:
        row = (await second.execute(_SHOW)).one()
    assert row[0] in ('', None)
    assert row[1] in ('', None)


@pytest.mark.asyncio
async def test_two_projects_in_a_row_each_see_their_own_binding(bound_async_engine) -> None:
    factory = async_session_factory()
    for project in (PROJECT, OTHER):
        async with factory() as session, project_tx(session, project):
            assert (await session.execute(_SHOW)).one()[1] == str(project)


@pytest.mark.asyncio
async def test_a_scoped_transaction_needs_a_session_with_no_transaction_open(
    bound_async_engine,
) -> None:
    async with async_session_factory()() as session:
        await session.execute(text('SELECT 1'))
        with pytest.raises(ScopeError, match='no transaction in progress'):
            async with project_tx(session, PROJECT):
                pass


@pytest.mark.asyncio
async def test_a_bad_project_id_is_refused(bound_async_engine) -> None:
    async with async_session_factory()() as session:
        with pytest.raises(ScopeError, match='not a project id'):
            project_tx(session, 'not-a-uuid')


@pytest.mark.asyncio
async def test_deployment_tx_from_a_module_outside_the_allowlist_raises(bound_async_engine) -> None:
    async with async_session_factory()() as session:
        with pytest.raises(ScopeNotAllowedError, match=r'src\.services\.rogue'):
            _call_from('src.services.rogue', '/srv/app/src/services/rogue.py', session)


@pytest.mark.asyncio
async def test_deployment_tx_from_a_listed_module_opens_the_deployment_scope(
    bound_async_engine,
) -> None:
    async with (
        async_session_factory()() as session,
        _call_from('src.tasks.reaper', '/srv/app/src/tasks/reaper.py', session),
    ):
        row = (await session.execute(_SHOW)).one()
    assert tuple(row) == ('deployment', '')


@pytest.mark.asyncio
async def test_deployment_tx_needs_a_reason(bound_async_engine) -> None:
    async with async_session_factory()() as session:
        with pytest.raises(ScopeError, match='reason'):
            deployment_tx(session, '  ')


def test_the_sync_helper_scopes_the_same_way(pg_env) -> None:
    with sync_session_factory()() as session, sync_project_tx(session, PROJECT):
        assert tuple(session.execute(_SHOW).one()) == ('project', str(PROJECT))
    assert isinstance(session, Session)


@pytest.mark.asyncio
async def test_a_clean_exit_commits_and_an_error_rolls_back(bound_async_engine) -> None:
    factory = async_session_factory()
    async with factory() as session, deployment_tx(session, 'test'):
        await session.execute(text("INSERT INTO deployment_counters (name) VALUES ('kept')"))

    async def write_then_fail() -> None:
        async with factory() as session, deployment_tx(session, 'test'):
            await session.execute(text("INSERT INTO deployment_counters (name) VALUES ('lost')"))
            raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await write_then_fail()
    async with factory() as session:
        names = (
            (await session.execute(text('SELECT name FROM deployment_counters'))).scalars().all()
        )
    assert 'kept' in names
    assert 'lost' not in names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('module', 'filename'),
    [
        ('src.tasks.runtime.gpu_app', '/srv/app/src/tasks/runtime/gpu_app.py'),
        ('src.tasks.runtime.job_task', '/srv/app/src/tasks/runtime/job_task.py'),
        ('src.services.projects.delete', '/srv/app/src/services/projects/delete.py'),
    ],
)
async def test_the_modules_the_plan_names_may_open_the_deployment_scope(
    bound_async_engine, module: str, filename: str
) -> None:
    async with async_session_factory()() as session, _call_from(module, filename, session):
        await session.execute(text('SELECT 1'))
