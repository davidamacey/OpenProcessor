"""The config document repository: OCC saves, immutable revisions, epochs and the upgrade load."""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
from _config_fixtures import make_project
from psycopg import errors

from src.db.repos import config as repo
from src.db.session import async_session_factory, deployment_tx, project_tx


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]

BODY = {'prompt': 'a'}


@pytest.fixture(scope='module')
def project(pg) -> uuid.UUID:
    return make_project(pg, 'cfg-repo')


async def _save(project_id, name, expected, body=None, kind='prompt_pack'):
    async with async_session_factory()() as session, project_tx(session, project_id):
        return await repo.save_config(
            session,
            project_id=project_id,
            kind=kind,
            name=name,
            body=body or BODY,
            expected_revision=expected,
        )


@pytest.mark.asyncio
async def test_two_saves_with_one_expected_revision_one_wins(pg_env, project) -> None:
    await _save(project, 'race', None)
    results = await asyncio.gather(
        _save(project, 'race', 1, {'prompt': 'x'}),
        _save(project, 'race', 1, {'prompt': 'y'}),
        return_exceptions=True,
    )
    winners = [r for r in results if not isinstance(r, Exception)]
    losers = [r for r in results if isinstance(r, repo.RevisionConflictError)]
    assert len(winners) == 1
    assert len(losers) == 1
    assert losers[0].current_revision == 2
    async with async_session_factory()() as session, project_tx(session, project):
        doc = await repo.get_config(session, project_id=project, kind='prompt_pack', name='race')
    assert doc is not None
    assert doc.revision == 2
    assert doc.body == winners[0].body


@pytest.mark.asyncio
async def test_creating_an_existing_name_is_a_conflict(pg_env, project) -> None:
    await _save(project, 'dup', None)
    with pytest.raises(repo.RevisionConflictError) as exc:
        await _save(project, 'dup', None)
    assert exc.value.current_revision == 1


@pytest.mark.asyncio
async def test_a_stale_revision_on_a_missing_document_reports_zero(pg_env, project) -> None:
    with pytest.raises(repo.RevisionConflictError) as exc:
        await _save(project, 'never-created', 3)
    assert exc.value.current_revision == 0


@pytest.mark.asyncio
async def test_a_revision_row_cannot_be_updated(pg, pg_env, project) -> None:
    doc = await _save(project, 'immutable', None)
    with psycopg.connect(pg.conninfo('app')) as conn:
        conn.execute(
            "SELECT set_config('op.scope', 'project', true), set_config('op.project_id', %s, true)",
            (str(project),),
        )
        with pytest.raises(errors.IntegrityConstraintViolation, match='config_revision_immutable'):
            conn.execute("UPDATE config_revisions SET body = '{}' WHERE config_id = %s", (doc.id,))


@pytest.mark.asyncio
async def test_revisions_are_kept_and_numbers_are_never_reused(pg_env, project) -> None:
    await _save(project, 'history', None, {'v': 1})
    await _save(project, 'history', 1, {'v': 2})
    await _save(project, 'history', 2, {'v': 3})
    async with async_session_factory()() as session, project_tx(session, project):
        first = await repo.get_config(
            session, project_id=project, kind='prompt_pack', name='history', revision=1
        )
        current = await repo.get_config(
            session, project_id=project, kind='prompt_pack', name='history'
        )
        assert first is not None
        assert current is not None
        revisions = await repo.list_revisions(session, current.id)
    assert first.body == {'v': 1}
    assert current.body == {'v': 3}
    assert [r.revision for r in revisions] == [1, 2, 3]
    with pytest.raises(repo.RevisionConflictError):
        await _save(project, 'history', 1)


@pytest.mark.asyncio
async def test_every_write_bumps_the_project_epoch(pg_env, project) -> None:
    async with async_session_factory()() as session, project_tx(session, project):
        before = await repo.get_epoch(session, project)
    await _save(project, 'epoch', None)
    async with async_session_factory()() as session, project_tx(session, project):
        after = await repo.get_epoch(session, project)
    assert after == before + 1


@pytest.mark.asyncio
async def test_deleting_removes_the_document_and_needs_the_current_revision(
    pg_env, project
) -> None:
    await _save(project, 'gone', None)
    async with async_session_factory()() as session, project_tx(session, project):
        with pytest.raises(repo.RevisionConflictError):
            await repo.delete_config(
                session, project_id=project, kind='prompt_pack', name='gone', expected_revision=9
            )
    async with async_session_factory()() as session, project_tx(session, project):
        await repo.delete_config(
            session, project_id=project, kind='prompt_pack', name='gone', expected_revision=1
        )
    async with async_session_factory()() as session, project_tx(session, project):
        assert (
            await repo.get_config(session, project_id=project, kind='prompt_pack', name='gone')
            is None
        )
        with pytest.raises(KeyError):
            await repo.delete_config(
                session, project_id=project, kind='prompt_pack', name='gone', expected_revision=1
            )


@pytest.mark.asyncio
async def test_a_deployment_kind_with_a_project_is_refused(pg_env, project) -> None:
    with pytest.raises(repo.ConfigScopeError):
        await _save(project, 'ep', None, kind='vlm_endpoint')


@pytest.mark.asyncio
async def test_a_deployment_document_round_trips_in_a_deployment_transaction(pg_env) -> None:
    async with async_session_factory()() as session, deployment_tx(session, 'config repo test'):
        before = await repo.get_epoch(session, None)
        await repo.save_config(
            session,
            project_id=None,
            kind='vlm_endpoint',
            name='local',
            body={'url': 'http://vlm'},
            expected_revision=None,
        )
    async with async_session_factory()() as session, deployment_tx(session, 'config repo test'):
        doc = await repo.get_config(session, project_id=None, kind='vlm_endpoint', name='local')
        assert await repo.get_epoch(session, None) == before + 1
    assert doc is not None
    assert doc.body == {'url': 'http://vlm'}


@pytest.mark.asyncio
async def test_load_preserves_ids_revisions_and_timestamps_then_reads_back(pg_env, project) -> None:
    config_id = uuid.uuid4()
    t1 = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    t2 = dt.datetime(2026, 2, 1, tzinfo=dt.UTC)
    load = repo.ConfigDocumentLoad(
        project_id=project,
        kind='region_profile',
        name='loaded',
        id=config_id,
        revisions=[
            repo.ConfigRevisionLoad(1, {'a': 1}, t1, description='first'),
            repo.ConfigRevisionLoad(2, {'a': 2}, t2, description='from 0.6.0'),
        ],
    )
    async with async_session_factory()() as session, deployment_tx(session, 'load test'):
        assert await repo.load_config_document(session, load) == config_id
    async with async_session_factory()() as session, project_tx(session, project):
        current = await repo.get_config(
            session, project_id=project, kind='region_profile', name='loaded'
        )
        first = await repo.get_config(
            session, project_id=project, kind='region_profile', name='loaded', revision=1
        )
    assert current is not None
    assert first is not None
    assert (current.id, current.revision, current.body, current.description) == (
        config_id,
        2,
        {'a': 2},
        'from 0.6.0',
    )
    assert (first.body, first.description) == ({'a': 1}, 'first')
    assert current.created_at == t1
    assert current.updated_at == t2
    async with async_session_factory()() as session, deployment_tx(session, 'load test'):
        with pytest.raises(Exception):  # noqa: B017, PT011  # the unique constraint, whichever driver class wraps it
            await repo.load_config_document(session, load)


def test_the_digest_is_over_canonical_json() -> None:
    assert repo.body_sha256({'b': 1, 'a': 2}) == repo.body_sha256({'a': 2, 'b': 1})
    assert repo.body_sha256({'a': 1}) != repo.body_sha256({'a': 2})


@pytest.mark.asyncio
async def test_each_revision_keeps_its_own_description_and_saved_by(pg_env, project) -> None:
    async with async_session_factory()() as session, project_tx(session, project):
        first = await repo.save_config(
            session,
            project_id=project,
            kind='prompt_pack',
            name='described',
            body=BODY,
            expected_revision=None,
            description='one',
        )
        await repo.save_config(
            session,
            project_id=project,
            kind='prompt_pack',
            name='described',
            body={'prompt': 'b'},
            expected_revision=1,
            description='two',
            by='alice',
        )
        old = await repo.get_config(
            session, project_id=project, kind='prompt_pack', name='described', revision=1
        )
        now = await repo.get_config(
            session, project_id=project, kind='prompt_pack', name='described'
        )
        listed = await repo.list_revisions(session, first.id)
    assert old is not None
    assert now is not None
    assert (old.description, now.description) == ('one', 'two')
    assert [(r.revision, r.description, r.created_by) for r in listed] == [
        (1, 'one', 'system'),
        (2, 'two', 'alice'),
    ]


@pytest.mark.asyncio
async def test_a_clone_tag_is_derived_from_the_joined_source(pg_env, pg, project) -> None:
    other = make_project(pg, f'cfg-clone-{uuid.uuid4().hex[:8]}')
    async with async_session_factory()() as session, deployment_tx(session, 'clone tag test'):
        source = await repo.save_config(
            session,
            project_id=project,
            kind='region_profile',
            name='src',
            body=BODY,
            expected_revision=None,
        )
        clone = await repo.save_config(
            session,
            project_id=other,
            kind='region_profile',
            name='copy',
            body=BODY,
            expected_revision=None,
            cloned_from_id=source.id,
            cloned_from_revision=1,
        )
        plain = await repo.get_config(session, project_id=other, kind='region_profile', name='copy')
        endpoint = await repo.save_config(
            session,
            project_id=None,
            kind='vlm_endpoint',
            name=f'ep-{uuid.uuid4().hex[:6]}',
            body={'u': 1},
            expected_revision=None,
        )
        ep_clone = await repo.save_config(
            session,
            project_id=None,
            kind='vlm_endpoint',
            name=f'epc-{uuid.uuid4().hex[:6]}',
            body={'u': 1},
            expected_revision=None,
            cloned_from_id=endpoint.id,
            cloned_from_revision=1,
        )
    assert plain is not None
    assert clone.cloned_from == plain.cloned_from == f'{pg_slug(project, pg)}:src@1'
    assert ep_clone.cloned_from == f'{endpoint.name}@1'
    assert source.cloned_from is None


def pg_slug(project_id: uuid.UUID, pg) -> str:
    with psycopg.connect(pg.conninfo('super')) as conn:
        row = conn.execute('SELECT slug FROM projects WHERE id = %s', (project_id,)).fetchone()
    assert row is not None
    return str(row[0])


@pytest.mark.asyncio
async def test_ensure_config_materializes_once_and_versions_on_a_changed_body(
    pg_env, project
) -> None:
    async with async_session_factory()() as session, project_tx(session, project):
        a = await repo.ensure_config(
            session,
            project_id=project,
            kind='prompt_pack',
            name='builtin-x',
            body=BODY,
            description='Built-in example',
        )
        again = await repo.ensure_config(
            session,
            project_id=project,
            kind='prompt_pack',
            name='builtin-x',
            body=BODY,
            description='Built-in example',
        )
        changed = await repo.ensure_config(
            session,
            project_id=project,
            kind='prompt_pack',
            name='builtin-x',
            body={'prompt': 'new'},
            description='Built-in example',
        )
        revisions = await repo.list_revisions(session, a.id)
    assert (a.revision, again.revision, changed.revision) == (1, 1, 2)
    assert [(r.created_by, r.description) for r in revisions] == [
        ('system', 'Built-in example')
    ] * 2


async def _unused(_: AsyncSession) -> None:  # keeps the import used for typing in editors
    return None
