"""The projects repository against a real PostgreSQL: constraints, races, cascades, load."""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
from _rls_seed import ids, statements
from sqlalchemy import text

from src.db.repos import projects as repo
from src.db.session import async_session_factory, deployment_tx


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = [pytest.mark.db, pytest.mark.asyncio, pytest.mark.usefixtures('bound_async_engine')]


def new(slug: str, **kw: Any) -> repo.NewProject:
    return repo.NewProject(
        slug=slug,
        display_name=slug.title(),
        index_prefix='op_prj_',
        storage_prefix=f'projects/{slug}',
        model_prefix=f'{slug}__',
        mlflow_experiment=f'openprocessor-{slug}',
        indexes={'images': f'op_prj_{slug}__images', 'items': f'op_prj_{slug}__items'},
        created_by='api',
        **kw,
    )


async def create(slug: str, **kw: Any) -> repo.ProjectRow:
    async with async_session_factory()() as session, deployment_tx(session, 'test create'):
        return await repo.create_project(session, new(slug, **kw))


async def read(fn: Any, *args: Any, **kw: Any) -> Any:
    async with async_session_factory()() as session, deployment_tx(session, 'test read'):
        return await fn(session, *args, **kw)


async def read_sync(fn: Any) -> Any:
    async with async_session_factory()() as session, deployment_tx(session, 'test read'):
        return await session.run_sync(fn)


async def write(fn: Any, *args: Any, **kw: Any) -> Any:
    async with async_session_factory()() as session, deployment_tx(session, 'test write'):
        return await fn(session, *args, **kw)


async def test_create_then_get_and_list_return_the_project(repo_env: None) -> None:
    created = await create('alpha', description='first')
    assert created.status == 'building'
    assert created.revision == 1
    assert created.indexes == {'images': 'op_prj_alpha__images', 'items': 'op_prj_alpha__items'}
    got = await read(repo.get_by_slug, 'alpha')
    assert got == created
    assert await read(repo.get_by_id, created.id) == created
    assert 'alpha' in [p.slug for p in await read_sync(repo.list_projects_sync)]
    assert await read(repo.get_by_slug, 'nope') is None


async def test_create_writes_the_satellite_rows_and_bumps_the_counter(
    repo_env: None,
) -> None:
    before = await read_sync(repo.read_projects_revision_sync)
    created = await create('satellites')
    assert await read_sync(repo.read_projects_revision_sync) == before + 1
    async with async_session_factory()() as session, deployment_tx(session, 'test'):
        counts = (
            await session.execute(
                text(
                    'SELECT (SELECT count(*) FROM class_registries WHERE project_id = :p),'
                    ' (SELECT count(*) FROM project_settings WHERE project_id = :p),'
                    ' (SELECT count(*) FROM project_keymaps WHERE project_id = :p),'
                    ' (SELECT count(*) FROM project_indexes WHERE project_id = :p)'
                ),
                {'p': created.id},
            )
        ).one()
    assert tuple(counts) == (1, 1, 1, 2)


async def test_a_duplicate_slug_is_slug_taken(repo_env: None) -> None:
    await create('dupe')
    with pytest.raises(repo.SlugTakenError):
        await create('dupe')


async def test_a_retired_slug_is_slug_retired_and_never_reused(repo_env: None) -> None:
    first = await create('gone', status='active')
    await write(repo.mark_deleting, 'gone', from_states=('active',))
    await write(repo.finish_delete, 'gone')
    assert await read(repo.get_by_slug, 'gone') is None
    assert await read_sync(repo.retired_slugs_sync) == ['gone']
    with pytest.raises(repo.SlugRetiredError):
        await create('gone')
    # the identity key is never reissued: a different slug gets a later key
    second = await create('gone-two')
    assert second.lock_key > first.lock_key


async def test_ten_concurrent_creates_of_one_slug_leave_one_row(repo_env: None) -> None:
    async def attempt() -> object:
        try:
            return await create('race')
        except repo.ProjectRepoError as exc:
            return exc

    results = await asyncio.gather(*(attempt() for _ in range(10)))
    assert sum(isinstance(r, repo.ProjectRow) for r in results) == 1
    assert all(
        isinstance(r, repo.SlugTakenError) for r in results if not isinstance(r, repo.ProjectRow)
    )
    assert [p.slug for p in await read_sync(repo.list_projects_sync) if p.slug == 'race'] == [
        'race'
    ]


async def test_a_stale_revision_is_a_conflict(repo_env: None) -> None:
    created = await create('occ', status='active')
    patched = await write(
        repo.patch_project, 'occ', expected_revision=created.revision, display_name='Renamed'
    )
    assert (patched.display_name, patched.revision) == ('Renamed', created.revision + 1)
    with pytest.raises(repo.RevisionConflictError) as caught:
        await write(repo.patch_project, 'occ', expected_revision=created.revision, description='x')
    assert caught.value.current == patched.revision
    with pytest.raises(repo.ProjectNotFoundError):
        await write(repo.patch_project, 'missing', expected_revision=1)


async def test_archive_and_unarchive_allow_only_their_source_states(repo_env: None) -> None:
    await create('arch', status='active')
    archived = await write(
        repo.set_status, 'arch', to='archived', from_states=('active',), expected_revision=1
    )
    assert (archived.status, archived.revision) == ('archived', 2)
    with pytest.raises(repo.InvalidTransitionError) as caught:
        await write(repo.set_status, 'arch', to='archived', from_states=('active',))
    assert caught.value.status == 'archived'
    back = await write(repo.set_status, 'arch', to='active', from_states=('archived',))
    assert back.status == 'active'


async def test_mark_deleting_remembers_the_prior_status_and_rollback_restores_it(
    repo_env: None,
) -> None:
    await create('roll', status='active')
    await write(repo.set_status, 'roll', to='archived', from_states=('active',))
    deleting = await write(repo.mark_deleting, 'roll', from_states=('active', 'archived', 'failed'))
    assert (deleting.status, deleting.pre_delete_status) == ('deleting', 'archived')
    with pytest.raises(repo.InvalidTransitionError):
        await write(repo.mark_deleting, 'roll', from_states=('active', 'archived', 'failed'))
    restored = await write(repo.rollback_delete, 'roll')
    assert (restored.status, restored.pre_delete_status) == ('archived', None)


async def test_two_finishers_of_one_delete_cannot_both_claim(repo_env: None) -> None:
    await create('claim', status='active')
    deleting = await write(repo.mark_deleting, 'claim', from_states=('active',))
    claimed = await write(repo.claim_delete, 'claim', expected_revision=deleting.revision)
    assert claimed.revision == deleting.revision + 1
    with pytest.raises(repo.RevisionConflictError):
        await write(repo.claim_delete, 'claim', expected_revision=deleting.revision)


async def test_finish_requires_deleting(repo_env: None) -> None:
    await create('live', status='active')
    with pytest.raises(repo.InvalidTransitionError):
        await write(repo.finish_delete, 'live')
    with pytest.raises(repo.ProjectNotFoundError):
        await write(repo.finish_delete, 'never-was')


async def test_pause_and_resume_write_paused_at(repo_env: None) -> None:
    await create('pausable', status='active')
    paused = await write(repo.set_paused, 'pausable', paused=True, by='api')
    assert paused.paused_at is not None
    assert paused.paused_by == 'api'
    again = await write(repo.set_paused, 'pausable', paused=True, by='other')
    assert (again.paused_at, again.paused_by) == (paused.paused_at, 'api')
    resumed = await write(repo.set_paused, 'pausable', paused=False)
    assert (resumed.paused_at, resumed.paused_by) == (None, None)


async def test_finish_delete_cascades_every_control_row(repo_env: None, pg: PgDatabase) -> None:
    created = await create('cascade', status='active')
    skip = {'project_indexes', 'class_registries', 'project_settings', 'project_keymaps'}
    params = ids(created.id, 'c')
    params.update(index_name='unused', upload_key='up/c', triton_name='m-c')
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        for table, statement in statements():
            if table not in skip:
                conn.execute(statement, params)
        before = conn.execute(
            'SELECT (SELECT count(*) FROM jobs WHERE project_id = %s),'
            ' (SELECT count(*) FROM project_classes WHERE project_id = %s)',
            (created.id, created.id),
        ).fetchone()
        assert before is not None
        assert before[0] > 0
        assert before[1] > 0
    await write(repo.mark_deleting, 'cascade', from_states=('active',))
    await write(repo.finish_delete, 'cascade')
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        for table in (
            'jobs',
            'project_classes',
            'class_registries',
            'project_settings',
            'project_keymaps',
            'project_indexes',
            'config_documents',
            'models',
        ):
            row = conn.execute(
                f'SELECT count(*) FROM {table} WHERE project_id = %s',  # nosec B608  fixed names
                (created.id,),
            ).fetchone()
            assert row == (0,), table
        assert conn.execute(
            'SELECT retired_at IS NOT NULL FROM project_slugs WHERE slug = %s', ('cascade',)
        ).fetchone() == (True,)


async def test_load_project_preserves_ids_and_timestamps_and_reads_back(
    repo_env: None,
) -> None:
    created_at = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.UTC)
    updated_at = dt.datetime(2026, 2, 3, 4, 5, 6, tzinfo=dt.UTC)
    paused_at = dt.datetime(2026, 3, 4, 5, 6, 7, tzinfo=dt.UTC)
    load = repo.ProjectLoad(
        slug='legacy',
        display_name='Legacy',
        description='from 0.6.0',
        status='archived',
        revision=7,
        created_at=created_at,
        updated_at=updated_at,
        index_prefix='op_prj_legacy__',
        storage_prefix='projects/legacy',
        model_prefix='legacy__',
        mlflow_experiment='openprocessor-legacy',
        indexes={'images': 'op_prj_legacy__images', 'items': 'op_prj_legacy__items'},
        created_by='upgrade',
        config_epoch=4,
        paused_at=paused_at,
        paused_by='cli',
        origin=[{'project_id': str(uuid.uuid4()), 'slug': 'src'}],
    )
    before = await read_sync(repo.read_projects_revision_sync)
    project_id = await write(repo.load_project, load)
    got = await read(repo.get_by_slug, 'legacy')
    assert got.id == project_id
    assert (got.status, got.revision, got.config_epoch) == ('archived', 7, 4)
    assert (got.created_at, got.updated_at, got.paused_at) == (created_at, updated_at, paused_at)
    assert got.origin == load.origin
    assert got.indexes == load.indexes
    assert await read_sync(repo.read_projects_revision_sync) == before + 1
    with pytest.raises(repo.SlugTakenError):
        await write(repo.load_project, load)
