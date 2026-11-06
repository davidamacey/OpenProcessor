"""Settings, keymap, state markers and projection state: revisions, OCC and the upgrade load."""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid

import pytest
from _config_fixtures import make_project

from src.db.repos import config as config_repo, settings as repo
from src.db.session import async_session_factory, deployment_tx, project_tx


pytestmark = pytest.mark.db


@pytest.fixture
def project(pg) -> uuid.UUID:
    return make_project(pg, f'set-{uuid.uuid4().hex[:12]}')


async def _put_defaults(project_id, expected, value):
    async with async_session_factory()() as session, project_tx(session, project_id):
        return await repo.put_settings(
            session, project_id, expected_revision=expected, changes={'defaults': value}
        )


@pytest.mark.asyncio
async def test_two_settings_writes_with_one_revision_one_wins(pg_env, project) -> None:
    results = await asyncio.gather(
        _put_defaults(project, 1, {'sort': 'a'}),
        _put_defaults(project, 1, {'sort': 'b'}),
        return_exceptions=True,
    )
    assert sorted(type(r).__name__ for r in results) == ['RevisionConflictError', 'SettingsRow']
    async with async_session_factory()() as session, project_tx(session, project):
        row = await repo.get_settings(session, project)
    assert row is not None
    assert row.revision == 2


@pytest.mark.asyncio
async def test_a_policy_column_write_leaves_the_others_alone(pg_env, project) -> None:
    await _put_defaults(project, 1, {'sort': 'a'})
    async with async_session_factory()() as session, project_tx(session, project):
        row = await repo.put_settings(
            session, project, expected_revision=2, changes={'ingest_policy': {'revision': 1}}
        )
    assert row.defaults == {'sort': 'a'}
    assert row.ingest_policy == {'revision': 1}
    assert row.revision == 3


@pytest.mark.asyncio
async def test_settings_refuse_unknown_columns_and_policies(pg_env, project) -> None:
    async with async_session_factory()() as session, project_tx(session, project):
        with pytest.raises(ValueError, match='not settings columns'):
            await repo.put_settings(session, project, expected_revision=1, changes={'revision': 9})
        with pytest.raises(ValueError, match='location_policy'):
            await repo.put_settings(
                session, project, expected_revision=1, changes={'location_policy': 'everything'}
            )


@pytest.mark.asyncio
async def test_keymap_writes_are_optimistic(pg_env, project) -> None:
    async def put(expected, combo):
        async with async_session_factory()() as session, project_tx(session, project):
            return await repo.put_keymap(
                session, project, overrides={'a': [combo]}, expected_revision=expected
            )

    results = await asyncio.gather(put(1, 'x'), put(1, 'y'), return_exceptions=True)
    assert sorted(type(r).__name__ for r in results) == ['KeymapRow', 'RevisionConflictError']


@pytest.mark.asyncio
async def test_state_markers_are_optimistic_and_keys_are_closed(pg_env, project) -> None:
    async with async_session_factory()() as session, project_tx(session, project):
        first = await repo.put_state(
            session, project, 'cluster_refresh', {'n': 1}, expected_revision=0
        )
        assert first.revision == 1
        with pytest.raises(config_repo.RevisionConflictError):
            await repo.put_state(session, project, 'cluster_refresh', {'n': 2}, expected_revision=0)
        second = await repo.put_state(
            session, project, 'cluster_refresh', {'n': 2}, expected_revision=1
        )
        assert (second.revision, second.value) == (2, {'n': 2})
        with pytest.raises(ValueError, match='unknown project_state key'):
            await repo.get_state(session, project, 'last_clustering')


@pytest.mark.asyncio
async def test_projection_state_is_replaced_in_place(pg_env, project) -> None:
    when = dt.datetime(2026, 4, 1, tzinfo=dt.UTC)
    for version in (1, 2):
        async with async_session_factory()() as session, project_tx(session, project):
            await repo.put_projection_state(
                session,
                project,
                projection_version=version,
                scope='all',
                cluster_id=None,
                n_points=10,
                n_components=2,
                metric='cosine',
                fitted_at=when,
                reducer_storage_key=f'projects/x/umap/{version}.bin',
            )
    async with async_session_factory()() as session, project_tx(session, project):
        row = await repo.get_projection_state(session, project)
    assert row is not None
    assert (row.projection_version, row.reducer_storage_key) == (2, 'projects/x/umap/2.bin')


@pytest.mark.asyncio
async def test_load_setting_and_keymap_set_the_given_revision_and_read_back(
    pg_env, project
) -> None:
    async with async_session_factory()() as session, deployment_tx(session, 'load test'):
        await repo.load_setting(session, project, 'vlm_policy', {'scope': 'all'}, revision=7)
        await repo.load_keymap(session, project, {'save': ['ctrl+s']}, revision=4)
    async with async_session_factory()() as session, project_tx(session, project):
        settings = await repo.get_settings(session, project)
        keymap = await repo.get_keymap(session, project)
    assert settings is not None
    assert (settings.vlm_policy, settings.revision) == ({'scope': 'all'}, 7)
    assert keymap is not None
    assert (keymap.overrides, keymap.revision) == ({'save': ['ctrl+s']}, 4)
    async with async_session_factory()() as session, deployment_tx(session, 'load test'):
        with pytest.raises(ValueError, match='not a settings column'):
            await repo.load_setting(session, project, 'revision', 1, revision=1)
