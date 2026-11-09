"""Cluster-refresh state in project_state and the UMAP view state in projection_states."""

from __future__ import annotations

import datetime as dt

import pytest
from _config_fixtures import make_project

from src.services.config_store.bound_project import bound_project_id
from src.services.curation import embedding_viz, embedding_viz_read
from src.services.curation.clustering import umap_state
from src.services.curation.policy_doc_store import PolicyConflictError
from src.services.periodic import cluster_refresh_clock as clock


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


@pytest.fixture(scope='module', autouse=True)
def _default_project(pg):
    """The autouse binding of the suite names the project ``default``."""
    return make_project(pg, 'default')


@pytest.fixture
def storage_root(tmp_path, monkeypatch):
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_refresh_state_defaults_then_round_trips(pg_env) -> None:
    assert await clock.load_refresh_state() == clock.RefreshState()
    saved = await clock.save_refresh_state(clock.RefreshState(prev_count=40, job_id='j1'))
    assert saved.revision == 1
    loaded = await clock.load_refresh_state()
    assert (loaded.revision, loaded.prev_count, loaded.job_id) == (1, 40, 'j1')


@pytest.mark.asyncio
async def test_a_refresh_state_built_from_a_stale_revision_conflicts(pg_env) -> None:
    first = await clock.load_refresh_state()
    await clock.save_refresh_state(first.model_copy(update={'prev_count': 1}))
    with pytest.raises(PolicyConflictError):
        await clock.save_refresh_state(first.model_copy(update={'prev_count': 2}))


@pytest.mark.asyncio
async def test_the_reducer_bytes_go_through_the_storage_seam(pg_env, storage_root) -> None:
    key = umap_state.save_reducer({'weights': [1, 2, 3]}, state_id='current')
    assert key == 'projects/default/umap/current.bin'
    assert (storage_root / 'default' / 'umap' / 'current.bin').is_file()
    assert umap_state.load_reducer(state_id='current') == {'weights': [1, 2, 3]}
    assert umap_state.load_reducer(state_id='current_cuml') is None


@pytest.mark.asyncio
async def test_a_fit_writes_the_view_state_and_the_viz_read_serves_the_same_shape(
    pg_env, storage_root
) -> None:
    with pytest.raises(embedding_viz_read.ProjectionNotBuiltError):
        await embedding_viz_read._load_run_metadata()
    umap_state.save_reducer({'fitted': True}, state_id=embedding_viz.VIZ_REDUCER_STATE_ID)
    fitted_at = dt.datetime(2026, 5, 1, 12, tzinfo=dt.UTC).isoformat()
    await embedding_viz._save_run_metadata(
        scope='all', cluster_uid=None, n_points=7, fitted_at=fitted_at
    )
    meta = await embedding_viz_read._load_run_metadata()
    assert meta == {
        'projection_version': embedding_viz.VIZ_PROJECTION_VERSION,
        'scope': 'all',
        'cluster_uid': None,
        'n_points': 7,
        'fitted_at': fitted_at,
        'n_components': embedding_viz.VIZ_N_COMPONENTS,
        'metric': embedding_viz.VIZ_METRIC,
    }
    from src.db.repos import settings as settings_repo
    from src.db.session import async_session_factory, project_tx

    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        row = await settings_repo.get_projection_state(session, project_id)
    assert row is not None
    assert row.reducer_storage_key == 'projects/default/umap/viz.bin'
