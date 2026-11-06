"""The research config kinds: serving_profile (deployment), cluster_profile and assist_loop_preset."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from _config_fixtures import make_project
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.db.repos import config as repo
from src.db.session import async_session_factory, deployment_tx, project_tx
from src.schemas.curation.research_config import AssistLoopPresetBody, ServingProfileBody


pytestmark = pytest.mark.db

SERVING: dict[str, Any] = {
    'servers': [
        {
            'name': 'gpu0',
            'grpc_url': 'triton:4601',
            'metrics_url': 'triton:4602',
            'gpu_uuids': ['GPU-aaaa'],
            'numa_node': 0,
        }
    ],
    'load_list': {'gpu0': ['yolo11n', 'scrfd']},
    'instances': {'yolo11n': 2, 'scrfd': 1},
    'image_prep': {
        'instances': 1,
        'workers': 2,
        'max_in_flight': 8,
        'device_budget_mib': 1024,
        'min_megapixels': 0,
    },
    'priorities': {'interactive': 10, 'bulk': 1},
}
CLUSTER: dict[str, Any] = {
    'target': 'item',
    'level': 'item',
    'embedding_field': 'pe_embedding',
    'method': 'ivf',
    'refresh': {'growth_threshold': 0.2, 'max_deferral_s': 1800, 'retry_backoff_s': 600},
}
PRESET: dict[str, Any] = {'budget': {'max_vlm_crops': 3000}}


@pytest.fixture(scope='module', autouse=True)
def _default_project(pg):
    return make_project(pg, 'default')


@pytest.fixture
def client(pg_env) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router
    from src.routers.curation.projects import global_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.include_router(global_router, prefix='/curation')
    app.dependency_overrides[_raw_opensearch_dep] = lambda: object()  # noqa: PLW0108  # FastAPI reads a bare class as a dependency with parameters
    return TestClient(app)


@pytest.mark.asyncio
async def test_a_serving_profile_with_a_project_is_refused_by_the_repository_and_the_database(
    pg_env, pg
) -> None:
    import psycopg
    from psycopg import errors

    project = make_project(pg, f'srv-{uuid.uuid4().hex[:10]}')
    async with async_session_factory()() as session, project_tx(session, project):
        with pytest.raises(repo.ConfigScopeError):
            await repo.save_config(
                session,
                project_id=project,
                kind='serving_profile',
                name='x',
                body=SERVING,
                expected_revision=None,
            )
    with (
        psycopg.connect(pg.conninfo('super'), autocommit=True) as conn,
        pytest.raises(errors.CheckViolation),
    ):
        conn.execute(
            'INSERT INTO config_documents (project_id, kind, name, current_revision, updated_by) '
            "VALUES (%s, 'serving_profile', 'raw', 1, 'test')",
            (project,),
        )


@pytest.mark.asyncio
async def test_a_cluster_profile_on_the_vlm_axis_is_refused(pg_env, pg) -> None:
    from src.services.config_store import research_kinds as svc

    project = make_project(pg, f'cp-{uuid.uuid4().hex[:10]}')
    async with async_session_factory()() as session, project_tx(session, project):
        doc = await repo.save_config(
            session,
            project_id=project,
            kind='cluster_profile',
            name='ivf',
            body=CLUSTER,
            expected_revision=None,
        )
    async with async_session_factory()() as session, deployment_tx(session, 'axis test'):
        with pytest.raises(repo.AxisKindError):
            await repo.activate(
                session,
                project_id=project,
                axis='vlm',
                config_id=doc.id,
                revision=1,
                expected_active=None,
            )
    assert svc.KIND_AXES['cluster_profile'] == ('clustering_item', 'clustering_region_box')


def test_an_assist_loop_preset_without_max_vlm_crops_fails_validation() -> None:
    AssistLoopPresetBody.model_validate(PRESET)
    with pytest.raises(ValidationError):
        AssistLoopPresetBody.model_validate({'budget': {}})
    with pytest.raises(ValidationError):
        AssistLoopPresetBody.model_validate({'budget': {'max_vlm_crops': 10}, 'unknown': 1})


def test_a_serving_profile_must_name_known_servers() -> None:
    ServingProfileBody.model_validate(SERVING)
    with pytest.raises(ValidationError, match='servers that do not exist'):
        ServingProfileBody.model_validate({**SERVING, 'load_list': {'ghost': ['m']}})


def test_the_serving_routes_save_activate_and_refuse_a_stale_expected_revision(
    client: TestClient,
) -> None:
    put = client.put(
        '/curation/serving_profiles/standard', json={'body': SERVING, 'expected_revision': None}
    )
    assert put.status_code == 200, put.text
    assert put.json()['revision'] == 1
    assert client.get('/curation/serving_profiles/active').status_code == 404

    first = client.post('/curation/serving_profiles/standard/activate', json={})
    assert first.status_code == 200, first.text
    setting = first.json()['setting_revision']
    stale = client.post(
        '/curation/serving_profiles/standard/activate',
        json={'expected_revision': setting + 5},
    )
    assert stale.status_code == 409, stale.text
    assert stale.json()['detail']['error'] == 'revision_conflict'
    again = client.post(
        '/curation/serving_profiles/standard/activate', json={'expected_revision': setting}
    )
    assert again.status_code == 200
    assert client.get('/curation/serving_profiles/active').json()['name'] == 'standard'


def test_a_serving_profile_save_is_validated_and_optimistic(client: TestClient) -> None:
    bad = client.put(
        '/curation/serving_profiles/broken',
        json={'body': {**SERVING, 'servers': []}, 'expected_revision': None},
    )
    assert bad.status_code == 422
    client.put('/curation/serving_profiles/occ', json={'body': SERVING, 'expected_revision': None})
    clash = client.put(
        '/curation/serving_profiles/occ', json={'body': SERVING, 'expected_revision': None}
    )
    assert clash.status_code == 409


def test_cluster_profile_and_preset_routes_activate_on_their_axes(client: TestClient) -> None:
    base = '/curation/projects/default'
    assert (
        client.put(
            f'{base}/cluster_profiles/ivf', json={'body': CLUSTER, 'expected_revision': None}
        ).status_code
        == 200
    )
    item = client.post(f'{base}/cluster_profiles/ivf/activate', json={'axis': 'clustering_item'})
    assert item.status_code == 200, item.text
    region = client.post(
        f'{base}/cluster_profiles/ivf/activate', json={'axis': 'clustering_region_box'}
    )
    assert region.status_code == 200
    assert client.get(f'{base}/cluster_profiles/active').json()['name'] == 'ivf'
    wrong = client.post(f'{base}/cluster_profiles/ivf/activate', json={'axis': 'vlm'})
    assert wrong.status_code == 422

    missing_budget = client.put(
        f'{base}/assist_loop_presets/p', json={'body': {'budget': {}}, 'expected_revision': None}
    )
    assert missing_budget.status_code == 422
    ok = client.put(
        f'{base}/assist_loop_presets/p', json={'body': PRESET, 'expected_revision': None}
    )
    assert ok.status_code == 200, ok.text
    activated = client.post(f'{base}/assist_loop_presets/p/activate', json={})
    assert activated.status_code == 200
    stale = client.post(
        f'{base}/assist_loop_presets/p/activate',
        json={'expected_active': {'name': 'other', 'revision': 1}},
    )
    assert stale.status_code == 409
    assert stale.json()['detail']['error'] == 'active_conflict'
