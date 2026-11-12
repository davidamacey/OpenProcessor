"""The activation axes of each preset kind are served with the schema, not guessed by the client."""

from __future__ import annotations

import pytest
from _curation_app import SCOPED, mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.curation import router as curation_router
from src.routers.curation.projects import global_router
from src.services.config_store.research_kinds import KIND_AXES, RESEARCH_KINDS


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.include_router(global_router, prefix='/curation')
    return TestClient(app)


_COLLECTIONS = {
    'serving_profile': '/curation/serving_profiles',
    'cluster_profile': f'{SCOPED}/cluster_profiles',
    'assist_loop_preset': f'{SCOPED}/assist_loop_presets',
    'near_dup_profile': f'{SCOPED}/near_dup_profiles',
}


def _axes(client: TestClient, kind: str) -> list[dict]:
    r = client.get(f'{_COLLECTIONS[kind]}/schema')
    assert r.status_code == 200, r.text
    return r.json()['axes']


def test_the_cluster_profile_serves_its_two_axes_with_label_and_target(client: TestClient) -> None:
    assert _axes(client, 'cluster_profile') == [
        {'axis': 'clustering_item', 'label': 'Items', 'target': 'item'},
        {'axis': 'clustering_region_box', 'label': 'Region boxes', 'target': 'region_box'},
    ]


def test_single_axis_kinds_serve_one_axis_without_a_target(client: TestClient) -> None:
    assert _axes(client, 'assist_loop_preset') == [
        {'axis': 'assist_loop_preset', 'label': 'Assist loop', 'target': None}
    ]
    assert _axes(client, 'near_dup_profile') == [
        {'axis': 'near_dup', 'label': 'Near-duplicates', 'target': None}
    ]


def test_the_deployment_wide_serving_profile_has_no_axis(client: TestClient) -> None:
    assert _axes(client, 'serving_profile') == []


@pytest.mark.parametrize('kind', sorted(RESEARCH_KINDS))
def test_the_served_axes_are_exactly_the_axes_the_activation_accepts(
    client: TestClient, kind: str
) -> None:
    assert [a['axis'] for a in _axes(client, kind)] == list(KIND_AXES.get(kind, ()))
