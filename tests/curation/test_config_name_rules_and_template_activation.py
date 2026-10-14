"""Name rules on PUT and clone for prompt packs and region profiles (reserved and invalid
names are refused, clone onto an existing name conflicts), and template packs are read-only."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


pytestmark = pytest.mark.usefixtures('vlm_env')


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = FakeConfigOpenSearch()
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


PREFIX = '/curation/projects/default/prompt_packs'


def pack_body() -> dict[str, object]:
    body = GENERIC_ITEM_PACK.to_dict()
    body.pop('name')
    return body


def test_put_cannot_create_a_reserved_name(app_client: TestClient) -> None:
    r = app_client.put(f'{PREFIX}/off', json={'expected_revision': 0, 'body': pack_body()})
    print('PUT off ->', r.status_code)
    assert r.status_code in (404, 422), r.text


def test_put_cannot_create_an_invalid_name(app_client: TestClient) -> None:
    r = app_client.put(f'{PREFIX}/Not_A_Slug', json={'expected_revision': 0, 'body': pack_body()})
    print('PUT Not_A_Slug ->', r.status_code)
    assert r.status_code in (404, 422), r.text


def test_clone_cannot_create_a_reserved_name(app_client: TestClient) -> None:
    r = app_client.post(f'{PREFIX}/{GENERIC_ITEM_PACK.name}/clone', json={'new_name': 'default'})
    print('clone -> default', r.status_code)
    assert r.status_code == 422, r.text


def test_template_pack_cannot_be_activated(app_client: TestClient) -> None:
    from src.services.labeling.vlm_prompts import active_prompt_pack

    r = app_client.post(f'{PREFIX}/vehicle_wheel/activate', json={'expected_active': None})
    print('activate template ->', r.status_code, active_prompt_pack().name)
    assert r.status_code in (403, 422), r.text


def test_profile_put_cannot_create_reserved_name(app_client: TestClient) -> None:
    r = app_client.put(
        '/curation/projects/default/region_profiles/none',
        json={'expected_revision': 0, 'body': {'segmenter_text_prompt': 'x region'}},
    )
    print('profile PUT none ->', r.status_code, r.text[:300])
    assert r.status_code in (404, 422), r.text


def _profile_body() -> dict[str, object]:
    from dataclasses import asdict

    from src.config import DetectionProfile

    raw = asdict(DetectionProfile(name='p'))
    raw.pop('name')
    for k, v in raw.items():
        if isinstance(v, frozenset):
            raw[k] = sorted(v)
        elif isinstance(v, tuple):
            raw[k] = list(v)
    raw.update(
        detector_model='',
        text_reader='none',
        segmenter_text_prompt='test region',
        display_name='Regions',
        display_name_singular='Region',
    )
    return raw


def test_profile_put_and_clone_enforce_name_rules(app_client, monkeypatch) -> None:
    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://nowhere:1')
    base = '/curation/projects/default/region_profiles'
    r = app_client.put(f'{base}/none', json={'expected_revision': 0, 'body': _profile_body()})
    print('profile PUT none ->', r.status_code)
    # Clone from a real stored profile (not the never-created 'none') so
    # the test actually reaches the new_name check instead of 404ing on
    # an unknown clone source.
    created = app_client.post(base, json={'name': 'src_profile', 'body': _profile_body()})
    assert created.status_code == 201, created.text
    r2 = app_client.post(f'{base}/src_profile/clone', json={'new_name': 'Bad Name'})
    print('profile clone -> "Bad Name"', r2.status_code)
    assert r.status_code in (404, 422)
    assert r2.status_code == 422


def test_from_project_clone_onto_name_unknown_to_cold_target_store_is_409(
    app_client, monkeypatch
) -> None:
    import asyncio

    from src.config.curation import base_curation_config
    from src.config.project_context import bind_project
    from src.config.projects import ProjectRecord, resources_for_new
    from src.services.config_store.index import save_config
    from src.services.config_store.store import reset_config_stores
    from src.services.projects import lifecycle as lifecycle_mod
    from src.services.projects.registry import ProjectRegistry

    fake_os = app_client.fake_os
    other = ProjectRecord(
        slug='other',
        display_name='Other',
        description='',
        status='active',
        revision=1,
        created_at='',
        updated_at='',
        origin=None,
        resources=resources_for_new('other', base_curation_config()),
    )

    async def _seed() -> None:
        from src.config import get_curation_config

        with bind_project(other):
            idx = get_curation_config().configs_index
            await save_config(
                fake_os,
                idx,
                kind='prompt_pack',
                name='other_pack',
                body={**GENERIC_ITEM_PACK.to_dict(), 'name': 'other_pack'},
                expected_revision=None,
            )
        # target ('default', bound by the test conftest) already has 'taken' in OpenSearch,
        # written by another worker process -> this process's store never saw it
        idx = get_curation_config().configs_index
        await save_config(
            fake_os, idx, kind='prompt_pack', name='taken', body=pack_body(), expected_revision=None
        )

    asyncio.run(_seed())
    reset_config_stores()
    monkeypatch.setattr(ProjectRegistry, 'ensure_fresh', AsyncMock(return_value=None))
    monkeypatch.setattr(lifecycle_mod, '_resolve_existing', AsyncMock(return_value=other))
    client = TestClient(app_client.app, raise_server_exceptions=False)
    r = client.post(
        f'{PREFIX}/other_pack/clone', json={'new_name': 'taken', 'from_project': 'other'}
    )
    print('cold-target from_project clone onto existing name ->', r.status_code, r.text[:120])
    assert r.status_code == 409, r.text


def test_template_pack_activation_is_read_only(app_client):
    r = app_client.post(f'{PREFIX}/vehicle_wheel/activate', json={'expected_active': None})
    print(r.status_code, r.text[:200])
    assert r.status_code == 403
    assert r.json()['detail']['error'] == 'read_only' if 'detail' in r.json() else True


def test_pack_put_and_clone_name_rules(app_client):
    for bad in ('off', 'default', 'none', 'Not_A_Slug', 'Bad Name'):
        r = app_client.put(f'{PREFIX}/{bad}', json={'expected_revision': 0, 'body': pack_body()})
        assert r.status_code in (404, 422), (bad, r.status_code)
    assert app_client.post(PREFIX, json={'name': 'srcp', 'body': pack_body()}).status_code == 201
    for bad in ('off', 'default', 'none', 'Not_A_Slug', 'Bad Name'):
        r = app_client.post(f'{PREFIX}/srcp/clone', json={'new_name': bad})
        print('clone', bad, r.status_code)
        assert r.status_code == 422, (bad, r.status_code, r.text[:200])
    r = app_client.post(f'{PREFIX}/srcp/clone', json={'new_name': 'srcp'})
    assert r.status_code == 409


def test_profile_clone_and_put_reserved_names(app_client, monkeypatch):

    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://nowhere:1')
    base = '/curation/projects/default/region_profiles'
    assert (
        app_client.post(base, json={'name': 'srcprof', 'body': _profile_body()}).status_code == 201
    )
    for bad in ('off', 'default', 'none', 'Not_A_Slug'):
        r = app_client.put(f'{base}/{bad}', json={'expected_revision': 0, 'body': _profile_body()})
        assert r.status_code in (404, 422), (bad, r.status_code)
        r = app_client.post(f'{base}/srcprof/clone', json={'new_name': bad})
        print('profile clone', bad, r.status_code)
        assert r.status_code == 422, (bad, r.status_code, r.text[:200])


def test_put_on_active_profile_does_not_change_runtime(app_client, monkeypatch):
    from src.services.detection.profile_registry import get_active_region_profile

    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://nowhere:1')
    base = '/curation/projects/default/region_profiles'
    assert (
        app_client.post(base, json={'name': 'liveprof', 'body': _profile_body()}).status_code == 201
    )
    r = app_client.post(f'{base}/liveprof/activate', json={'expected_active': None, 'force': True})
    assert r.status_code == 200, r.text
    b2 = _profile_body()
    b2['max_regions_per_item'] = 7
    b2['display_name'] = 'EDITED'
    r = app_client.put(f'{base}/liveprof', json={'expected_revision': 1, 'body': b2})
    assert r.status_code == 200, r.text
    p = get_active_region_profile()
    assert p is not None
    print('active profile', p.name, p.max_regions_per_item, p.display_name)
    assert p.max_regions_per_item != 7
    assert p.display_name != 'EDITED'
