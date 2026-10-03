"""W4: ``/region_profiles*`` full CRUD lifecycle, template read-only-ness,
activation/rollback/deactivate, and revisions (any_domain_plan.md §4/§7.3)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from _region_profile_fixture import EXAMPLE_LICENSE_PLATE_PROFILE
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch


PREFIX = '/curation/projects/default/region_profiles'


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.services.config_store.store import reset_config_stores
    from src.services.detection import profile_registry

    reset_config_stores()
    profile_registry._reset_registry_for_tests()
    yield
    reset_config_stores()
    profile_registry._reset_registry_for_tests()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = FakeConfigOpenSearch()

    async def _fake_repo_index() -> list[dict]:
        return [
            {'name': 'my_detector', 'state': 'READY', 'version': '1'},
            {'name': 'paddleocr_det_trt', 'state': 'READY', 'version': '1'},
            {'name': 'paddleocr_rec_trt', 'state': 'READY', 'version': '1'},
            {'name': 'ocr_pipeline', 'state': 'READY', 'version': '1'},
        ]

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    from src.services.triton_control import TritonControlService

    monkeypatch.setattr(
        TritonControlService, 'get_repository_index', lambda _self: _fake_repo_index()
    )

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


def _body(**overrides: object) -> dict[str, object]:
    from dataclasses import asdict

    from src.config import DetectionProfile

    raw = asdict(DetectionProfile(name='p'))
    raw.pop('name')
    for key, value in raw.items():
        if isinstance(value, frozenset):
            raw[key] = sorted(value)
        elif isinstance(value, tuple):
            raw[key] = list(value)
    raw['text_reader'] = 'none'
    raw['detector_model'] = 'my_detector'
    raw['display_name'] = 'X'
    raw['display_name_singular'] = 'x'
    raw.update(overrides)
    return raw


def test_full_lifecycle(app_client: TestClient) -> None:
    r = app_client.post(PREFIX, json={'name': 'my_profile', 'description': 'd1', 'body': _body()})
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc['source'] == 'stored'
    assert doc['revision'] == 1

    r = app_client.put(
        f'{PREFIX}/my_profile', json={'expected_revision': 0, 'description': 'd2', 'body': _body()}
    )
    assert r.status_code == 409, r.text
    assert r.json()['detail']['error'] == 'revision_conflict'

    r = app_client.put(
        f'{PREFIX}/my_profile', json={'expected_revision': 1, 'description': 'd2', 'body': _body()}
    )
    assert r.status_code == 200, r.text
    assert r.json()['revision'] == 2
    # PUT does not change what's active (there's no activation yet either way).
    assert r.json()['active'] is False

    r = app_client.post(
        f'{PREFIX}/my_profile/activate',
        json={'expected_active': {'name': 'someone_else', 'revision': None}, 'force': False},
    )
    assert r.status_code == 409, r.text
    assert r.json()['detail']['error'] == 'active_conflict'

    r = app_client.post(
        f'{PREFIX}/my_profile/activate', json={'expected_active': None, 'force': False}
    )
    assert r.status_code == 200, r.text
    assert r.json()['active']['name'] == 'my_profile'
    assert 'impact' in r.json()

    r = app_client.get(f'{PREFIX}/active')
    assert r.json()['active']['name'] == 'my_profile'

    r = app_client.delete(f'{PREFIX}/my_profile', params={'expected_revision': 2})
    assert r.status_code == 409, r.text
    assert r.json()['detail']['error'] == 'in_use'

    active = app_client.get(f'{PREFIX}/active').json()
    r = app_client.post(f'{PREFIX}/deactivate', json={'expected_active': active['active']})
    assert r.status_code == 200, r.text
    assert r.json()['active']['name'] is None

    r = app_client.delete(f'{PREFIX}/my_profile', params={'expected_revision': 2})
    assert r.status_code == 204, r.text

    r = app_client.get(f'{PREFIX}/my_profile')
    assert r.status_code == 404


def test_activate_then_rollback(app_client: TestClient) -> None:
    app_client.post(PREFIX, json={'name': 'p1', 'body': _body()})
    app_client.post(PREFIX, json={'name': 'p2', 'body': _body()})
    app_client.post(f'{PREFIX}/p1/activate', json={'expected_active': None, 'force': False})
    active = app_client.get(f'{PREFIX}/active').json()
    app_client.post(
        f'{PREFIX}/p2/activate', json={'expected_active': active['active'], 'force': False}
    )
    active2 = app_client.get(f'{PREFIX}/active').json()
    assert active2['active']['name'] == 'p2'

    r = app_client.post(f'{PREFIX}/active/rollback', json={'expected_active': active2['active']})
    assert r.status_code == 200, r.text
    assert r.json()['active']['name'] == 'p1'


def test_rollback_with_no_previous_409s(app_client: TestClient) -> None:
    r = app_client.post(f'{PREFIX}/active/rollback', json={'expected_active': None})
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'no_previous'


def test_template_activation_is_403(app_client: TestClient) -> None:
    r = app_client.get(f'{PREFIX}?include_templates=true')
    assert r.status_code == 200, r.text
    names = {t['name'] for t in r.json()['templates']}
    assert 'license_plate' in names

    r = app_client.post(
        f'{PREFIX}/license_plate/activate', json={'expected_active': None, 'force': False}
    )
    assert r.status_code == 403, r.text
    assert r.json()['detail']['error'] == 'read_only'


def test_delete_active_is_409_in_use(app_client: TestClient) -> None:
    app_client.post(PREFIX, json={'name': 'my_profile', 'body': _body()})
    app_client.post(f'{PREFIX}/my_profile/activate', json={'expected_active': None, 'force': False})
    r = app_client.delete(f'{PREFIX}/my_profile', params={'expected_revision': 1})
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'in_use'


def test_revisions_list_and_get_by_revision(app_client: TestClient) -> None:
    app_client.post(PREFIX, json={'name': 'my_profile', 'body': _body()})
    body2 = _body(display_name='Changed')
    app_client.put(f'{PREFIX}/my_profile', json={'expected_revision': 1, 'body': body2})

    r = app_client.get(f'{PREFIX}/my_profile/revisions')
    assert r.status_code == 200, r.text
    revisions = sorted(rv['revision'] for rv in r.json()['revisions'])
    assert revisions == [1, 2]

    r2 = app_client.get(f'{PREFIX}/my_profile/revisions/2')
    assert r2.json()['body']['display_name'] == 'Changed'


def test_etag_header(app_client: TestClient) -> None:
    app_client.post(PREFIX, json={'name': 'my_profile', 'body': _body()})
    r = app_client.get(f'{PREFIX}/my_profile')
    assert r.headers['etag'] == '"region_profile:my_profile:1"'


def test_clone_from_template(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://segmenter:8000')
    from src.routers.curation import _models_segmenter

    async def _health(url: str) -> tuple[str, str | None]:
        return 'ready', None

    monkeypatch.setattr(_models_segmenter, '_segmenter_health', _health)
    r = app_client.post(
        f'{PREFIX}/license_plate/clone', json={'new_name': 'my_plates', 'source': 'template'}
    )
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc['source'] == 'stored'
    assert doc['body']['region_class_name'] == EXAMPLE_LICENSE_PLATE_PROFILE.region_class_name


def test_validate_route_reports_no_candidate_source(app_client: TestClient) -> None:
    body = _body(detector_model='', segmenter_text_prompt='')
    r = app_client.post(f'{PREFIX}/validate', json={'name': None, 'body': body})
    assert r.status_code == 200, r.text
    codes = {e['code'] for e in r.json()['errors']}
    assert 'no_candidate_source' in codes


def test_active_impact_route(app_client: TestClient) -> None:
    r = app_client.get(f'{PREFIX}/active/impact')
    assert r.status_code == 200, r.text
    body = r.json()
    assert 'items_total' in body
    assert 'by_profile' in body


def test_from_project_clone_is_read_only_and_never_writes_to_source(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one legitimate cross-project read in this wave (mirrors W3's
    prompt-pack from_project pin): cloning a region profile out of
    another project must never write anything to that project's own
    config-store index."""
    from src.config.curation import base_curation_config
    from src.config.project_context import bind_project
    from src.config.projects import ProjectRecord, resources_for_new
    from src.services.projects.registry import ProjectRegistry

    fake_os = app_client.fake_os  # type: ignore[attr-defined]

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

    async def _seed_other() -> str:
        from src.services.config_store.index import save_config

        with bind_project(other):
            from src.config import get_curation_config

            idx = get_curation_config().configs_index
            body = _body()
            await save_config(
                fake_os,
                idx,
                kind='region_profile',
                name='other_profile',
                body=body,
                expected_revision=None,
            )
            return idx

    import asyncio

    other_index = asyncio.run(_seed_other())

    monkeypatch.setattr(ProjectRegistry, 'ensure_fresh', AsyncMock(return_value=None))
    from src.services.projects import lifecycle as lifecycle_mod

    monkeypatch.setattr(lifecycle_mod, '_resolve_existing', AsyncMock(return_value=other))

    before = dict(fake_os._docs.get(other_index, {}))

    r = app_client.post(
        f'{PREFIX}/other_profile/clone',
        json={'new_name': 'cloned_from_other', 'from_project': 'other'},
    )
    assert r.status_code == 201, r.text
    assert r.json()['cloned_from'].startswith('other:other_profile@')

    after = dict(fake_os._docs.get(other_index, {}))
    assert before == after

    r_get = app_client.get(f'{PREFIX}/cloned_from_other')
    assert r_get.status_code == 200
    assert r_get.json()['body']['detector_model'] == 'my_detector'


def test_a_profile_created_through_another_worker_can_be_activated_at_once(
    app_client: TestClient,
) -> None:
    """Found live (32 API workers): create on one worker, activate on another
    within a second answered 404 -- the second worker's snapshot was under a
    second old, so it never looked at the revision counter."""
    import time
    from dataclasses import replace

    from src.services.config_store.store import get_config_store

    r = app_client.post(PREFIX, json={'name': 'my_profile', 'description': '', 'body': _body()})
    assert r.status_code == 201, r.text
    store = get_config_store()
    store.current = replace(
        store.current, profiles={}, config_revision=0, loaded_at=time.monotonic()
    )

    r = app_client.post(
        f'{PREFIX}/my_profile/activate', json={'expected_active': None, 'force': False}
    )

    assert r.status_code == 200, r.text


def test_deactivate_conflict_message_spells_out_the_expected_active_shape(
    app_client: TestClient,
) -> None:
    r = app_client.post(f'{PREFIX}', json={'name': 'p1', 'description': '', 'body': _body()})
    assert r.status_code in (200, 201), r.text
    r = app_client.post(f'{PREFIX}/p1/activate', json={'expected_active': None, 'force': False})
    assert r.status_code == 200, r.text

    r = app_client.post(f'{PREFIX}/deactivate', json={})
    assert r.status_code == 409, r.text
    detail = r.json()['detail']
    assert detail['error'] == 'active_conflict'
    assert 'expected_active' in detail['message']
    assert '"name"' in detail['message']
    assert detail['current']['name'] == 'p1'
