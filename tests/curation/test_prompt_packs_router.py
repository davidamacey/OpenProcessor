"""W3: ``/prompt_packs*`` full CRUD lifecycle, template read-only-ness,
clone (same-project and cross-project), revisions and ETag
(any_domain_plan.md §3/§7.2)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


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


def _body() -> dict[str, object]:
    body = GENERIC_ITEM_PACK.to_dict()
    body.pop('name')
    return body


def test_full_lifecycle(app_client: TestClient) -> None:
    # create
    r = app_client.post(PREFIX, json={'name': 'my_pack', 'description': 'd1', 'body': _body()})
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc['source'] == 'stored'
    assert doc['revision'] == 1
    assert doc['read_only'] is False

    # validate (never writes, always 200)
    r = app_client.post(f'{PREFIX}/validate', json={'name': 'my_pack', 'body': _body()})
    assert r.status_code == 200, r.text
    assert any(e['code'] == 'name_conflict' for e in r.json()['errors'])

    # PUT with a stale revision -> 409 revision_conflict
    r = app_client.put(
        f'{PREFIX}/my_pack', json={'expected_revision': 0, 'description': 'd2', 'body': _body()}
    )
    assert r.status_code == 409, r.text
    assert r.json()['detail']['error'] == 'revision_conflict'

    # real save
    r = app_client.put(
        f'{PREFIX}/my_pack', json={'expected_revision': 1, 'description': 'd2', 'body': _body()}
    )
    assert r.status_code == 200, r.text
    assert r.json()['revision'] == 2

    # activate with a stale expected_active -> 409 active_conflict
    r = app_client.post(
        f'{PREFIX}/my_pack/activate',
        json={'expected_active': {'name': 'someone_else', 'revision': None}, 'force': False},
    )
    assert r.status_code == 409, r.text
    assert r.json()['detail']['error'] == 'active_conflict'

    # real activate (no prior activation -> expected_active null)
    r = app_client.post(
        f'{PREFIX}/my_pack/activate', json={'expected_active': None, 'force': False}
    )
    assert r.status_code == 200, r.text
    assert r.json()['active']['name'] == 'my_pack'
    assert r.json()['active']['revision'] == 2

    # activate a second (builtin) pack, so 'my_pack' becomes the previous
    active = app_client.get(f'{PREFIX}/active').json()
    r = app_client.post(
        f'{PREFIX}/{GENERIC_ITEM_PACK.name}/activate',
        json={'expected_active': active['active'], 'force': False},
    )
    assert r.status_code == 200, r.text
    assert r.json()['active']['name'] == GENERIC_ITEM_PACK.name

    # rollback -> re-activates 'my_pack'
    active = app_client.get(f'{PREFIX}/active').json()
    r = app_client.post(f'{PREFIX}/active/rollback', json={'expected_active': active['active']})
    assert r.status_code == 200, r.text
    assert r.json()['active']['name'] == 'my_pack'

    # delete-active -> 409 in_use
    r = app_client.delete(f'{PREFIX}/my_pack', params={'expected_revision': 2})
    assert r.status_code == 409, r.text
    assert r.json()['detail']['error'] == 'in_use'

    # move the activation off my_pack, then delete succeeds
    active = app_client.get(f'{PREFIX}/active').json()
    app_client.post(f'{PREFIX}/active/rollback', json={'expected_active': active['active']})
    r = app_client.delete(f'{PREFIX}/my_pack', params={'expected_revision': 2})
    assert r.status_code == 204, r.text

    r = app_client.get(f'{PREFIX}/my_pack')
    assert r.status_code == 404


def test_rollback_with_no_previous_activation_409s(app_client: TestClient) -> None:
    r = app_client.post(f'{PREFIX}/active/rollback', json={'expected_active': None})
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'no_previous'


def test_template_is_read_only(app_client: TestClient) -> None:
    """``examples/prompt_packs/vehicle_wheel.json`` ships as a template."""
    r = app_client.get(f'{PREFIX}?include_templates=true')
    assert r.status_code == 200, r.text
    template_names = {t['name'] for t in r.json()['templates']}
    assert 'vehicle_wheel' in template_names

    r = app_client.put(f'{PREFIX}/vehicle_wheel', json={'expected_revision': 0, 'body': _body()})
    assert r.status_code == 403, r.text
    assert r.json()['detail']['error'] == 'read_only'

    r = app_client.delete(f'{PREFIX}/vehicle_wheel', params={'expected_revision': 0})
    assert r.status_code == 403, r.text


def test_clone_of_builtin(app_client: TestClient) -> None:
    r = app_client.post(
        f'{PREFIX}/{GENERIC_ITEM_PACK.name}/clone',
        json={'new_name': 'my_clone', 'source': 'builtin'},
    )
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc['source'] == 'stored'
    assert doc['revision'] == 1
    assert doc['cloned_from'].startswith('default:')
    assert doc['body']['class_system'] == GENERIC_ITEM_PACK.class_system


def test_clone_onto_taken_name_409s(app_client: TestClient) -> None:
    app_client.post(PREFIX, json={'name': 'taken', 'body': _body()})
    r = app_client.post(f'{PREFIX}/{GENERIC_ITEM_PACK.name}/clone', json={'new_name': 'taken'})
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'name_conflict'


def test_revisions_list_and_get_by_revision(app_client: TestClient) -> None:
    app_client.post(PREFIX, json={'name': 'my_pack', 'body': _body()})
    body2 = _body()
    body2['class_system'] = 'changed'
    app_client.put(f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': body2})

    r = app_client.get(f'{PREFIX}/my_pack/revisions')
    assert r.status_code == 200, r.text
    revisions = [rv['revision'] for rv in r.json()['revisions']]
    assert sorted(revisions) == [1, 2]

    r1 = app_client.get(f'{PREFIX}/my_pack/revisions/1')
    assert r1.status_code == 200
    assert r1.json()['body']['class_system'] == GENERIC_ITEM_PACK.class_system
    r2 = app_client.get(f'{PREFIX}/my_pack/revisions/2')
    assert r2.json()['body']['class_system'] == 'changed'


def test_etag_header_on_get(app_client: TestClient) -> None:
    app_client.post(PREFIX, json={'name': 'my_pack', 'body': _body()})
    r = app_client.get(f'{PREFIX}/my_pack')
    assert r.status_code == 200
    assert r.headers['etag'] == '"prompt_pack:my_pack:1"'


def test_from_project_clone_is_read_only_and_never_writes_to_source(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one legitimate cross-project read in this wave: cloning a pack
    out of another project must never write anything to that project's
    own config-store index, even though both projects share the same
    fake OpenSearch transport in this test."""
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
            body = {**GENERIC_ITEM_PACK.to_dict(), 'name': 'other_pack'}
            await save_config(
                fake_os,
                idx,
                kind='prompt_pack',
                name='other_pack',
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
        f'{PREFIX}/other_pack/clone',
        json={'new_name': 'cloned_from_other', 'from_project': 'other'},
    )
    assert r.status_code == 201, r.text
    assert r.json()['cloned_from'].startswith('other:other_pack@')

    # The source project's own index is byte-for-byte unchanged.
    after = dict(fake_os._docs.get(other_index, {}))
    assert before == after

    # And the clone really landed in the (default) target project.
    r_get = app_client.get(f'{PREFIX}/cloned_from_other')
    assert r_get.status_code == 200
    assert r_get.json()['body']['class_system'] == GENERIC_ITEM_PACK.class_system
