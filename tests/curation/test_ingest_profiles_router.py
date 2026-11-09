"""The ``/ingest_profiles*`` config axis: CRUD lifecycle, validation, the Triton
activation gate, rollback, clone, schema."""

from __future__ import annotations

from typing import Any

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._ingest_profile_world import Pool, body


PREFIX = '/curation/projects/default/ingest_profiles'


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


@pytest.fixture
def pool(monkeypatch: pytest.MonkeyPatch) -> Pool:
    fake_pool = Pool()
    monkeypatch.setattr('src.app.resources.get_async_triton_pool', lambda: fake_pool)
    return fake_pool


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, pool: Pool) -> TestClient:
    from unittest.mock import AsyncMock

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    del pool
    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    fake = object()  # the OpenSearch client: the ingest-profile routes never read it
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _codes(resp: Any, key: str = 'errors') -> list[str]:
    return [i['code'] for i in resp.json()['detail']['report'][key]]


def test_full_lifecycle(client: TestClient) -> None:
    r = client.post(PREFIX, json={'name': 'birds', 'description': 'd1', 'body': body()})
    assert r.status_code == 201, r.text
    assert r.json()['revision'] == 1
    assert r.json()['body']['primary']['version'] == '1'

    r = client.put(f'{PREFIX}/birds', json={'expected_revision': 0, 'body': body()})
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'revision_conflict'
    r = client.put(
        f'{PREFIX}/birds',
        json={'expected_revision': 1, 'body': body(primary={'confidence_floor': 0.6})},
    )
    assert r.status_code == 200, r.text
    assert r.json()['revision'] == 2
    revisions = client.get(f'{PREFIX}/birds/revisions').json()['revisions']
    assert [x['revision'] for x in revisions] == [2, 1]
    first = client.get(f'{PREFIX}/birds/revisions/1').json()
    assert first['body']['primary']['confidence_floor'] is None

    r = client.post(f'{PREFIX}/birds/activate', json={})
    assert r.status_code == 200, r.text
    assert r.json()['active'] == {'name': 'birds', 'revision': 2}
    assert r.json()['axis'] == 'ingest_profile'
    assert client.get(PREFIX).json()['active']['name'] == 'birds'

    r = client.delete(f'{PREFIX}/birds', params={'expected_revision': 2})
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'in_use'

    active = client.get(f'{PREFIX}/active').json()['active']
    r = client.post(f'{PREFIX}/deactivate', json={'expected_active': active})
    assert r.json()['active']['name'] is None
    assert client.delete(f'{PREFIX}/birds', params={'expected_revision': 2}).status_code == 204
    assert client.get(f'{PREFIX}/birds').status_code == 404


def test_activate_then_rollback(client: TestClient) -> None:
    client.post(PREFIX, json={'name': 'one', 'body': body('m1')})
    client.post(PREFIX, json={'name': 'two', 'body': body('m2')})
    client.post(f'{PREFIX}/one/activate', json={})
    active = client.get(f'{PREFIX}/active').json()['active']
    client.post(f'{PREFIX}/two/activate', json={'expected_active': active})
    active = client.get(f'{PREFIX}/active').json()['active']
    r = client.post(f'{PREFIX}/active/rollback', json={'expected_active': active})
    assert r.status_code == 200, r.text
    assert r.json()['active']['name'] == 'one'


def test_rollback_with_nothing_previous_is_409(client: TestClient) -> None:
    r = client.post(f'{PREFIX}/active/rollback', json={})
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'no_previous'


def test_activation_requires_a_servable_model_and_force_does_not_bypass(
    client: TestClient, pool: Pool
) -> None:
    client.post(PREFIX, json={'name': 'birds', 'body': body()})
    pool.ready['birds'] = False
    r = client.post(f'{PREFIX}/birds/activate', json={'force': True})
    assert r.status_code == 422
    assert _codes(r) == ['detector_model_not_ready']
    assert client.get(f'{PREFIX}/active').json()['active']['name'] is None

    pool.ready['birds'] = True
    pool.outputs['birds'] = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
    r = client.post(f'{PREFIX}/birds/activate', json={})
    assert r.status_code == 422
    assert _codes(r) == ['detector_model_unusable']


def test_a_primary_whose_output0_is_not_the_fused_tensor_is_refused_with_the_export_command(
    client: TestClient, pool: Pool
) -> None:
    client.post(PREFIX, json={'name': 'birds', 'body': body()})
    pool.shapes['birds'] = (-1, 5, 8400)
    r = client.post(f'{PREFIX}/birds/activate', json={'force': True})
    assert r.status_code == 422
    assert _codes(r) == ['detector_model_unusable']
    assert 'export' in r.json()['detail']['report']['errors'][0]['message']
    assert client.get(f'{PREFIX}/active').json()['active']['name'] is None


def test_a_primary_serving_the_fused_tensor_activates(client: TestClient, pool: Pool) -> None:
    client.post(PREFIX, json={'name': 'birds', 'body': body()})
    pool.shapes['birds'] = (300, 6)
    assert client.post(f'{PREFIX}/birds/activate', json={}).status_code == 200


def test_the_secondary_is_checked_against_the_raw_output_contract(
    client: TestClient, pool: Pool
) -> None:
    both = body(secondary={'model': 'cls'})
    client.post(PREFIX, json={'name': 'both', 'body': both})
    pool.outputs['cls'] = ['num_dets']
    r = client.post(f'{PREFIX}/both/activate', json={})
    assert r.status_code == 422
    assert [i['field'] for i in r.json()['detail']['report']['errors']] == ['secondary.model']
    pool.outputs['cls'] = ['output0']
    assert client.post(f'{PREFIX}/both/activate', json={}).status_code == 200


def test_a_pool_that_cannot_be_reached_blocks_activation_unless_forced(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _down() -> Any:
        msg = 'no pool'
        raise RuntimeError(msg)

    monkeypatch.setattr('src.app.resources.get_async_triton_pool', _down)
    client.post(PREFIX, json={'name': 'birds', 'body': body()})
    r = client.post(f'{PREFIX}/birds/activate', json={})
    assert r.status_code == 422
    assert _codes(r) == ['triton_unreachable']
    assert r.json()['detail']['report']['force_allowed'] is True
    # an outage may be stale: force bypasses it, and only it
    assert client.post(f'{PREFIX}/birds/activate', json={'force': True}).status_code == 200


def test_save_accepts_a_model_that_is_not_loaded_yet(client: TestClient, pool: Pool) -> None:
    pool.default_ready = False
    assert client.post(PREFIX, json={'name': 'later', 'body': body()}).status_code == 201


def test_names_and_bodies_are_validated(client: TestClient) -> None:
    r = client.post(PREFIX, json={'name': 'Bad Name', 'body': body()})
    assert r.status_code == 422
    assert _codes(r) == ['ingest_profile_name_invalid']
    r = client.post(PREFIX, json={'name': 'off', 'body': body()})
    assert 'ingest_profile_name_reserved' in _codes(r)
    client.post(PREFIX, json={'name': 'birds', 'body': body()})
    assert client.post(PREFIX, json={'name': 'birds', 'body': body()}).status_code == 409
    r = client.post(PREFIX, json={'name': 'x1', 'body': {'primary': {'model': 'm', 'extra': 1}}})
    assert r.status_code == 422
    small = {'primary': {'model': 'm', 'input_size': 8}}
    assert client.post(PREFIX, json={'name': 'x2', 'body': small}).status_code == 422


def test_templates_are_read_only_and_cloneable(client: TestClient) -> None:
    listing = client.get(PREFIX, params={'include_templates': True}).json()
    assert 'coco80' in [t['name'] for t in listing['templates']]
    put = client.put(f'{PREFIX}/coco80', json={'expected_revision': 0, 'body': body()})
    assert put.status_code == 403
    assert client.post(f'{PREFIX}/coco80/activate', json={}).status_code == 403
    cloned = client.post(f'{PREFIX}/coco80/clone', json={'new_name': 'mine'})
    assert cloned.status_code == 201, cloned.text
    assert cloned.json()['body']['primary']['model'] == 'yolov11_small_trt_end2end'
    assert client.post(f'{PREFIX}/mine/activate', json={}).status_code == 200


def test_validate_route_and_schema(client: TestClient, pool: Pool) -> None:
    pool.ready['birds'] = False
    ok = client.post(f'{PREFIX}/validate', json={'name': 'birds', 'body': body()})
    assert ok.json()['ok'] is True
    gated = client.post(
        f'{PREFIX}/validate', params={'for_activation': True}, json={'body': body()}
    )
    assert gated.json()['ok'] is False
    fields = client.get(f'{PREFIX}/schema').json()['fields']
    assert {f['scope'] for f in fields} == {'primary', 'secondary'}
    assert {f['field'] for f in fields} >= {'model', 'confidence_floor', 'batch_limit'}
