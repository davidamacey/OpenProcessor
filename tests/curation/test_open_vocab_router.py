"""Wave 3: the ``/open_vocab*`` config axis -- CRUD lifecycle, validation,
activation gate, rollback, clone, schema."""

from __future__ import annotations

from typing import Any

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch


PREFIX = '/curation/projects/default/open_vocab'


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


@pytest.fixture
def segmenter(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {'status': 'ready'}

    async def _health() -> tuple[str, str | None]:
        return state['status'], None if state['status'] == 'ready' else 'down'

    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://segmenter.invalid:8000')
    monkeypatch.setattr('src.routers.curation.region_profiles._segmenter_health_fn', _health)
    return state


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, segmenter: dict[str, Any]) -> TestClient:
    from unittest.mock import AsyncMock

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    monkeypatch.setattr(
        'src.config.ingest_profiles.ingest_primary_profile',
        lambda: type('P', (), {'detector_model': ''})(),
    )
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    fake = FakeConfigOpenSearch()
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        'display_name': 'Street',
        'targets': [
            {'prompt': 'traffic cone', 'class_name': 'traffic cone'},
            {'prompt': 'cup', 'class_name': 'cup', 'enabled': False},
        ],
    }
    body.update(overrides)
    return body


def _codes(resp: Any, key: str = 'errors') -> list[str]:
    return [i['code'] for i in resp.json()['detail']['report'][key]]


def test_full_lifecycle(client: TestClient) -> None:
    r = client.post(PREFIX, json={'name': 'street', 'description': 'd1', 'body': _body()})
    assert r.status_code == 201, r.text
    assert r.json()['revision'] == 1
    assert r.json()['body']['targets'][0]['min_score'] == 0.5

    r = client.put(
        f'{PREFIX}/street', json={'expected_revision': 0, 'description': 'x', 'body': _body()}
    )
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'revision_conflict'
    r = client.put(
        f'{PREFIX}/street', json={'expected_revision': 1, 'description': 'x', 'body': _body()}
    )
    assert r.status_code == 200
    assert r.json()['revision'] == 2

    r = client.post(f'{PREFIX}/street/activate', json={'expected_active': None})
    assert r.status_code == 200, r.text
    assert r.json()['active'] == {'name': 'street', 'revision': 2}
    assert client.get(f'{PREFIX}/active').json()['axis'] == 'open_vocab'
    assert client.get(PREFIX).json()['active']['name'] == 'street'

    r = client.delete(f'{PREFIX}/street', params={'expected_revision': 2})
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'in_use'

    active = client.get(f'{PREFIX}/active').json()['active']
    r = client.post(f'{PREFIX}/deactivate', json={'expected_active': active})
    assert r.json()['active']['name'] is None
    assert client.delete(f'{PREFIX}/street', params={'expected_revision': 2}).status_code == 204
    assert client.get(f'{PREFIX}/street').status_code == 404


def test_activate_then_rollback(client: TestClient) -> None:
    client.post(PREFIX, json={'name': 'one', 'body': _body()})
    client.post(PREFIX, json={'name': 'two', 'body': _body()})
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


def test_activation_requires_a_reachable_segmenter_and_force_bypasses(
    client: TestClient, segmenter: dict[str, Any]
) -> None:
    client.post(PREFIX, json={'name': 'street', 'body': _body()})
    segmenter['status'] = 'down'
    r = client.post(f'{PREFIX}/street/activate', json={})
    assert r.status_code == 422
    assert _codes(r) == ['segmenter_unreachable']
    assert client.get(f'{PREFIX}/active').json()['active']['name'] is None

    r = client.post(f'{PREFIX}/street/activate', json={'force': True})
    assert r.status_code == 200, r.text


def test_activation_needs_an_enabled_target_and_force_does_not_bypass_it(
    client: TestClient,
) -> None:
    body = _body(targets=[{'prompt': 'cup', 'class_name': 'cup', 'enabled': False}])
    assert client.post(PREFIX, json={'name': 'idle', 'body': body}).status_code == 201
    r = client.post(f'{PREFIX}/idle/activate', json={'force': True})
    assert r.status_code == 422
    assert _codes(r) == ['open_vocab_no_enabled_targets']


def test_a_stored_set_with_no_segmenter_configured_saves_but_does_not_activate(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_SEGMENTER_URL')
    r = client.post(PREFIX, json={'name': 'street', 'body': _body()})
    assert r.status_code == 201
    assert 'segmenter_not_configured' in [w['code'] for w in r.json()['validation']['warnings']]
    r = client.post(f'{PREFIX}/street/activate', json={})
    assert r.status_code == 422
    assert _codes(r) == ['segmenter_not_configured']


@pytest.mark.parametrize(
    ('targets', 'extra', 'code'),
    [
        ([{'prompt': '', 'class_name': 'cup'}], {}, 'segmenter_prompt_empty'),
        ([{'prompt': 'a\nb', 'class_name': 'cup'}], {}, 'segmenter_prompt_multiline'),
        (
            [{'prompt': 'cup', 'class_name': 'cup'}, {'prompt': 'Cup', 'class_name': 'CUP'}],
            {},
            'open_vocab_duplicate_target',
        ),
        ([{'prompt': 'cup', 'class_name': ' cup'}], {}, 'open_vocab_class_name_invalid'),
        ([{'prompt': 'cup', 'min_score': 1.5}], {}, 'open_vocab_field_range'),
        (
            [{'prompt': 'cup', 'min_area_frac': 0.5, 'max_area_frac': 0.2}],
            {},
            'open_vocab_field_range',
        ),
        ([{'prompt': 'cup'}], {'dedup_iou': -0.1}, 'open_vocab_field_range'),
        (
            [{'prompt': f'p{i}', 'class_name': f'c{i}'} for i in range(9)],
            {},
            'open_vocab_too_many_targets',
        ),
        (
            [{'prompt': 'cup', 'bogus': 1}],
            {},
            'open_vocab_field_invalid',
        ),
    ],
)
def test_validation_errors(
    client: TestClient, targets: list[dict[str, Any]], extra: dict[str, Any], code: str
) -> None:
    body = _body(targets=targets, **extra)
    r = client.post(f'{PREFIX}/validate', json={'name': 'x-set', 'body': body})
    assert code in [e['code'] for e in r.json()['errors']], r.text
    assert r.json()['ok'] is False


def test_cap_is_configurable_within_the_ceiling(client: TestClient) -> None:
    targets = [{'prompt': f'p{i}', 'class_name': f'c{i}'} for i in range(9)]
    ok = client.post(
        f'{PREFIX}/validate',
        json={'name': 'big', 'body': _body(targets=targets, max_enabled_targets=9)},
    )
    assert ok.json()['ok'] is True
    over = client.post(
        f'{PREFIX}/validate',
        json={'name': 'big', 'body': _body(targets=targets, max_enabled_targets=99)},
    )
    assert 'open_vocab_field_range' in [e['code'] for e in over.json()['errors']]


def test_target_named_like_a_detector_class_is_a_warning(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        'src.config.ingest_profiles.ingest_primary_profile',
        lambda: type('P', (), {'detector_model': 'det'})(),
    )
    monkeypatch.setattr('src.utils.class_names.get_class_names', lambda _m: {0: 'Cup', 1: 'person'})
    r = client.post(
        f'{PREFIX}/validate',
        json={'name': 'dup', 'body': _body(targets=[{'prompt': 'mug', 'class_name': 'cup'}])},
    )
    assert r.json()['ok'] is True
    assert 'open_vocab_detector_class' in [w['code'] for w in r.json()['warnings']]


def test_discovery_target_without_a_class_name_is_valid(client: TestClient) -> None:
    r = client.post(
        f'{PREFIX}/validate', json={'name': 'disc', 'body': _body(targets=[{'prompt': 'barrel'}])}
    )
    assert r.json()['ok'] is True


@pytest.mark.parametrize('name', ['A', 'x', 'off', 'a b'])
def test_bad_names_are_rejected_on_create(client: TestClient, name: str) -> None:
    r = client.post(PREFIX, json={'name': name, 'body': _body()})
    assert r.status_code == 422
    assert _codes(r)[0] in ('open_vocab_name_invalid', 'open_vocab_name_reserved')


def test_name_conflict(client: TestClient) -> None:
    client.post(PREFIX, json={'name': 'street', 'body': _body()})
    r = client.post(PREFIX, json={'name': 'street', 'body': _body()})
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'name_conflict'


def test_template_is_read_only_and_clonable(client: TestClient) -> None:
    listing = client.get(PREFIX, params={'include_templates': 'true'}).json()
    assert [t['name'] for t in listing['templates']] == ['street_objects']
    assert listing['sets'] == []
    assert client.put(
        f'{PREFIX}/street_objects', json={'expected_revision': 1, 'body': _body()}
    ).status_code in (403, 404)
    assert client.post(f'{PREFIX}/street_objects/activate', json={}).status_code == 403
    r = client.post(f'{PREFIX}/street_objects/clone', json={'new_name': 'mine'})
    assert r.status_code == 201, r.text
    assert r.json()['cloned_from'].endswith(':street_objects@-')
    assert r.json()['body']['targets'][0]['prompt'] == 'traffic cone'
    assert (
        client.post(f'{PREFIX}/street_objects/clone', json={'new_name': 'mine'}).status_code == 409
    )


def test_revisions_listing_and_fetch(client: TestClient) -> None:
    client.post(PREFIX, json={'name': 'street', 'body': _body()})
    client.put(f'{PREFIX}/street', json={'expected_revision': 1, 'body': _body(display_name='v2')})
    revs = client.get(f'{PREFIX}/street/revisions').json()['revisions']
    assert [r['revision'] for r in revs] == [2, 1]
    old = client.get(f'{PREFIX}/street/revisions/1').json()
    assert old['body']['display_name'] == 'Street'
    assert client.get(f'{PREFIX}/street/revisions/9').status_code == 404


def test_schema_describes_every_field_with_ranges(client: TestClient) -> None:
    schema = client.get(f'{PREFIX}/schema').json()
    rows = {(f['scope'], f['field']): f for f in schema['fields']}
    assert rows[('target', 'min_score')]['max'] == 1.0
    assert rows[('target', 'prompt')]['type'] == 'string'
    assert rows[('set', 'max_enabled_targets')]['default'] == 8
    assert rows[('set', 'run_on_ingest')]['default'] is False
    assert rows[('tier3_hit_rate', 'enabled')]['default'] is False
    assert rows[('gating', 'tier2_vlm_precheck')]['default'] is False
    assert schema['max_enabled_targets_ceiling'] == 32
