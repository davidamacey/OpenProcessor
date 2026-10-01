"""The wire contract of the VLM surfaces (W9, §7.8): everything a UI needs
is served (labels, choices, warnings), and what one project can see of the
registry never tells it about another project."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import pytest


from curation.conftest import ACTIVE, GLOBAL_VLM, SCOPED
from src.main import app


EXTERNAL = {'base_url': 'https://api.example.com/v1', 'allow_external': True}


def _enum_values(schema: dict, name: str) -> set[str]:
    node = schema['components']['schemas'][name]
    return set(node.get('enum') or ())


def _property_enum(schema: dict, model: str, prop: str) -> set[str]:
    spec = schema['components']['schemas'][model]['properties'][prop]
    if '$ref' in spec:
        return _enum_values(schema, spec['$ref'].rsplit('/', 1)[1])
    options = spec.get('anyOf') or [spec]
    found: set[str] = set()
    for option in options:
        found |= set(option.get('enum') or ())
    return found


def test_every_served_enum_value_has_a_served_label(vlm_api) -> None:
    """The enums are Literals in OpenAPI; a client must never hardcode the
    words for them, so each value has a label in the listing."""
    schema = app.openapi()
    listing = vlm_api.client.get(GLOBAL_VLM).json()
    labels = listing['labels']
    assert _property_enum(schema, 'VlmEndpointSummary', 'status') == set(labels['status'])
    assert _property_enum(schema, 'VlmEndpointSummary', 'source') == set(labels['source'])
    assert _property_enum(schema, 'VlmEndpointSummary', 'locality') <= set(labels['locality'])
    catalog = vlm_api.client.get('/curation/vlm/catalog').json()
    assert _property_enum(schema, 'VlmCatalogEntry', 'status') == set(catalog['labels']['status'])
    assert all(labels[k] and all(labels[k].values()) for k in ('status', 'locality', 'source'))


def test_the_new_error_and_validation_codes_are_in_the_contract() -> None:
    schema = app.openapi()
    errors = _property_enum(schema, 'ConfigErrorDetail', 'error')
    codes = _property_enum(schema, 'ValidationIssue', 'code')
    for code in (
        'unknown_vlm',
        'in_use',
        'no_previous',
        'previous_deleted',
        'probe_busy',
        'no_local_vlm',
        'unknown_catalog_id',
        'vlm_catalog_does_not_fit',
        'vlm_external_not_acknowledged',
        'vlm_endpoint_unavailable',
        'vlm_not_configured',
    ):
        assert code in errors, code
    for code in (
        'vlm_url_invalid',
        'vlm_url_denied_internal_service',
        'vlm_url_denied_address',
        'vlm_external_not_acknowledged',
        'vlm_external_denied',
        'vlm_api_key_ref_invalid',
        'vlm_api_key_unresolved',
        'vlm_not_probed',
        'vlm_probe_failed',
        'vlm_max_images_exceeds_server',
        'vlm_context_too_small',
        'vlm_multi_box_unverified',
        'vlm_reads_text_unverified',
        'vlm_json_mode_off',
        'vlm_no_vision',
        'vlm_model_not_listed',
    ):
        assert code in codes, code


def test_the_listing_carries_everything_the_picker_shows(vlm_api) -> None:
    vlm_api.ready('local1')
    vlm_api.ready('ext', **EXTERNAL)
    listing = vlm_api.client.get(GLOBAL_VLM).json()
    by_name = {e['name']: e for e in listing['endpoints']}
    assert listing['external_policy'] == 'ack'
    local, ext = by_name['local1'], by_name['ext']
    assert (local['locality'], local['sends_images_externally'], local['warning']) == (
        'compose',
        False,
        None,
    )
    assert (ext['locality'], ext['sends_images_externally']) == ('external', True)
    assert 'api.example.com' in ext['warning']
    assert ext['status'] == 'ready'
    assert ext['last_probe_at']
    assert {'revision', 'etag', 'model', 'api_key_ref', 'api_key_present', 'active_in'} <= set(ext)


def test_methods_serves_the_vlm_axis_with_the_acknowledgement_rule(vlm_api) -> None:
    vlm_api.ready('local1')
    vlm_api.ready('ext', **EXTERNAL)
    assert vlm_api.activate('local1', expected_active=None).status_code == 200
    body = vlm_api.client.get(f'{SCOPED}/methods').json()
    assert 'vlm' in {a['axis'] for a in body['axes']}
    strategies = {s['id']: s for s in body['strategies'] if s['axis'] == 'vlm'}
    assert set(strategies) == {'local1', 'ext', 'off'}
    assert strategies['local1']['default'] is True
    assert strategies['ext']['default'] is False
    # what the run will enforce, served by the same function that enforces it
    assert strategies['ext']['sends_images_externally'] is True
    assert strategies['ext']['per_run_ack_required'] is True
    assert strategies['ext']['default_ack_recorded'] is False
    assert strategies['local1']['per_run_ack_required'] is False
    assert strategies['ext']['endpoint_status_label'] == 'Ready'


def test_the_acknowledged_default_needs_no_per_run_flag(vlm_api) -> None:
    vlm_api.ready('ext', **EXTERNAL)
    assert (
        vlm_api.activate('ext', expected_active=None, acknowledge_external=True).status_code == 200
    )
    body = vlm_api.client.get(f'{SCOPED}/methods').json()
    (ext,) = [s for s in body['strategies'] if s['axis'] == 'vlm' and s['id'] == 'ext']
    assert ext['default'] is True
    assert ext['default_ack_recorded'] is True
    assert ext['per_run_ack_required'] is False


def test_the_models_listing_names_only_the_bound_project_as_running_an_endpoint(vlm_api) -> None:
    """Which OTHER projects use an endpoint is deployment-wide information
    and lives on the global listing, not on a project-scoped one."""
    vlm_api.ready('one')
    vlm_api.ready('two')
    assert vlm_api.activate('one', expected_active=None).status_code == 200
    rows = {
        r['name']: r
        for r in vlm_api.client.get(f'{SCOPED}/models/status').json()['models']
        if r['kind'] == 'vlm'
    }
    assert rows['one']['active'] is True
    assert rows['one']['active_in'] == ['default']
    assert rows['two']['active'] is False
    assert rows['two']['active_in'] == []
    assert next(iter(rows)) == 'one'  # the active endpoint first
    assert rows['one']['unloadable'] is False


def test_a_credentialed_env_url_is_never_listed(vlm_api, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_VLM_URL', 'http://user:pw@vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'served')
    rows = [
        r
        for r in vlm_api.client.get(f'{SCOPED}/models/status').json()['models']
        if r['kind'] == 'vlm'
    ]
    assert rows
    assert all('pw' not in (r['endpoint'] or '') for r in rows)


def test_the_config_vocabulary_lists_every_endpoint_with_its_locality(vlm_api) -> None:
    vlm_api.ready('local1')
    vlm_api.ready('ext', **EXTERNAL)
    vocabulary = vlm_api.client.get(f'{SCOPED}/config/vocabulary').json()['vlm']
    endpoints = {e['name']: e for e in vocabulary['endpoints']}
    assert set(endpoints) == {'local1', 'ext'}
    assert endpoints['ext']['sends_images_externally'] is True
    assert endpoints['ext']['warning']
    assert endpoints['local1']['resolved_model'] == 'org/real-model'
    assert vocabulary['active']['name'] is None


def test_the_region_vocabulary_offers_each_endpoints_resolved_model_as_a_verifier(
    vlm_api, reference_region_profile: None
) -> None:
    vlm_api.probe_record[0] = vlm_api.probe_record[0].model_copy(update={'root': 'org/one-model'})
    vlm_api.ready('one')
    body = vlm_api.client.get(f'{SCOPED}/regions/vocabulary').json()
    verifiers = [d for d in body['detectors'] if d['role'] == 'verifier']
    assert [v['id'] for v in verifiers] == ['org/one-model']
    assert verifiers[0]['filterable'] is False


def test_setting_the_default_vlm_accepts_only_advertised_ids(vlm_api) -> None:
    vlm_api.ready('one')
    rejected = vlm_api.client.put(f'{SCOPED}/settings', json={'defaults': {'vlm': 'nope'}})
    assert rejected.status_code == 422
    detail = rejected.json()['detail']
    assert detail['error'] == 'unknown_vlm'
    assert detail['axis'] == 'vlm'
    assert set(detail['valid_ids']) == {'one', 'off'}
    ok = vlm_api.client.put(f'{SCOPED}/settings', json={'defaults': {'vlm': 'one'}})
    assert ok.status_code == 200, ok.text
    assert vlm_api.client.get(f'{SCOPED}/settings').json()['defaults']['vlm'] == 'one'
    assert vlm_api.active()['active']['name'] == 'one'
    off = vlm_api.client.put(f'{SCOPED}/settings', json={'defaults': {'vlm': 'off'}})
    assert off.status_code == 200
    assert vlm_api.active()['source'] == 'off'


def test_the_active_response_is_the_shared_activation_shape(vlm_api) -> None:
    vlm_api.ready('one')
    body = vlm_api.client.get(f'{ACTIVE}/active').json()
    assert body['axis'] == 'vlm'
    assert set(body) >= {'active', 'source', 'previous', 'config_revision', 'applied', 'stale'}
    # never activated and no built-in configured: nothing runs, and the source
    # says it is the built-in default (not an explicit off)
    assert body['active'] == {'name': None, 'revision': None}
    assert body['source'] == 'env'


def test_every_new_route_is_in_the_openapi_document() -> None:
    paths = app.openapi()['paths']
    for path in (
        '/curation/vlm/endpoints',
        '/curation/vlm/endpoints/schema',
        '/curation/vlm/endpoints/validate',
        '/curation/vlm/endpoints/{name}',
        '/curation/vlm/endpoints/{name}/revisions',
        '/curation/vlm/endpoints/{name}/revisions/{revision}',
        '/curation/vlm/endpoints/{name}/clone',
        '/curation/vlm/endpoints/{name}/probe',
        '/curation/vlm/catalog',
        '/curation/vlm/local',
        '/curation/vlm/local/select',
        '/curation/projects/{project}/vlm/endpoints/active',
        '/curation/projects/{project}/vlm/endpoints/active/rollback',
        '/curation/projects/{project}/vlm/endpoints/deactivate',
        '/curation/projects/{project}/vlm/endpoints/{name}/activate',
    ):
        assert path in paths, path
