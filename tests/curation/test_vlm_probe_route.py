"""``POST /vlm/endpoints/validate?probe=true`` and ``/{name}/probe``: the
probe resolves the endpoint's key and sends it, so it is gated like a use
(W9 review M1, M2, P2) and records its result per revision (M3, M4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.conftest import ACTIVE, GLOBAL_VLM, good_probe


if TYPE_CHECKING:
    from pathlib import Path


def _secret(tmp_path: Path, slug: str, value: str) -> None:
    directory = tmp_path / 'secrets'
    directory.mkdir(exist_ok=True)
    (directory / slug).write_text(value)


def _validate(vlm_api, body: dict, **top):
    return vlm_api.lenient.post(f'{GLOBAL_VLM}/validate?probe=true', json={'body': body, **top})


def test_an_unacknowledged_external_endpoint_is_never_probed_with_a_secret(vlm_api, tmp_path):
    _secret(tmp_path, 'prod_openai', 'sk-REALKEY-123')
    vlm_api.dns['evil.example.net'] = ['93.184.216.34']
    body = vlm_api.body(base_url='https://evil.example.net/v1', api_key_ref='secret:prod_openai')
    response = _validate(vlm_api, body)
    assert response.status_code == 200
    codes = [e['code'] for e in response.json()['validation']['errors']]
    assert 'vlm_external_not_acknowledged' in codes
    assert response.json()['probe'] is None
    assert vlm_api.probe_keys == []


def test_the_saved_endpoint_probe_route_is_gated_the_same_way(vlm_api, tmp_path):
    _secret(tmp_path, 'prod_openai', 'sk-REALKEY-123')
    vlm_api.dns['evil.example.net'] = ['93.184.216.34']
    vlm_api.dns['evil.example.net'] = ['10.0.0.9']
    vlm_api.create('ext', base_url='https://evil.example.net/v1', api_key_ref='secret:prod_openai')
    # the host now resolves outside the deployment; the endpoint never acknowledged that
    vlm_api.dns['evil.example.net'] = ['93.184.216.34']
    from src.services.labeling import vlm_url_policy

    vlm_url_policy.reset_policy_caches()
    response = vlm_api.lenient.post(f'{GLOBAL_VLM}/ext/probe')
    assert response.status_code == 422
    assert vlm_api.probe_keys == []


def test_an_acknowledged_external_endpoint_is_probed(vlm_api, tmp_path):
    _secret(tmp_path, 'prod_openai', 'sk-REALKEY-123')
    vlm_api.dns['api.vendor.com'] = ['93.184.216.34']
    body = vlm_api.body(
        base_url='https://api.vendor.com/v1', api_key_ref='secret:prod_openai', allow_external=True
    )
    assert _validate(vlm_api, body).json()['probe'] is not None
    assert vlm_api.probe_keys == ['sk-REALKEY-123']


def test_the_deny_policy_blocks_the_probe(vlm_api, monkeypatch):
    monkeypatch.setenv('OP_VLM_EXTERNAL_POLICY', 'deny')
    vlm_api.dns['ext.example.com'] = ['93.184.216.34']
    body = vlm_api.body(base_url='https://ext.example.com/v1', allow_external=True)
    assert _validate(vlm_api, body).json()['probe'] is None
    assert vlm_api.probe_keys == []


@pytest.mark.parametrize('cap', [0, 65, 20000, 100_000_000])
def test_an_out_of_range_body_is_refused_before_any_probe_work(vlm_api, cap):
    response = _validate(vlm_api, vlm_api.body(max_images_per_call=cap))
    codes = [e['code'] for e in response.json()['validation']['errors']]
    assert 'vlm_field_range' in codes
    assert response.json()['probe'] is None
    assert vlm_api.probe_keys == []


def test_the_probe_clamps_the_image_cap_itself():
    """Defence in depth: ``probe_endpoint`` never builds more images than the
    documented maximum even if a caller skipped validation."""
    import asyncio

    import httpx

    from src.services.labeling.vlm_endpoint_body import FIELD_RANGES, VlmEndpointBody
    from src.services.labeling.vlm_endpoint_probe import probe_endpoint

    sent: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': 'm', 'root': 'm'}]})
        import json

        content = json.loads(request.content)['messages'][0]['content']
        sent.append(
            sum(1 for part in content if isinstance(part, dict) and part.get('type') == 'image_url')
        )
        return httpx.Response(
            200, json={'choices': [{'message': {'content': '{"c": "red"} red'}}], 'usage': {}}
        )

    body = VlmEndpointBody(base_url='http://vlm:8000/v1', model='m', max_images_per_call=500)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    asyncio.run(probe_endpoint(body, api_key=None, client=client))
    assert max(sent) <= FIELD_RANGES['max_images_per_call'][1]


def test_a_draft_named_env_with_the_env_key_ref_is_a_validation_error_not_a_500(
    vlm_api, monkeypatch
):
    monkeypatch.setenv('OP_VLM_API_KEY', 'sk-ENV')
    response = _validate(vlm_api, vlm_api.body(api_key_ref='env:OP_VLM_API_KEY'), name='env')
    assert response.status_code == 200
    assert response.json()['probe'] is None
    codes = [e['code'] for e in response.json()['validation']['errors']]
    assert 'vlm_api_key_ref_invalid' in codes
    assert vlm_api.probe_keys == []


def test_a_stored_endpoint_with_an_unresolvable_key_ref_probes_to_a_4xx(vlm_api, monkeypatch):
    from src.routers.curation import vlm_endpoints

    def refuse(*_a, **_k):
        from src.services.labeling.vlm_endpoints import VlmKeyRefInvalidError

        raise VlmKeyRefInvalidError('nope')

    vlm_api.create('alpha')
    monkeypatch.setattr(vlm_endpoints, 'resolve_api_key', refuse)
    response = vlm_api.lenient.post(f'{GLOBAL_VLM}/alpha/probe')
    assert response.status_code == 422


def test_probing_a_new_revision_keeps_the_running_revisions_probe(vlm_api):
    vlm_api.probe_record[0] = good_probe(root='org/r1', json_mode_supported=False)
    vlm_api.ready('alpha')
    assert vlm_api.activate('alpha', expected_active=None).status_code == 200
    saved = vlm_api.client.put(
        f'{GLOBAL_VLM}/alpha',
        json={'expected_revision': 1, 'body': vlm_api.body(model='other-model')},
    )
    assert saved.status_code == 200
    vlm_api.probe_record[0] = good_probe(
        root='org/r2', json_mode_supported=True, probed_at='2026-09-29T00:00:00+00:00'
    )
    vlm_api.probe('alpha')
    first = vlm_api.client.get(f'{GLOBAL_VLM}/alpha/revisions/1').json()['last_probe']
    second = vlm_api.client.get(f'{GLOBAL_VLM}/alpha/revisions/2').json()['last_probe']
    assert first['root'] == 'org/r1'
    assert first['json_mode_supported'] is False
    assert second['root'] == 'org/r2'


def test_a_draft_probe_is_recorded_only_for_the_revision_it_is(vlm_api):
    vlm_api.probe_record[0] = good_probe(root='org/r1')
    vlm_api.ready('alpha')
    vlm_api.client.put(
        f'{GLOBAL_VLM}/alpha',
        json={'expected_revision': 1, 'body': vlm_api.body(model='other-model')},
    )
    vlm_api.probe_record[0] = good_probe(root='org/r2', probed_at='2026-09-29T00:00:00+00:00')
    vlm_api.probe('alpha')
    vlm_api.probe_record[0] = good_probe(root='org/draft', probed_at='2026-09-30T00:00:00+00:00')
    # a draft that is neither revision records nothing and clobbers nothing
    assert _validate(vlm_api, vlm_api.body(model='third'), name='alpha').json()['probe']
    for revision, root in ((1, 'org/r1'), (2, 'org/r2')):
        probe = vlm_api.client.get(f'{GLOBAL_VLM}/alpha/revisions/{revision}').json()['last_probe']
        assert probe['root'] == root
    # a draft equal to the current revision is recorded under it
    assert _validate(vlm_api, vlm_api.body(model='other-model'), name='alpha').json()['probe']
    assert vlm_api.client.get(f'{GLOBAL_VLM}/alpha/revisions/2').json()['last_probe']['root'] == (
        'org/draft'
    )


def test_rolling_back_to_the_previous_revision_of_the_same_endpoint(vlm_api):
    vlm_api.ready('alpha')
    assert vlm_api.activate('alpha', expected_active=None).status_code == 200
    saved = vlm_api.client.put(
        f'{GLOBAL_VLM}/alpha',
        json={'expected_revision': 1, 'body': vlm_api.body(model='better-model')},
    )
    assert saved.status_code == 200
    vlm_api.probe('alpha')
    switched = vlm_api.activate('alpha', expected_active={'name': 'alpha', 'revision': 1})
    assert switched.status_code == 200, switched.text
    back = vlm_api.client.post(
        f'{ACTIVE}/active/rollback', json={'expected_active': {'name': 'alpha', 'revision': 2}}
    )
    assert back.status_code == 200, back.text
