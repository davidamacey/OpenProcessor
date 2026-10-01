"""``/vlm/catalog`` and ``/vlm/local*`` (W9.7): the API records which local
model is DESIRED and never restarts anything; the host CLI applies it."""

from __future__ import annotations

import pytest

from curation.conftest import GLOBAL_VLM, good_probe
from src.services.labeling.vlm_catalog import catalog_entry


CATALOG = '/curation/vlm/catalog'
LOCAL = '/curation/vlm/local'
SELECT = '/curation/vlm/local/select'


@pytest.fixture
def local_api(vlm_api, monkeypatch: pytest.MonkeyPatch):
    """A stack with an in-compose vLLM: ``env`` names the endpoint that
    reaches it and the card is 48 GB."""
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'local-vlm')
    monkeypatch.setenv('OP_LOCAL_VLM_ENDPOINT', 'env')
    monkeypatch.setenv('OP_LOCAL_VLM_GPU_TOTAL_MIB', str(48 * 1024))
    return vlm_api


def _probe_env(api, *, root: str) -> None:
    api.probe_record[0] = good_probe(root=root, models_listed=['local-vlm'])
    response = api.client.post(f'{GLOBAL_VLM}/env/probe')
    assert response.status_code == 200, response.text


def test_without_an_in_compose_vlm_selecting_is_a_409(vlm_api) -> None:
    status = vlm_api.client.get(LOCAL).json()
    assert status['configured'] is False
    response = vlm_api.client.post(SELECT, json={'catalog_id': 'gemma-4-e4b'})
    assert response.status_code == 409
    assert response.json()['detail']['error'] == 'no_local_vlm'


def test_select_records_the_desire_and_names_the_host_command(local_api) -> None:
    gemma = catalog_entry('gemma-4-e4b')
    assert gemma is not None
    _probe_env(local_api, root=gemma.hf_repo)
    response = local_api.client.post(SELECT, json={'catalog_id': 'qwen3-vl-4b'})
    assert response.status_code == 202, response.text
    body = response.json()
    assert body['desired']['catalog_id'] == 'qwen3-vl-4b'
    assert body['desired']['command'] == 'openprocessor vlm use qwen3-vl-4b'
    assert body['restart_required'] is True
    assert body['can_restart_from_api'] is False
    assert body['served']['catalog_id'] == 'gemma-4-e4b'


def test_serving_flips_only_after_the_host_probe_reports_the_new_root(local_api) -> None:
    gemma = catalog_entry('gemma-4-e4b')
    qwen = catalog_entry('qwen3-vl-4b')
    assert gemma is not None
    assert qwen is not None
    _probe_env(local_api, root=gemma.hf_repo)
    assert local_api.client.post(SELECT, json={'catalog_id': qwen.id}).status_code == 202
    # asking again changes nothing about what is served
    entries = {e['id']: e for e in local_api.client.get(CATALOG).json()['entries']}
    assert entries[gemma.id]['serving'] is True
    assert entries[qwen.id]['serving'] is False
    assert entries[qwen.id]['desired'] is True

    _probe_env(local_api, root=qwen.hf_repo)  # what `openprocessor vlm use` does
    after = local_api.client.get(CATALOG).json()
    assert after['local']['restart_required'] is False
    served = {e['id']: e['serving'] for e in after['entries']}
    assert served[qwen.id] is True
    assert served[gemma.id] is False


def test_clearing_the_request(local_api) -> None:
    assert local_api.client.post(SELECT, json={'catalog_id': 'qwen3-vl-4b'}).status_code == 202
    cleared = local_api.client.delete(SELECT)
    assert cleared.status_code == 200
    assert cleared.json()['desired'] is None
    assert local_api.client.get(LOCAL).json()['restart_required'] is False


def test_an_unknown_id_is_refused_with_the_valid_ones(local_api) -> None:
    response = local_api.client.post(SELECT, json={'catalog_id': 'not-a-model'})
    assert response.status_code == 422
    detail = response.json()['detail']
    assert detail['error'] == 'unknown_catalog_id'
    assert 'gemma-4-e4b' in detail['valid_ids']


def test_a_model_that_does_not_fit_is_refused_unless_forced(
    local_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_LOCAL_VLM_GPU_TOTAL_MIB', str(12 * 1024))
    response = local_api.client.post(SELECT, json={'catalog_id': 'gemma-4-e4b'})
    assert response.status_code == 422
    assert response.json()['detail']['error'] == 'vlm_catalog_does_not_fit'
    forced = local_api.client.post(SELECT, json={'catalog_id': 'gemma-4-e4b', 'force': True})
    assert forced.status_code == 202


def test_fits_is_unknown_when_the_card_size_is_not_configured(
    local_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_LOCAL_VLM_GPU_TOTAL_MIB')
    entries = local_api.client.get(CATALOG).json()['entries']
    assert {e['fits'] for e in entries} == {None}
    assert local_api.client.post(SELECT, json={'catalog_id': 'gemma-4-e4b'}).status_code == 202


def test_the_catalog_serves_its_own_labels_and_choices(local_api) -> None:
    body = local_api.client.get(CATALOG).json()
    assert set(body['labels']['status']) == {'tested', 'to_verify'}
    for entry in body['entries']:
        assert entry['choice'] == {'id': entry['id'], 'label': entry['hf_repo']}
        assert entry['license_url'].startswith('https://huggingface.co/')
        assert entry['status'] in body['labels']['status']


def test_the_desire_survives_a_cold_process(local_api) -> None:
    """It lives in the registry, not in process memory: a second process
    (or a restarted API) reads the same request."""
    from src.services.config_store.global_store import reset_global_config_store

    assert local_api.client.post(SELECT, json={'catalog_id': 'qwen3-vl-4b'}).status_code == 202
    reset_global_config_store()
    assert local_api.client.get(LOCAL).json()['desired']['catalog_id'] == 'qwen3-vl-4b'
