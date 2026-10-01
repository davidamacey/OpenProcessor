"""Secrets by reference only (W9.9): a key is written to a file on the host,
an endpoint stores ``secret:<slug>``, and the value is read at call time. It
is never stored, served, logged or echoed -- and nothing but a plain file
inside the secrets mount can be a reference."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import pytest

from curation.conftest import ACTIVE, GLOBAL_VLM, SCOPED
from src.services.labeling.vlm_endpoints import (
    VlmKeyRefInvalidError,
    api_key_present,
    list_secret_refs,
    resolve_api_key,
)


if TYPE_CHECKING:
    from pathlib import Path


KEY = 'SECRETVALUE-9f3c1d7a'


@pytest.fixture
def secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / 'secrets'
    root.mkdir()
    monkeypatch.setenv('OP_VLM_SECRETS_DIR', str(root))
    (root / 'vendor').write_text(f'{KEY}\n')
    return root


def _serialised(api) -> str:
    """Every doc the store holds, as text."""
    return json.dumps(api.fake_os._docs, default=str)


# ---- resolving ---------------------------------------------------------------


def test_a_reference_resolves_to_the_trimmed_file_content(secrets: Path) -> None:
    assert resolve_api_key('secret:vendor') == KEY
    assert api_key_present('secret:vendor', is_env_builtin=False)
    assert list_secret_refs() == ['secret:vendor']
    assert resolve_api_key(None) is None


def test_only_plain_files_inside_the_mount_are_references(secrets: Path, tmp_path: Path) -> None:
    outside = tmp_path / 'outside-secret'
    outside.write_text('outside-value')
    (secrets / 'link').symlink_to(outside)  # a symlink out of the mount
    (secrets / 'adir').mkdir()
    (secrets / 'empty').write_text('')
    (secrets / 'huge').write_text('x' * 9000)
    for name in ('link', 'adir', 'empty', 'huge'):
        assert resolve_api_key(f'secret:{name}') is None, name
        assert not api_key_present(f'secret:{name}', is_env_builtin=False), name
    # names outside the slug alphabet are not references at all, and a file
    # with such a name is not listed
    (secrets / '.hidden').write_text('h')
    (secrets / 'Upper_Case').write_text('u')
    for name in ('.hidden', 'Upper_Case'):
        with pytest.raises(VlmKeyRefInvalidError):
            resolve_api_key(f'secret:{name}')
    assert list_secret_refs() == ['secret:vendor']


@pytest.mark.parametrize(
    'ref',
    [
        'sk-live-literal-key',
        'secret:../outside-secret',
        'secret:vendor/../vendor',
        'secret:/etc/passwd',
        'secret:',
        'secret:a b',
        'env:HOME',
        'env:OP_VLM_API_KEY',  # only the built-in may name its own key
        'file:/etc/passwd',
    ],
)
def test_anything_but_a_slug_reference_is_refused(secrets: Path, ref: str) -> None:
    with pytest.raises(VlmKeyRefInvalidError):
        resolve_api_key(ref)
    assert not api_key_present(ref, is_env_builtin=False)


def test_the_builtin_reads_its_key_from_its_own_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_VLM_API_KEY', f'  {KEY}  ')
    assert resolve_api_key('env:OP_VLM_API_KEY', is_env_builtin=True) == KEY
    monkeypatch.delenv('OP_VLM_API_KEY')
    assert resolve_api_key('env:OP_VLM_API_KEY', is_env_builtin=True) is None


# ---- never served -------------------------------------------------------------


def _surfaces(api) -> dict[str, Any]:
    client = api.client
    return {
        'list': client.get(GLOBAL_VLM),
        'doc': client.get(f'{GLOBAL_VLM}/ext'),
        'revisions': client.get(f'{GLOBAL_VLM}/ext/revisions'),
        'revision': client.get(f'{GLOBAL_VLM}/ext/revisions/1'),
        'active': client.get(f'{ACTIVE}/active'),
        'methods': client.get(f'{SCOPED}/methods'),
        'models': client.get(f'{SCOPED}/models/status'),
        'vocabulary': client.get(f'{SCOPED}/config/vocabulary'),
        'settings': client.get(f'{SCOPED}/settings'),
        'validate': client.post(
            f'{GLOBAL_VLM}/validate',
            params={'probe': 'true'},
            json={
                'name': 'ext',
                'body': api.body(
                    base_url='https://api.example.com/v1',
                    allow_external=True,
                    api_key_ref='secret:vendor',
                ),
            },
        ),
        'probe': client.post(f'{GLOBAL_VLM}/ext/probe'),
        'schema': client.get(f'{GLOBAL_VLM}/schema'),
        'openapi': client.get('/openapi.json'),
    }


def test_the_key_value_appears_in_no_response_and_no_stored_document(
    vlm_api, secrets: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    vlm_api.ready(
        'ext',
        base_url='https://api.example.com/v1',
        allow_external=True,
        api_key_ref='secret:vendor',
    )
    assert (
        vlm_api.activate('ext', expected_active=None, acknowledge_external=True).status_code == 200
    )
    responses = _surfaces(vlm_api)

    for name, response in responses.items():
        assert KEY not in response.text, f'{name} served the key'
    assert KEY not in _serialised(vlm_api)
    assert KEY not in caplog.text

    doc = responses['doc'].json()
    assert doc['body']['api_key_ref'] == 'secret:vendor'
    assert doc['api_key_present'] is True
    listed = responses['list'].json()
    assert [r['ref'] for r in listed['secret_refs']] == ['secret:vendor']
    assert all(set(r) <= {'ref', 'present', 'choice'} for r in listed['secret_refs'])


def test_the_probe_is_the_only_place_the_key_is_used_and_it_is_read_fresh(
    vlm_api, secrets: Path
) -> None:
    vlm_api.ready('keyed', api_key_ref='secret:vendor')
    assert vlm_api.probe_keys == [KEY]
    (secrets / 'vendor').write_text('ROTATED-KEY-1')  # a rotated key file
    vlm_api.probe('keyed')
    assert vlm_api.probe_keys == [KEY, 'ROTATED-KEY-1']
    assert 'ROTATED-KEY-1' not in _serialised(vlm_api)


def test_a_missing_key_file_is_reported_by_name_only(vlm_api, secrets: Path) -> None:
    (secrets / 'vendor').unlink()
    vlm_api.create('keyed', api_key_ref='secret:vendor')
    doc = vlm_api.client.get(f'{GLOBAL_VLM}/keyed').json()
    assert doc['api_key_present'] is False
    assert doc['body']['api_key_ref'] == 'secret:vendor'
    refused = vlm_api.activate('keyed', expected_active=None, force=True)
    assert refused.status_code == 422
    assert 'vlm_api_key_unresolved' in refused.text


def test_a_literal_key_in_the_reference_field_is_refused_and_not_echoed(vlm_api) -> None:
    literal = 'sk-live-LITERAL-1234567890'
    response = vlm_api.client.post(
        GLOBAL_VLM,
        json={'name': 'leaky', 'body': vlm_api.body(api_key_ref=literal)},
    )
    assert response.status_code == 422
    assert 'vlm_api_key_ref_invalid' in response.text
    assert literal not in response.text
    assert literal not in _serialised(vlm_api)
    validated = vlm_api.client.post(
        f'{GLOBAL_VLM}/validate', json={'body': vlm_api.body(api_key_ref=literal)}
    )
    assert validated.status_code == 200
    assert literal not in validated.text


def test_the_builtin_key_is_never_served_and_is_dropped_from_a_clone(
    vlm_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'served-alias')
    monkeypatch.setenv('OP_VLM_API_KEY', KEY)
    listed = vlm_api.client.get(GLOBAL_VLM)
    doc = vlm_api.client.get(f'{GLOBAL_VLM}/env')
    clone = vlm_api.client.post(f'{GLOBAL_VLM}/env/clone', json={'new_name': 'copy'})
    for response in (listed, doc, clone, vlm_api.client.get(f'{SCOPED}/models/status')):
        assert KEY not in response.text
    assert doc.json()['body']['api_key_ref'] == 'env:OP_VLM_API_KEY'
    assert doc.json()['api_key_present'] is True
    assert clone.json()['body']['api_key_ref'] is None
    assert KEY not in _serialised(vlm_api)


def test_a_url_with_credentials_is_refused_so_none_can_leak_through_a_listing(vlm_api) -> None:
    response = vlm_api.client.post(
        GLOBAL_VLM,
        json={'name': 'creds', 'body': vlm_api.body(base_url='http://user:hunter2@vlm:8000/v1')},
    )
    assert response.status_code == 422
    assert 'hunter2' not in response.text


def test_the_models_listing_strips_credentials_from_an_environment_url(
    vlm_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env built-in is not validated like a stored endpoint (it is the
    operator's own setting), but a listing must still never print a password
    that was put in the URL."""
    monkeypatch.setenv('OP_VLM_URL', 'http://user:hunter2@vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'served-alias')
    response = vlm_api.client.get(f'{SCOPED}/models/status')
    assert response.status_code == 200
    assert 'hunter2' not in response.text
