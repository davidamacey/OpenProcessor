"""Activation, rollback and deactivation of a project's VLM endpoint through
the real routes (W9.3, W9.6, W9.9), over an in-memory config store."""

from __future__ import annotations

from typing import Any

import pytest

from curation.conftest import ACTIVE, GLOBAL_VLM, good_probe


def _err(response: Any) -> dict[str, Any]:
    return response.json()['detail']


def _codes(detail: dict[str, Any]) -> list[str]:
    return [e['code'] for e in detail['report']['errors']]


# ---- the happy path and its history -----------------------------------------


def test_activate_switch_and_roll_back(vlm_api) -> None:
    vlm_api.ready('alpha')
    vlm_api.ready('beta')
    first = vlm_api.activate('alpha', expected_active=None)
    assert first.status_code == 200, first.text
    assert first.json()['active'] == {'name': 'alpha', 'revision': 1}
    assert first.json()['previous'] is None

    second = vlm_api.activate('beta', expected_active={'name': 'alpha', 'revision': 1})
    assert second.status_code == 200, second.text
    assert second.json()['active'] == {'name': 'beta', 'revision': 1}
    assert second.json()['previous'] == {'name': 'alpha', 'revision': 1}

    back = vlm_api.client.post(
        f'{ACTIVE}/active/rollback', json={'expected_active': {'name': 'beta', 'revision': 1}}
    )
    assert back.status_code == 200, back.text
    assert back.json()['active'] == {'name': 'alpha', 'revision': 1}
    assert vlm_api.active()['active'] == {'name': 'alpha', 'revision': 1}


def test_rollback_with_no_history_is_a_409(vlm_api) -> None:
    response = vlm_api.client.post(f'{ACTIVE}/active/rollback', json={})
    assert response.status_code == 409
    assert _err(response)['error'] == 'no_previous'


def test_rollback_to_a_deleted_endpoint_is_refused(vlm_api) -> None:
    vlm_api.ready('alpha')
    vlm_api.ready('beta')
    assert vlm_api.activate('alpha', expected_active=None).status_code == 200
    assert (
        vlm_api.activate('beta', expected_active={'name': 'alpha', 'revision': 1}).status_code
        == 200
    )
    deleted = vlm_api.client.delete(f'{GLOBAL_VLM}/alpha', params={'expected_revision': 1})
    assert deleted.status_code == 204, deleted.text
    response = vlm_api.client.post(f'{ACTIVE}/active/rollback', json={})
    assert response.status_code == 409
    assert _err(response)['error'] == 'previous_deleted'
    assert vlm_api.active()['active']['name'] == 'beta'


def test_deactivate_turns_the_vlm_off(vlm_api) -> None:
    vlm_api.ready('alpha')
    assert vlm_api.activate('alpha', expected_active=None).status_code == 200
    response = vlm_api.client.post(
        f'{ACTIVE}/deactivate', json={'expected_active': {'name': 'alpha', 'revision': 1}}
    )
    assert response.status_code == 200
    body = response.json()
    assert body['active'] == {'name': None, 'revision': None}
    assert body['source'] == 'off'
    # off is remembered: a stale caller cannot assume "never activated"
    stale = vlm_api.activate('alpha', expected_active=None)
    assert stale.status_code == 409
    assert _err(stale)['error'] == 'active_conflict'


def test_a_stale_expected_active_is_a_conflict_carrying_the_current_state(vlm_api) -> None:
    vlm_api.ready('alpha')
    vlm_api.ready('beta')
    assert vlm_api.activate('alpha', expected_active=None).status_code == 200
    response = vlm_api.activate('beta', expected_active=None)
    assert response.status_code == 409
    detail = _err(response)
    assert detail['error'] == 'active_conflict'
    assert detail['current'] == {'name': 'alpha', 'revision': 1}
    assert vlm_api.active()['active']['name'] == 'alpha'


def test_unknown_names_and_revisions(vlm_api) -> None:
    vlm_api.ready('alpha')
    missing = vlm_api.activate('nope', expected_active=None)
    assert (missing.status_code, _err(missing)['error']) == (404, 'not_found')
    bad_revision = vlm_api.activate('alpha', revision=9, expected_active=None)
    assert (bad_revision.status_code, _err(bad_revision)['error']) == (404, 'unknown_revision')


# ---- revisions are immutable once activated ---------------------------------


def test_saving_a_new_revision_changes_nothing_until_it_is_activated(vlm_api) -> None:
    vlm_api.ready('alpha')
    assert vlm_api.activate('alpha', expected_active=None).status_code == 200
    saved = vlm_api.client.put(
        f'{GLOBAL_VLM}/alpha',
        json={'expected_revision': 1, 'body': vlm_api.body(model='other-model')},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()['revision'] == 2
    assert vlm_api.active()['active'] == {'name': 'alpha', 'revision': 1}

    vlm_api.probe('alpha')
    moved = vlm_api.activate('alpha', expected_active={'name': 'alpha', 'revision': 1})
    assert moved.status_code == 200, moved.text
    assert moved.json()['active'] == {'name': 'alpha', 'revision': 2}
    # each revision keeps its own probe: going back to revision 1 needs none
    old = {'name': 'alpha', 'revision': 2}
    back = vlm_api.activate('alpha', revision=1, expected_active=old)
    assert back.status_code == 200, back.text
    assert back.json()['active'] == {'name': 'alpha', 'revision': 1}
    # a revision that was never probed is still refused
    third = vlm_api.client.put(
        f'{GLOBAL_VLM}/alpha',
        json={'expected_revision': 2, 'body': vlm_api.body(model='third-model')},
    )
    assert third.status_code == 200, third.text
    untested = vlm_api.activate(
        'alpha', revision=3, expected_active={'name': 'alpha', 'revision': 1}
    )
    assert untested.status_code == 422
    assert _codes(_err(untested)) == ['vlm_not_probed']


# ---- the probe requirement, and what force may (not) bypass -----------------


def test_an_unprobed_endpoint_is_refused_but_force_may_bypass_only_that(vlm_api) -> None:
    vlm_api.create('alpha')
    refused = vlm_api.activate('alpha', expected_active=None)
    assert refused.status_code == 422
    detail = _err(refused)
    assert detail['error'] == 'validation_failed'
    assert _codes(detail) == ['vlm_not_probed']
    assert detail['report']['force_allowed'] is True

    forced = vlm_api.activate('alpha', expected_active=None, force=True)
    assert forced.status_code == 200, forced.text
    # the bypassed error is still reported back
    assert [e['code'] for e in forced.json()['validation']['errors']] == ['vlm_not_probed']


def test_force_never_bypasses_the_servers_image_cap(vlm_api) -> None:
    vlm_api.probe_record[0] = good_probe(
        ok=False,
        issues=[
            {
                'code': 'vlm_max_images_exceeds_server',
                'severity': 'error',
                'message': 'the server takes 2 images',
                'detail': {},
            }
        ],
    )
    vlm_api.ready('alpha')
    refused = vlm_api.activate('alpha', expected_active=None, force=True)
    assert refused.status_code == 422
    assert _codes(_err(refused)) == ['vlm_max_images_exceeds_server']
    assert _err(refused)['report']['force_allowed'] is False


# ---- endpoints outside the deployment ---------------------------------------


EXTERNAL = {'base_url': 'https://api.example.com/v1', 'allow_external': True}


def test_an_external_endpoint_cannot_be_saved_without_its_own_flag(vlm_api) -> None:
    response = vlm_api.client.post(
        GLOBAL_VLM,
        json={'name': 'ext', 'body': vlm_api.body(base_url='https://api.example.com/v1')},
    )
    assert response.status_code == 422
    assert 'vlm_external_not_acknowledged' in _codes(_err(response))


def test_activating_an_external_endpoint_needs_an_explicit_acknowledgement(vlm_api) -> None:
    vlm_api.ready('ext', **EXTERNAL)
    refused = vlm_api.activate('ext', expected_active=None)
    assert refused.status_code == 422
    detail = _err(refused)
    assert detail['error'] == 'vlm_external_not_acknowledged'
    assert detail['endpoint'] == 'ext'
    assert detail['activate_via'].endswith('/vlm/endpoints/ext/activate')
    # force is not an acknowledgement
    still = vlm_api.activate('ext', expected_active=None, force=True)
    assert still.status_code == 422
    assert vlm_api.active()['active']['name'] is None

    acked = vlm_api.activate('ext', expected_active=None, acknowledge_external=True)
    assert acked.status_code == 200, acked.text
    assert acked.json()['active'] == {'name': 'ext', 'revision': 1}


def test_a_recorded_acknowledgement_covers_rollback_but_not_a_new_revision(vlm_api) -> None:
    vlm_api.ready('ext', **EXTERNAL)
    vlm_api.ready('lan')
    assert (
        vlm_api.activate('ext', expected_active=None, acknowledge_external=True).status_code == 200
    )
    assert (
        vlm_api.activate('lan', expected_active={'name': 'ext', 'revision': 1}).status_code == 200
    )
    back = vlm_api.client.post(
        f'{ACTIVE}/active/rollback', json={'expected_active': {'name': 'lan', 'revision': 1}}
    )
    assert back.status_code == 200, back.text
    assert back.json()['active'] == {'name': 'ext', 'revision': 1}

    # an edit is a new revision: it is not acknowledged until someone says so
    saved = vlm_api.client.put(
        f'{GLOBAL_VLM}/ext',
        json={'expected_revision': 1, 'body': vlm_api.body(model='changed', **EXTERNAL)},
    )
    assert saved.status_code == 200, saved.text
    vlm_api.probe('ext')
    refused = vlm_api.activate('ext', expected_active={'name': 'ext', 'revision': 1})
    assert refused.status_code == 422
    assert _err(refused)['error'] == 'vlm_external_not_acknowledged'


def test_rollback_to_an_unacknowledged_external_endpoint_is_refused(vlm_api) -> None:
    """The activation doc names ``ext`` as previous, but this project never
    acknowledged it (the doc was written by something that skipped the gate);
    rollback has no flag to fix that, so it refuses."""
    vlm_api.ready('ext', **EXTERNAL)
    vlm_api.ready('lan')
    assert vlm_api.activate('lan', expected_active=None).status_code == 200
    doc = vlm_api.fake_os._docs['op_prj_default__configs']['activation:vlm']['_source']
    doc['previous'] = {'name': 'ext', 'revision': 1}
    response = vlm_api.client.post(
        f'{ACTIVE}/active/rollback', json={'expected_active': {'name': 'lan', 'revision': 1}}
    )
    assert response.status_code == 422
    assert _err(response)['error'] == 'vlm_external_not_acknowledged'
    assert vlm_api.active()['active']['name'] == 'lan'


def test_the_deny_policy_refuses_external_even_with_the_flag(
    vlm_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    vlm_api.ready('ext', **EXTERNAL)
    monkeypatch.setenv('OP_VLM_EXTERNAL_POLICY', 'deny')
    response = vlm_api.activate('ext', expected_active=None, acknowledge_external=True, force=True)
    assert response.status_code == 422
    assert 'vlm_external_denied' in _codes(_err(response))


# ---- SSRF -------------------------------------------------------------------


@pytest.mark.parametrize(
    ('url', 'code'),
    [
        ('http://169.254.169.254/v1', 'vlm_url_denied_address'),
        ('http://2852039166/v1', 'vlm_url_denied_address'),
        ('http://[::ffff:169.254.169.254]/v1', 'vlm_url_denied_address'),
        ('http://opensearch:9200/v1', 'vlm_url_denied_internal_service'),
        ('http://yolo-api:4603/v1', 'vlm_url_denied_internal_service'),
    ],
)
def test_forbidden_targets_cannot_be_created_validated_or_probed(
    vlm_api, url: str, code: str
) -> None:
    created = vlm_api.client.post(
        GLOBAL_VLM, json={'name': 'evil', 'body': vlm_api.body(base_url=url)}
    )
    assert created.status_code == 422
    assert code in _codes(_err(created))

    validated = vlm_api.client.post(
        f'{GLOBAL_VLM}/validate',
        params={'probe': 'true'},
        json={'body': vlm_api.body(base_url=url)},
    )
    assert validated.status_code == 200
    body = validated.json()
    assert not body['validation']['ok']
    assert body['probe'] is None  # the URL was never contacted
    assert body['locality'] is None


# ---- deleting -----------------------------------------------------------------


def test_an_endpoint_a_project_runs_cannot_be_deleted(vlm_api) -> None:
    vlm_api.ready('alpha')
    assert vlm_api.activate('alpha', expected_active=None).status_code == 200
    response = vlm_api.client.delete(f'{GLOBAL_VLM}/alpha', params={'expected_revision': 1})
    assert response.status_code == 409
    detail = _err(response)
    assert detail['error'] == 'in_use'
    assert detail['projects'] == ['default']

    assert (
        vlm_api.client.post(
            f'{ACTIVE}/deactivate', json={'expected_active': {'name': 'alpha', 'revision': 1}}
        ).status_code
        == 200
    )
    assert (
        vlm_api.client.delete(f'{GLOBAL_VLM}/alpha', params={'expected_revision': 1}).status_code
        == 204
    )
    assert vlm_api.client.get(f'{GLOBAL_VLM}/alpha').status_code == 404


# ---- the built-in --------------------------------------------------------------


def test_the_env_builtin_is_listed_first_read_only_and_needs_no_probe(
    vlm_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'served-alias')
    vlm_api.create('alpha')
    listed = vlm_api.client.get(GLOBAL_VLM).json()['endpoints']
    assert [e['name'] for e in listed] == ['env', 'alpha']
    assert listed[0]['read_only'] is True
    assert listed[0]['source'] == 'env'

    put = vlm_api.client.put(
        f'{GLOBAL_VLM}/env', json={'expected_revision': 1, 'body': vlm_api.body()}
    )
    assert put.status_code == 403
    delete = vlm_api.client.delete(f'{GLOBAL_VLM}/env', params={'expected_revision': 1})
    assert delete.status_code == 403

    activated = vlm_api.activate('env', expected_active=None)
    assert activated.status_code == 200, activated.text
    assert activated.json()['active']['name'] == 'env'


def test_cloning_the_builtin_drops_its_environment_key_reference(
    vlm_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_API_KEY', 'sekret')
    monkeypatch.setenv('OP_VLM_MODEL', 'served-alias')
    response = vlm_api.client.post(f'{GLOBAL_VLM}/env/clone', json={'new_name': 'mine'})
    assert response.status_code == 201, response.text
    doc = response.json()
    assert doc['body']['api_key_ref'] is None
    assert 'vlm_api_key_ref_dropped' in [w['code'] for w in doc['validation']['warnings']]
    assert 'sekret' not in response.text
