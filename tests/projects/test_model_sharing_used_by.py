"""``PUT /models/{name}/sharing`` names the projects that use the model.

``used_by`` comes from every other project's LIVE active detection profile
(read-only, one project at a time), so unsharing a model another project
runs on is refused (409 ``in_use``) unless ``force``.
"""

from __future__ import annotations

from typing import Any

from curation import test_cross_project_leak as _sweep
from curation.test_cross_project_leak import LeakEnv, _promoted_model
from fastapi.testclient import TestClient

from src.config.curation import IndexRole


leak_env = _sweep.leak_env

API = '/curation/projects'
MODEL = 'alpha__model'


def _client(env: LeakEnv) -> TestClient:
    return TestClient(env.app, raise_server_exceptions=False)


def _activate_profile(env: LeakEnv, slug: str, *, detector: str, name: str = 'det') -> None:
    """``slug``'s active detection profile pins revision 1 of ``name``."""
    index = env.records[slug].resources.indexes[IndexRole.CONFIGS]
    docs = env.transport.store.setdefault(index, {})
    docs['activation:detection_profile'] = {'doc_type': 'activation', 'name': name, 'revision': 1}
    docs[f'profile:{name}@1'] = {
        'doc_type': 'revision',
        'name': name,
        'revision': 1,
        'body': {'detector_model': detector},
    }


def _put(client: TestClient, *, shared: bool, revision: int, **params: Any) -> Any:
    return client.put(
        f'{API}/alpha/models/{MODEL}/sharing',
        json={'shared': shared, 'expected_revision': revision},
        params=params,
    )


def test_used_by_lists_only_projects_whose_active_profile_uses_the_model(
    leak_env: LeakEnv,
) -> None:
    _promoted_model(leak_env, 'alpha')
    _activate_profile(leak_env, 'beta', detector=MODEL)
    _activate_profile(leak_env, 'default', detector='some_other_model')
    r = _put(_client(leak_env), shared=True, revision=1)
    assert r.status_code == 200, r.text
    assert r.json()['used_by'] == [{'project': 'beta', 'profile': 'det'}]


def test_unsharing_a_model_another_project_runs_on_is_refused(leak_env: LeakEnv) -> None:
    _promoted_model(leak_env, 'alpha')
    _activate_profile(leak_env, 'beta', detector=MODEL)
    client = _client(leak_env)
    assert _put(client, shared=True, revision=1).status_code == 200
    refused = _put(client, shared=False, revision=2)
    assert refused.status_code == 409, refused.text
    detail = refused.json()['detail']
    assert detail['error'] == 'in_use'
    assert detail['projects'] == ['beta']
    forced = _put(client, shared=False, revision=2, force='true')
    assert forced.status_code == 200, forced.text
    assert forced.json()['shared'] is False


def test_unsharing_an_unused_model_is_allowed(leak_env: LeakEnv) -> None:
    _promoted_model(leak_env, 'alpha')
    _activate_profile(leak_env, 'beta', detector='unrelated')
    client = _client(leak_env)
    assert _put(client, shared=True, revision=1).status_code == 200
    r = _put(client, shared=False, revision=2)
    assert r.status_code == 200, r.text
    assert r.json()['used_by'] == []


def test_the_owner_using_its_own_model_does_not_block_unsharing(leak_env: LeakEnv) -> None:
    _promoted_model(leak_env, 'alpha')
    _activate_profile(leak_env, 'alpha', detector=MODEL)
    client = _client(leak_env)
    assert _put(client, shared=True, revision=1).status_code == 200
    assert _put(client, shared=False, revision=2).status_code == 200


def test_reading_other_projects_touches_only_their_config_index_and_never_writes(
    leak_env: LeakEnv,
) -> None:
    _promoted_model(leak_env, 'alpha')
    _activate_profile(leak_env, 'beta', detector=MODEL)
    beta = leak_env.records['beta'].resources.indexes
    before = len(leak_env.accesses)
    assert _put(_client(leak_env), shared=True, revision=1).status_code == 200
    seen = [a for a in leak_env.accesses[before:] if a[0] == 'beta']
    assert seen
    assert {a[3] for a in seen} == {beta[IndexRole.CONFIGS]}
    assert {a[1] for a in seen} <= {'GET', 'POST'}
    assert leak_env.transport.writes.count(beta[IndexRole.CONFIGS]) == 0
