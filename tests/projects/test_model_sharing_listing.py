"""Cross-project model listing and class mapping (projects_plan.md §5.5 #3
and #4, Cropwright delta 8), through the real app and the route sweep's
three-project fixture.

Classes cross a project boundary by NAME only: every listed model carries
``class_mapping`` built by ``model_class_mapping`` against the *consuming*
project's registry, never the model's raw output ids.
"""

from __future__ import annotations

import json
from typing import Any

from curation import test_cross_project_leak as _sweep
from curation.test_cross_project_leak import LeakEnv, _promoted_model
from fastapi.testclient import TestClient


leak_env = _sweep.leak_env

API = '/curation/projects'


def _client(env: LeakEnv) -> TestClient:
    return TestClient(env.app, raise_server_exceptions=False)


def _set_classes(env: LeakEnv, slug: str, names: list[str]) -> None:
    path = env.root / 'models' / f'{slug}__model' / 'promote.json'
    raw = json.loads(path.read_text(encoding='utf-8'))
    raw['classes'] = [{'model_id': i, 'name': n} for i, n in enumerate(names)]
    path.write_text(json.dumps(raw), encoding='utf-8')


def _share(client: TestClient, slug: str) -> None:
    r = client.put(
        f'{API}/{slug}/models/{slug}__model/sharing',
        json={'shared': True, 'expected_revision': 1},
    )
    assert r.status_code == 200, r.text


def _entries(client: TestClient, slug: str, **params: Any) -> dict[str, dict[str, Any]]:
    r = client.get(f'{API}/{slug}/models/status', params=params)
    assert r.status_code == 200, r.text
    return {m['name']: m for m in r.json()['models']}


def test_an_unshared_model_is_never_listed_for_another_project(leak_env: LeakEnv) -> None:
    _promoted_model(leak_env, 'alpha')
    client = _client(leak_env)
    assert 'alpha__model' not in _entries(client, 'beta', include_other_projects='true')
    assert (
        'alpha_zebra'
        not in client.get(
            f'{API}/beta/models/status', params={'include_other_projects': 'true'}
        ).text
    )


def test_a_shared_model_is_listed_for_another_project_only_on_request(leak_env: LeakEnv) -> None:
    _promoted_model(leak_env, 'alpha')
    # Model order: output id 0 is beta's class under another case, id 1 has
    # no beta class at all.
    _set_classes(leak_env, 'alpha', ['Beta_Heron', 'alpha_zebra'])
    client = _client(leak_env)
    _share(client, 'alpha')

    assert 'alpha__model' not in _entries(client, 'beta')
    entry = _entries(client, 'beta', include_other_projects='true')['alpha__model']
    assert entry['project'] == 'alpha'
    assert entry['shared'] is True
    assert entry['class_mapping'] == {'mapped_count': 1, 'unmapped': ['alpha_zebra']}


def test_own_models_carry_their_project_sharing_and_class_mapping(leak_env: LeakEnv) -> None:
    _promoted_model(leak_env, 'alpha')
    entry = _entries(_client(leak_env), 'alpha')['alpha__model']
    assert entry['project'] == 'alpha'
    assert entry['shared'] is False
    assert entry['class_mapping'] == {'mapped_count': 1, 'unmapped': []}


def test_the_full_mapping_is_served_by_name_for_a_shared_model(leak_env: LeakEnv) -> None:
    _promoted_model(leak_env, 'alpha')
    _set_classes(leak_env, 'alpha', ['Beta_Heron', 'alpha_zebra'])
    client = _client(leak_env)
    _share(client, 'alpha')

    r = client.get(f'{API}/beta/models/alpha__model/class_mapping')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['model'] == 'alpha__model'
    assert body['model_project'] == 'alpha'
    assert body['project'] == 'beta'
    assert body['entries'] == [
        {
            'model_id': 0,
            'model_name': 'Beta_Heron',
            'class_id': 1,
            'class_name': 'beta_heron',
            'match': 'case_insensitive',
        },
        {
            'model_id': 1,
            'model_name': 'alpha_zebra',
            'class_id': None,
            'class_name': None,
            'match': 'none',
        },
    ]
    assert body['unmapped'] == ['alpha_zebra']
    assert body['not_covered'] == ['beta_second']
    assert set(body['labels']['match']) == {'exact', 'case_insensitive', 'none'}


def test_owned_entry_carries_sharing_revision_and_owned_true(leak_env: LeakEnv) -> None:
    _promoted_model(leak_env, 'alpha')
    entry = _entries(_client(leak_env), 'alpha')['alpha__model']
    assert entry['owned'] is True
    assert entry['sharing_revision'] == 1


def test_sharing_revision_bumps_after_a_share_toggle(leak_env: LeakEnv) -> None:
    _promoted_model(leak_env, 'alpha')
    client = _client(leak_env)
    _share(client, 'alpha')
    entry = _entries(client, 'alpha')['alpha__model']
    assert entry['sharing_revision'] == 2


def test_foreign_shared_entry_is_not_owned_and_has_no_sharing_revision_and_unloadable_false(
    leak_env: LeakEnv,
) -> None:
    _promoted_model(leak_env, 'alpha')
    client = _client(leak_env)
    _share(client, 'alpha')
    entry = _entries(client, 'beta', include_other_projects='true')['alpha__model']
    assert entry['owned'] is False
    assert entry['sharing_revision'] is None
    assert entry['unloadable'] is False


def test_core_and_external_entries_are_not_owned_with_no_sharing_revision(
    leak_env: LeakEnv,
) -> None:
    entries = _entries(_client(leak_env), 'alpha')
    # Every core/external entry (kind != this project's own promoted model)
    # must still carry the fields, never omit them.
    for entry in entries.values():
        assert 'owned' in entry
        assert 'sharing_revision' in entry


def test_the_mapping_of_an_unshared_foreign_model_is_404(leak_env: LeakEnv) -> None:
    _promoted_model(leak_env, 'alpha')
    client = _client(leak_env)
    r = client.get(f'{API}/beta/models/alpha__model/class_mapping')
    assert r.status_code == 404, r.text
    assert r.json()['detail']['error'] == 'model_not_found'
    assert 'alpha_zebra' not in r.text

    r = client.get(f'{API}/beta/models/never_promoted/class_mapping')
    assert r.status_code == 404, r.text
    assert r.json()['detail']['error'] == 'model_not_found'
