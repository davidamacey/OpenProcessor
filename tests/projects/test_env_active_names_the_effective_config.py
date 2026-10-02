"""``GET /region_profiles/active`` and ``/prompt_packs/active`` on a project
nobody activated through the store: ``source`` is ``env`` and ``active`` names
the env/file default the worker really applies (the same resolver the worker
and its ``applied[]`` row use), not a nameless "off". Only an explicit
deactivation or no env default at all is nameless."""

from __future__ import annotations

from typing import TYPE_CHECKING

from curation import test_cross_project_leak as _sweep
from fastapi.testclient import TestClient

from src.config.curation import IndexRole


if TYPE_CHECKING:
    from curation.test_cross_project_leak import LeakEnv


leak_env = _sweep.leak_env

BASE = '/curation/projects/alpha'


def _client(env: LeakEnv) -> TestClient:
    index = env.records['alpha'].resources.indexes[IndexRole.CONFIGS]
    for axis in ('detection_profile', 'prompt_pack'):
        env.transport.store[index].pop(f'activation:{axis}', None)
    return TestClient(env.app, raise_server_exceptions=False)


def test_env_profile_is_named_like_the_worker_names_it(leak_env: LeakEnv) -> None:
    from src.services.detection.profile_registry import get_active_region_profile

    body = _client(leak_env).get(f'{BASE}/region_profiles/active').json()
    effective = get_active_region_profile()
    assert effective is not None
    assert body['source'] == 'env'
    assert body['active'] == {'name': effective.name, 'revision': None}


def test_env_prompt_pack_is_named_like_the_worker_names_it(leak_env: LeakEnv) -> None:
    from src.services.labeling.vlm_prompt_resolution import active_prompt_pack

    body = _client(leak_env).get(f'{BASE}/prompt_packs/active').json()
    assert body['source'] == 'env'
    assert body['active'] == {'name': active_prompt_pack().name, 'revision': None}


def test_explicit_deactivation_stays_nameless(leak_env: LeakEnv) -> None:
    client = _client(leak_env)
    r = client.post(f'{BASE}/region_profiles/deactivate', json={'expected_active': None})
    assert r.status_code == 200, r.text
    assert r.json()['source'] == 'off'
    assert r.json()['active'] == {'name': None, 'revision': None}
