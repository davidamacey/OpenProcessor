"""``PUT /models/{name}/sharing`` names the projects that use the model.

``used_by`` comes from every other project's LIVE active detection profile
(read-only, one project at a time), so unsharing a model another project
runs on is refused (409 ``in_use``) unless ``force``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from curation.cross_project_leak import _env as _sweep
from curation.cross_project_leak._seeders import _activate_detection_profile, _promoted_model
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from curation.cross_project_leak._transport import LeakEnv


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_registry')]


leak_env = _sweep.leak_env

API = '/curation/projects'
MODEL = 'alpha__model'


def _client(env: LeakEnv) -> TestClient:
    return TestClient(env.app, raise_server_exceptions=False)


def _activate_profile(env: LeakEnv, slug: str, *, detector: str, name: str = 'det') -> None:
    """``slug``'s active detection profile pins revision 1 of ``name``."""
    _activate_detection_profile(env, slug, detector=detector, name=name)


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


def test_reading_other_projects_reaches_none_of_their_opensearch_indexes(
    leak_env: LeakEnv,
) -> None:
    """The activations are read from PostgreSQL, so another project's OpenSearch indexes
    are neither read nor written to answer ``used_by`` (the per-project ``configs`` index
    this used to read is gone: 2F-1)."""
    _promoted_model(leak_env, 'alpha')
    _activate_profile(leak_env, 'beta', detector=MODEL)
    beta = set(leak_env.records['beta'].resources.indexes.values())
    before = len(leak_env.accesses)
    writes_before = len(leak_env.transport.writes)
    r = _put(_client(leak_env), shared=True, revision=1)
    assert r.status_code == 200, r.text
    assert r.json()['used_by'] == [{'project': 'beta', 'profile': 'det'}]
    assert not [a for a in leak_env.accesses[before:] if a[3] in beta or a[0] == 'beta']
    assert not beta & set(leak_env.transport.writes[writes_before:])


def test_an_unreadable_project_refuses_unsharing_instead_of_being_skipped(
    leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``read_each_project`` backs refusals, so one project whose config
    cannot be read must block the change, not silently count as "unused"."""
    from src.services.config_store import project_usage

    _promoted_model(leak_env, 'alpha')
    client = _client(leak_env)
    assert _put(client, shared=True, revision=1).status_code == 200

    async def _broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError('database unavailable')

    monkeypatch.setattr(project_usage.repo, 'list_active_bodies', _broken)
    refused = _put(client, shared=False, revision=2)
    assert refused.status_code == 503, refused.text
    assert refused.json()['detail']['error'] == 'config_store_unavailable'


def test_active_detector_users_propagates_a_failed_read(
    leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.config_store import project_usage

    async def _broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError('unreadable')

    monkeypatch.setattr(project_usage.repo, 'list_active_bodies', _broken)
    with pytest.raises(RuntimeError, match='unreadable'):
        asyncio.run(project_usage.active_detector_users(MODEL))
