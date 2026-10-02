"""``PUT /models/{name}/sharing`` publishes typed 409 ``in_use`` and 503
``config_store_unavailable`` bodies, and the served errors match them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from curation import test_cross_project_leak as _sweep
from curation.test_cross_project_leak import LeakEnv, _promoted_model
from fastapi.testclient import TestClient

from src.routers.curation import _models_sharing as sharing


if TYPE_CHECKING:
    import pytest


leak_env = _sweep.leak_env

PATH = '/curation/projects/{slug}/models/{model_name}/sharing'
URL = '/curation/projects/alpha/models/alpha__model/sharing'


def _put(client: TestClient, *, shared: bool, revision: int, **params: Any) -> Any:
    return client.put(URL, json={'shared': shared, 'expected_revision': revision}, params=params)


def _documented(env: LeakEnv, status: str) -> dict[str, Any]:
    spec = env.app.openapi()
    op = next(
        v['put'] for k, v in spec['paths'].items() if k.endswith('/models/{model_name}/sharing')
    )
    return op['responses'][status]['content']['application/json']['schema']


def test_409_and_503_are_published_as_typed_models(leak_env: LeakEnv) -> None:
    assert _documented(leak_env, '409')['$ref'].endswith('/ModelSharingConflictResponse')
    assert _documented(leak_env, '503')['$ref'].endswith('/ModelSharingUnavailableResponse')


def test_in_use_409_names_projects_and_each_users_profile(leak_env: LeakEnv) -> None:
    from projects.test_model_sharing_used_by import _activate_profile

    _promoted_model(leak_env, 'alpha')
    _activate_profile(leak_env, 'beta', detector='alpha__model', name='det')
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    assert _put(client, shared=True, revision=1).status_code == 200
    refused = _put(client, shared=False, revision=2)
    assert refused.status_code == 409, refused.text
    body = sharing.ModelSharingConflictResponse.model_validate(refused.json())
    assert body.detail.error == 'in_use'
    assert body.detail.projects == ['beta']
    assert [(u.project, u.profile) for u in body.detail.used_by or []] == [('beta', 'det')]


def test_revision_conflict_409_validates_against_the_same_model(leak_env: LeakEnv) -> None:
    _promoted_model(leak_env, 'alpha')
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    r = _put(client, shared=True, revision=99)
    assert r.status_code == 409, r.text
    body = sharing.ModelSharingConflictResponse.model_validate(r.json())
    assert body.detail.error == 'revision_conflict'
    assert body.detail.current_revision == 1


def test_unreadable_store_503_validates_against_its_model(
    leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError('opensearch unavailable')

    _promoted_model(leak_env, 'alpha')
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    assert _put(client, shared=True, revision=1).status_code == 200
    monkeypatch.setattr(sharing, 'active_detector_users', _broken)
    refused = _put(client, shared=False, revision=2)
    assert refused.status_code == 503, refused.text
    body = sharing.ModelSharingUnavailableResponse.model_validate(refused.json())
    assert body.detail.error == 'config_store_unavailable'
    forced = _put(client, shared=False, revision=2, force='true')
    assert forced.status_code == 200, forced.text
