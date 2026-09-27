"""P2 isolation probes through the real app (``src.main.app``, real routers,
real project guard), using the route sweep's three-project fixture.

Each test is an adversarial call a project should not be able to make
against another project's promoted model, pause flag or export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from curation import test_cross_project_leak as _sweep
from curation.test_cross_project_leak import LeakEnv, _promoted_model
from fastapi.testclient import TestClient


# The route sweep's fixture: the real app, three seeded projects, the real guard.
leak_env = _sweep.leak_env

API = '/curation/projects'


def _client(env: LeakEnv) -> TestClient:
    return TestClient(env.app, raise_server_exceptions=False)


def _promote_json(env: LeakEnv, slug: str) -> dict[str, Any]:
    path = env.root / 'models' / f'{slug}__model' / 'promote.json'
    return json.loads(path.read_text(encoding='utf-8'))


def test_a_project_cannot_share_or_delete_another_projects_model(leak_env: LeakEnv) -> None:
    _promoted_model(leak_env, 'alpha')
    before = _promote_json(leak_env, 'alpha')
    client = _client(leak_env)

    r = client.put(
        f'{API}/beta/models/alpha__model/sharing', json={'shared': True, 'expected_revision': 1}
    )
    assert r.status_code == 404, r.text
    r = client.delete(f'{API}/beta/models/alpha__model')
    assert r.status_code == 404, r.text
    # default owns un-prefixed names only, never another project's.
    r = client.put(
        f'{API}/default/models/alpha__model/sharing',
        json={'shared': True, 'expected_revision': 1},
    )
    assert r.status_code == 404, r.text
    assert _promote_json(leak_env, 'alpha') == before

    r = client.put(
        f'{API}/alpha/models/alpha__model/sharing', json={'shared': True, 'expected_revision': 1}
    )
    assert r.status_code == 200, r.text
    assert _promote_json(leak_env, 'alpha')['shared'] is True


def test_default_does_not_inherit_a_model_whose_project_left_the_registry(
    leak_env: LeakEnv,
) -> None:
    """``default`` owns every name without a *known* project prefix. Once
    ``alpha`` is gone from the registry (deleted, or not yet loaded),
    ``alpha__model``'s promote.json still names alpha as its owner, and
    default must not be able to share or unload it."""
    from src.services.projects.registry import get_project_registry

    _promoted_model(leak_env, 'alpha')
    get_project_registry()._by_slug.pop('alpha')
    r = _client(leak_env).put(
        f'{API}/default/models/alpha__model/sharing',
        json={'shared': True, 'expected_revision': 1},
    )
    assert r.status_code == 404, r.text
    assert _promote_json(leak_env, 'alpha')['shared'] is False


def test_pausing_one_project_leaves_the_others_running(leak_env: LeakEnv) -> None:
    client = _client(leak_env)
    paused_resp = client.post(f'{API}/beta/pause').json()
    assert paused_resp['project'] == 'beta'
    assert paused_resp['paused'] is True
    assert client.get(f'{API}/beta/pause').json()['paused'] is True
    for other in ('alpha', 'default'):
        other_resp = client.get(f'{API}/{other}/pause').json()
        assert other_resp['project'] == other
        assert other_resp['paused'] is False
    assert client.post(f'{API}/beta/resume').json()['paused'] is False


def _training_body(route: str, export_dir: Path) -> dict[str, Any]:
    if route == '/train/start_campaign':
        return {'dataset_export_dir': str(export_dir), 'runs': [{'profile': 'probe'}]}
    return {'dataset_export_dir': str(export_dir)}


@pytest.mark.parametrize('route', ['/train/preflight', '/train/start', '/train/start_campaign'])
def test_training_refuses_another_projects_export(leak_env: LeakEnv, route: str) -> None:
    """``dataset_export_dir`` is caller-supplied. Pointing it at another
    project's export must be refused at the API (not just by the trainer
    later), and ``force`` must not bypass it: the preflight otherwise reads
    that project's manifest, registry and labels into its report."""
    alpha_export = leak_env.records['alpha'].resources.export_root / 'alpha-v1'
    (alpha_export / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (alpha_export / 'manifest.json').write_text(
        json.dumps({'alpha_secret_marker': True}), encoding='utf-8'
    )
    r = _client(leak_env).post(
        f'{API}/beta{route}',
        params={'force': 'true'},
        json=_training_body(route, alpha_export),
    )
    assert r.status_code == 422, r.text
    assert r.json()['detail']['error'] == 'export_outside_project'
    assert 'alpha_secret_marker' not in r.text
    beta_jobs = Path(leak_env.records['beta'].resources.train_jobs_dir)
    assert not list(beta_jobs.glob('*.job.json'))
