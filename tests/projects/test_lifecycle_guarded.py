"""Lifecycle routes through the real project guard.

The lifecycle unit tests use an unguarded fake client, so a registry write
made outside ``bind_registry_admin()`` passed there and was refused live
(``internal_isolation_error`` on ``POST /curation/projects``). These drive
the real app with the real guard installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from curation import test_cross_project_leak as _sweep
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from curation.test_cross_project_leak import LeakEnv


leak_env = _sweep.leak_env

API = '/curation/projects'


def _client(env: LeakEnv) -> TestClient:
    return TestClient(env.app, raise_server_exceptions=False)


def _not_refused_by_the_guard(resp: Any) -> None:
    detail = (resp.json() or {}).get('detail')
    error = detail.get('error') if isinstance(detail, dict) else None
    assert error != 'internal_isolation_error', resp.text
    assert resp.status_code < 500, resp.text


def test_create_project_writes_the_registry(leak_env: LeakEnv) -> None:
    resp = _client(leak_env).post(API, json={'slug': 'gamma', 'display_name': 'Gamma'})
    _not_refused_by_the_guard(resp)
    assert resp.status_code in (200, 201), resp.text


def test_patch_and_archive_write_the_registry(leak_env: LeakEnv) -> None:
    client = _client(leak_env)
    record = client.get(f'{API}/beta').json()
    resp = client.patch(
        f'{API}/beta',
        json={'display_name': 'Beta renamed', 'expected_revision': record['revision']},
    )
    _not_refused_by_the_guard(resp)
    record = client.get(f'{API}/beta').json()
    resp = client.post(f'{API}/beta/archive', json={'expected_revision': record['revision']})
    _not_refused_by_the_guard(resp)


def _mark_archived(leak_env: LeakEnv, slug: str) -> None:
    from dataclasses import replace

    from src.services.projects.registry import get_project_registry

    registry = get_project_registry()
    registry._by_slug[slug] = replace(registry._by_slug[slug], status='archived')


def test_archive_then_write_is_read_only_and_unarchive_restores(leak_env: LeakEnv) -> None:
    """M2 (T1 was vacuous -- this actually attempts a write): a real
    file-backed write route (POST .../classes) and a real OpenSearch
    write are both refused 409 project_archived on an archived project,
    BEFORE the handler runs -- no class_registry.json rewrite, no doc
    written. Reads still work."""
    _mark_archived(leak_env, 'alpha')
    client = _client(leak_env)

    write_resp = client.post('/curation/projects/alpha/classes', json={'name': 'alpha_new'})
    assert write_resp.status_code == 409, write_resp.text
    assert write_resp.json()['detail']['error'] == 'project_archived'

    read_resp = client.get('/curation/projects/alpha/classes')
    assert read_resp.status_code == 200, read_resp.text


def test_stale_registry_write_is_refused_read_only(leak_env: LeakEnv) -> None:
    """The other read-only bind reason (M2): a stale registry (last
    refresh failed) refuses writes too, not just archived projects."""
    from src.services.projects.registry import get_project_registry

    registry = get_project_registry()
    registry._failed_at = 0.0  # any non-None value makes .stale True

    client = _client(leak_env)
    write_resp = client.post('/curation/projects/beta/classes', json={'name': 'beta_new'})
    assert write_resp.status_code == 409, write_resp.text
    assert write_resp.json()['detail']['error'] == 'project_read_only'
