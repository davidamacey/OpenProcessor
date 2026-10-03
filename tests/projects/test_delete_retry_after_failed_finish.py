"""P3F item 1 (M4 retry): a failed index-delete finish leaves the record
'deleting' (retryable), and a re-issued ``DELETE ?confirm=<slug>`` on it
must return 202 (not 409 invalid_transition), and the retried finish must
actually succeed -- through the real FastAPI app / TestClient, behind the
real guard, not a bare unit call.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from curation import test_cross_project_leak as _sweep
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.services.projects import lifecycle
from src.services.projects.guard import make_curation_opensearch


if TYPE_CHECKING:
    from curation.test_cross_project_leak import LeakEnv


leak_env = _sweep.leak_env

API = '/curation/projects'


class _NoBackgroundFinish:
    """Replaces the router module's ``asyncio`` name binding so
    ``asyncio.create_task(_finish())`` in the DELETE route never actually
    schedules the fire-and-forget finish task. TestClient's own portal
    timing around a non-awaited background task is documented as
    unreliable (see test_lifecycle_guarded.py and the P3 review's own
    probes), so this test drives ``delete_project_finish`` itself,
    deterministically, instead of racing whatever the HTTP layer happens
    to run."""

    def __getattr__(self, name: str) -> Any:
        import asyncio as _real_asyncio

        return getattr(_real_asyncio, name)

    def create_task(self, coro: Any, *_a: Any, **_k: Any) -> Any:
        coro.close()

        class _Dummy:
            def add_done_callback(self, *_a: Any, **_k: Any) -> None:
                return None

            def done(self) -> bool:
                # P3F pass-3 MA1: the router's own scheduling guard
                # checks `.done()` on any task already registered for
                # the slug before scheduling a new one. This stub never
                # actually runs (its coro is closed immediately above),
                # so it must report itself as done -- otherwise the
                # route's own re-DELETE below would see a "still
                # running" dummy and skip scheduling the real retry this
                # test drives manually.
                return True

        return _Dummy()


@pytest.fixture
def _no_background_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.routers.curation.projects as projects_router

    monkeypatch.setattr(projects_router, 'asyncio', _NoBackgroundFinish())


@pytest.mark.usefixtures('_no_background_finish')
def test_delete_retry_after_failed_index_delete_succeeds(leak_env: LeakEnv) -> None:
    client_http = TestClient(leak_env.app, raise_server_exceptions=False)

    alpha = leak_env.records['alpha']
    failing_index = sorted(set(alpha.resources.indexes.values()))[0]

    real_perform = leak_env.transport.perform_request
    state = {'failed_once': False}

    async def _flaky_perform(
        method: str, url: str, params: Any = None, body: Any = None, **kw: Any
    ) -> Any:
        target = url.split('?')[0].strip('/')
        if method == 'DELETE' and target == failing_index and not state['failed_once']:
            state['failed_once'] = True
            raise RuntimeError('simulated transient cluster hiccup')
        return await real_perform(method, url, params=params, body=body, **kw)

    leak_env.transport.perform_request = _flaky_perform  # type: ignore[method-assign]

    resp = client_http.delete(f'{API}/alpha', params={'confirm': 'alpha'})
    assert resp.status_code == 202, resp.text
    assert resp.json()['project']['status'] == 'deleting'

    guarded_client = asyncio.run(make_curation_opensearch())

    # First real attempt at the finish hits the fault: record stays
    # 'deleting', never tombstoned.
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(lifecycle.delete_project_finish(guarded_client, slug='alpha'))
    assert exc_info.value.detail['error'] == 'project_busy'

    get_resp = client_http.get(f'{API}/alpha')
    assert get_resp.status_code == 200, get_resp.text
    assert get_resp.json()['status'] == 'deleting'

    # The M4 fix under test: a re-issued DELETE on an already-'deleting'
    # record answers 202 (not 409 invalid_transition), and re-triggers
    # the finish.
    retry_resp = client_http.delete(f'{API}/alpha', params={'confirm': 'alpha'})
    assert retry_resp.status_code == 202, retry_resp.text
    assert retry_resp.json()['project']['status'] == 'deleting'

    # The transient fault has cleared; the retried finish now succeeds.
    finished = asyncio.run(lifecycle.delete_project_finish(guarded_client, slug='alpha')).record
    assert finished.status == 'deleted'

    final = client_http.get(f'{API}/alpha')
    assert final.status_code == 404, final.text
