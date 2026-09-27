"""P3F pass-3 MA1: the router's belt-and-braces guard against
scheduling a second delete-finish task for a slug that already has one
registered and not done (``_BACKGROUND_DELETE_TASKS`` keyed by slug).
Uses the real, guarded ``leak_env`` app exactly like
``test_delete_retry_after_failed_finish.py`` -- no local env fixture
here, since ``leak_env`` builds its own complete environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from curation import test_cross_project_leak as _sweep
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from curation.test_cross_project_leak import LeakEnv

leak_env = _sweep.leak_env

API = '/curation/projects'


def test_delete_route_never_schedules_a_second_finish_task_while_one_is_registered(
    leak_env: LeakEnv,
) -> None:
    """Exercised through the REAL route function (not a
    re-implementation of its branch logic): a DELETE while a finish
    task for that slug is already registered and not done must not
    replace it with a second task."""
    import src.routers.curation.projects as projects_router

    client_http = TestClient(leak_env.app, raise_server_exceptions=False)
    projects_router._BACKGROUND_DELETE_TASKS.clear()

    class _NotDoneYet:
        """Duck-types the one attribute the route's scheduling check
        reads (``.done()``) -- avoids depending on a real asyncio.Task
        surviving across TestClient's own portal event loop, which is
        not the loop this synchronous test body runs on."""

        def done(self) -> bool:
            return False

    fake_task = _NotDoneYet()
    projects_router._BACKGROUND_DELETE_TASKS['alpha'] = fake_task  # type: ignore[assignment]

    try:
        resp = client_http.delete(f'{API}/alpha', params={'confirm': 'alpha'})
        assert resp.status_code == 202, resp.text

        assert projects_router._BACKGROUND_DELETE_TASKS['alpha'] is fake_task, (  # type: ignore[comparison-overlap]
            'a finish already registered (not done) must not be replaced by a new task'
        )
    finally:
        projects_router._BACKGROUND_DELETE_TASKS.clear()
