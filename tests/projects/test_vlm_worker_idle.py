"""#207: the VLM worker decides once per cycle which projects have an active
VLM and makes no per-project call (OpenSearch search or label_batch route)
for the others; with none active it backs off instead of spinning on 409."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from scripts.curation import vlm_worker
from scripts.curation._vlm_activity import IDLE_BACKOFF_MAX_S, idle_backoff_s
from src.config.project_context import try_current_project
from tests.projects.test_worker_leak import SLUGS, _http_world, _items_index, _record


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord

pytestmark = pytest.mark.unbound


@pytest.fixture
def projects(tmp_path: Any) -> dict[str, ProjectRecord]:
    return {slug: _record(tmp_path, slug) for slug in SLUGS}


def _activate(monkeypatch: pytest.MonkeyPatch, active: set[str]) -> None:
    """Make the app's VLM facts say exactly ``active`` projects have a VLM."""

    async def _refresh(_client: Any) -> None:
        return None

    def _configured() -> bool:
        bound = try_current_project()
        return bound is not None and bound.record.slug in active

    monkeypatch.setattr('src.services.labeling.vlm_endpoints.refresh_vlm_state', _refresh)
    monkeypatch.setattr('src.services.labeling.vlm_endpoints.vlm_configured', _configured)


def _world(
    monkeypatch: pytest.MonkeyPatch, projects: dict[str, ProjectRecord], inactive: set[str]
) -> tuple[Any, Any, list[str]]:
    """The fake API answers the real route's 409 for ``inactive`` projects and
    counts those hits (they are the wasted calls of the bug)."""
    transport, api = _http_world(monkeypatch, vlm_worker, projects)
    real_handler = api.handler
    inactive_hits: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        slug = request.url.path.split('/projects/', 1)[1].split('/', 1)[0]
        if slug in inactive:
            inactive_hits.append(slug)
            return httpx.Response(
                409, json={'detail': {'error': 'vlm_not_configured', 'message': 'off'}}
            )
        return real_handler(request)

    api.handler = _handler  # type: ignore[method-assign]

    async def _no_heartbeat(*_a: Any, **_kw: Any) -> None:
        return None

    monkeypatch.setattr(vlm_worker, 'heartbeat_loop', _no_heartbeat)
    return transport, api, inactive_hits


def _args() -> Any:
    return vlm_worker.parse_args(
        ['--until-empty', '--vlm-batch-size', '4', '--concurrency', '2', '--poll-interval', '0']
    )


@pytest.mark.asyncio
async def test_no_active_vlm_makes_zero_per_project_calls(
    projects: dict[str, ProjectRecord],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, api, inactive_hits = _world(monkeypatch, projects, set(projects))
    _activate(monkeypatch, set())

    assert await asyncio.wait_for(vlm_worker.run(_args()), 20) == 0

    assert api.requests == []
    assert inactive_hits == []
    assert transport.calls == [], 'no project index may be searched while no VLM is active'


@pytest.mark.asyncio
async def test_only_the_project_with_an_active_vlm_is_polled(
    projects: dict[str, ProjectRecord],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, api, inactive_hits = _world(monkeypatch, projects, {'beta'})
    _activate(monkeypatch, {'alpha'})

    assert await asyncio.wait_for(vlm_worker.run(_args()), 20) == 0

    assert inactive_hits == []
    assert api.requests
    assert {path.split('/projects/', 1)[1].split('/', 1)[0] for path, _ in api.requests} == {
        'alpha'
    }
    searched = {idx for _, _, touched in transport.calls for idx in touched}
    assert _items_index(projects['alpha']) in searched
    assert not {idx for idx in searched if 'beta' in idx}, searched


def test_idle_backoff_doubles_and_is_bounded() -> None:
    assert idle_backoff_s(5.0, 1) == 5.0
    assert idle_backoff_s(5.0, 2) == 10.0
    assert idle_backoff_s(5.0, 50) == IDLE_BACKOFF_MAX_S
    assert idle_backoff_s(0.0, 50) == 0.0
