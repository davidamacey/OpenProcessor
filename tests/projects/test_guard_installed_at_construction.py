"""The project guard is installed when the shared OpenSearch client is
built, not as a side effect of some later call (P1 review M2): the very
first request to an image/crop route on a fresh app already goes through
it, and the lifespan never binds a project for its background work (M3).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from opensearchpy import AsyncOpenSearch


class _Bottom:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def perform_request(self, method: str, url: str, **_kw: Any) -> Any:
        self.calls.append(f'{method} {url}')
        if method == 'HEAD':
            return True
        return {'_id': 'default-item', 'found': False}

    async def close(self) -> None:
        return None


class _FreshClient:
    """Stands in for ``OpenSearchClient``: a real ``AsyncOpenSearch`` whose
    network transport is replaced by a recorder."""

    instances: list[_FreshClient] = []

    def __init__(self, *_a: Any, **_k: Any) -> None:
        self.client = AsyncOpenSearch(hosts=['http://127.0.0.1:9'])
        self.bottom = _Bottom()
        self.client.transport = self.bottom  # type: ignore[assignment]
        _FreshClient.instances.append(self)

    async def ping(self) -> bool:
        return bool(await self.client.ping())

    async def close(self) -> None:
        return None


def test_first_thumbnail_request_on_a_fresh_app_is_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import dependencies
    from src.main import app
    from src.services.projects.guard import ProjectGuardedTransport

    _FreshClient.instances.clear()
    monkeypatch.setattr(dependencies, 'OpenSearchClient', _FreshClient)
    monkeypatch.setattr(dependencies.app_state, '_opensearch_client', None)

    response = TestClient(app).get('/curation/projects/default/crops/default-item/thumbnail')

    (fresh,) = _FreshClient.instances
    assert isinstance(fresh.client.transport, ProjectGuardedTransport)
    assert response.status_code == 404
    # The ping and the crop read both went through the guard to the bottom.
    assert fresh.bottom.calls[0] == 'HEAD /'
    assert any('/_doc/default-item' in call for call in fresh.bottom.calls)


def test_the_lifespan_binds_no_project() -> None:
    """Background loops started by the lifespan inherit its context; it
    must be unbound so a global loop never silently acts on ``default``."""
    import inspect

    import src.main as main_module
    from src.services.projects import bootstrap

    source = inspect.getsource(main_module)
    assert 'set_bound_project' not in source
    assert 'bind_default_for_lifespan' not in source
    assert not hasattr(bootstrap, 'bind_default_for_lifespan')


@pytest.mark.unbound
def test_for_each_project_binds_each_active_project_in_turn() -> None:
    import dataclasses

    from src.config.curation import base_curation_config
    from src.config.project_context import current_project, is_project_bound
    from src.config.projects import ProjectStatus, resources_for_new
    from src.services.projects import registry as registry_mod
    from src.services.projects.bootstrap import for_each_project
    from src.services.projects.registry import default_project_record

    base = default_project_record()
    registry = registry_mod.get_project_registry()
    statuses: tuple[tuple[str, ProjectStatus], ...] = (
        ('alpha', 'active'),
        ('old', 'archived'),
        ('half', 'failed'),
    )
    for slug, status in statuses:
        registry._by_slug[slug] = dataclasses.replace(
            base,
            slug=slug,
            status=status,
            resources=resources_for_new(slug, base_curation_config()),
        )
    seen = [(slug, current_project().record.slug) for slug in for_each_project()]
    assert seen == [('alpha', 'alpha'), ('default', 'default')]
    assert not is_project_bound()
