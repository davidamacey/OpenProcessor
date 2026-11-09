"""A project count that could not be read is ``None`` (unknown), never ``0``: an
OpenSearch outage must not look like an empty project."""

from __future__ import annotations

from typing import Any

import pytest
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError

from src.schemas.curation.projects import ProjectCounts, ProjectStatsCounts
from src.services.projects.stats import _term_count, index_count
from tests.projects.test_project_counts import _record


class _Down:
    async def count(self, **_: Any) -> dict[str, int]:
        raise OpenSearchConnectionError('N/A', 'opensearch is down', Exception('down'))


@pytest.mark.asyncio
async def test_index_count_is_none_when_the_count_fails() -> None:
    assert await index_count(_Down(), 'items') is None


@pytest.mark.asyncio
async def test_term_count_is_none_when_the_count_fails() -> None:
    assert await _term_count(_Down(), 'items', 'region_status', 'pending_detection') is None


@pytest.mark.asyncio
async def test_a_zero_count_stays_zero() -> None:
    class _Empty:
        async def count(self, **_: Any) -> dict[str, int]:
            return {'count': 0}

    assert await index_count(_Empty(), 'items') == 0
    assert await _term_count(_Empty(), 'items', 'f', True) == 0


@pytest.mark.asyncio
async def test_project_list_counts_are_unknown_during_an_outage() -> None:
    from src.routers.curation.projects import _fetch_counts

    counts = await _fetch_counts(_Down(), {'alpha': _record('alpha')})

    assert counts['alpha'].images is None
    assert counts['alpha'].items is None
    assert counts['alpha'].validated is None
    assert counts['alpha'].items_embedded is None


def test_the_wire_models_carry_null() -> None:
    ProjectCounts(images=None, items=None)
    ProjectStatsCounts(
        images=None,
        items=None,
        validated=None,
        items_embedded=None,
        pending_detection=None,
        holdout_items=None,
        classes=0,
        promoted_models=0,
    )


class _Registry:
    def __init__(self, *records: Any) -> None:
        self._records = {r.slug: r for r in records}

    async def ensure_fresh(self) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return dict(self._records)

    def retired_slugs(self) -> list[str]:
        return []

    async def lookup(self, slug: str) -> Any:
        return self._records.get(slug)


async def _unreachable() -> Any:
    raise OpenSearchConnectionError('N/A', 'opensearch is down', Exception('down'))


@pytest.mark.asyncio
async def test_the_project_list_shows_unknown_counts_when_opensearch_is_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routers.curation import projects as router

    monkeypatch.setattr(router, 'get_project_registry', lambda: _Registry(_record('alpha')))
    monkeypatch.setattr(router, 'make_curation_opensearch', _unreachable)

    response = await router.list_projects()

    (alpha,) = response.projects
    assert alpha.counts.images is None
    assert alpha.counts.items is None


@pytest.mark.asyncio
async def test_one_project_shows_unknown_counts_when_opensearch_is_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routers.curation import projects as router

    monkeypatch.setattr(router, 'get_project_registry', lambda: _Registry(_record('alpha')))
    monkeypatch.setattr(router, 'make_curation_opensearch', _unreachable)

    response = await router.get_project('alpha')

    assert response.counts.images is None
    assert response.counts.items is None


class _UnpingableClient:
    """What ``OpenSearchClient`` is when the cluster cannot be reached at creation."""

    instances: list[_UnpingableClient] = []

    def __init__(self, **_: Any) -> None:
        self.client = object()
        self.closed = False
        self.instances.append(self)

    async def ping(self) -> bool:
        return False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_the_project_list_shows_unknown_counts_when_the_client_cannot_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outage surfaces from the real client factory (its ping fails), not from a
    stand-in raising the right class: the list route must still answer."""
    from src.core import dependencies
    from src.routers.curation import projects as router

    monkeypatch.setattr(dependencies.app_state, '_opensearch_client', None)
    monkeypatch.setattr(dependencies, 'OpenSearchClient', _UnpingableClient)
    monkeypatch.setattr(router, 'get_project_registry', lambda: _Registry(_record('alpha')))

    response = await router.list_projects()

    (alpha,) = response.projects
    assert alpha.counts.images is None
    assert response.capacity is None


@pytest.mark.asyncio
async def test_a_failed_ping_is_not_cached_so_the_next_call_pings_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import dependencies

    monkeypatch.setattr(dependencies.app_state, '_opensearch_client', None)
    monkeypatch.setattr(dependencies, 'OpenSearchClient', _UnpingableClient)
    _UnpingableClient.instances.clear()

    for _ in range(2):
        with pytest.raises(OpenSearchConnectionError):
            await dependencies.OpenSearchClientFactory.get_client()

    assert dependencies.app_state._opensearch_client is None
    assert len(_UnpingableClient.instances) == 2
    assert all(c.closed for c in _UnpingableClient.instances)
