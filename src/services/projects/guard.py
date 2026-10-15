"""The OpenSearch project guard: a transport-level wrapper that refuses
every request that is not one of the shapes the codebase really sends,
aimed at the bound project's concrete index names (see
``docs/design/openprocessor_internal/projects_plan.md`` §2.4).

It fails **closed by construction**: there is no list of bad shapes to
keep up to date. A request passes only if it matches an allowlisted shape
(index-scoped document/search/by-query/bulk/indices calls, a handful of
read-only cluster facts, and the scroll/task handles those calls hand
out), and every index it names -- in the URL, in a multi-doc body line,
or anywhere in a search body -- is a concrete name owned by the bound
project. Everything else raises :class:`CrossProjectAccess`: index-less
searches, wildcards and patterns, ``_all``, aliases, ``_reindex``,
``_sql``/``_ppl``, snapshots, remote-cluster names, unknown names under
the project index prefix, and any body that reaches another index.

Unbound code (the non-curation ``visual_search_*`` routers, the registry)
may touch only indexes no project owns. The ``op_projects`` registry
index is readable from anywhere and writable only inside
:func:`bind_registry_admin`.

Installed on the shared client at construction
(``src.core.dependencies.OpenSearchClientFactory``) and on every script
client (:func:`make_script_opensearch`). A static test
(``tests/projects/test_no_frozen_project_config.py``) forbids
constructing ``AsyncOpenSearch(...)`` / ``OpenSearch(...)`` anywhere else.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any
from urllib.parse import quote

from src.config.project_context import try_current_project
from src.services.projects.guard_policy import check_request


# --- The transport wrapper ----------------------------------------------------

_HANDLE_CAPACITY = 4096


class ProjectGuardedTransport:
    """Wraps an ``opensearchpy`` transport's ``perform_request`` with
    :func:`check_request`. Installed once per client by
    :func:`install_project_guard`.

    ``refresh`` (optional, script clients): a coroutine function that
    refreshes the registry, run at most every ``refresh_interval`` seconds
    on use (serialized), since a script client has no poll loop."""

    project_guarded = True

    def __init__(
        self,
        inner: Any,
        registry: Any,
        refresh: Any = None,
        *,
        refresh_interval: float = 1.0,
    ) -> None:
        self._inner = inner
        self._registry = registry
        self._refresh = refresh
        self._refresh_interval = refresh_interval
        self._refreshed_at: float | None = None
        self._refresh_lock = asyncio.Lock()
        self._handles: OrderedDict[str, str | None] = OrderedDict()

    async def _maybe_refresh(self) -> None:
        if self._refresh is None:
            return
        now = time.monotonic()
        if self._refreshed_at is not None and now - self._refreshed_at < self._refresh_interval:
            return
        async with self._refresh_lock:
            if (
                self._refreshed_at is None
                or time.monotonic() - self._refreshed_at >= self._refresh_interval
            ):
                await self._refresh()
                self._refreshed_at = time.monotonic()

    def _remember(self, result: Any) -> None:
        if not isinstance(result, dict):
            return
        bound = try_current_project()
        slug = bound.record.slug if bound is not None else None
        for key in ('_scroll_id', 'task', 'pit_id'):
            handle = result.get(key)
            if isinstance(handle, str):
                self._handles[handle] = slug
                self._handles.move_to_end(handle)
        while len(self._handles) > _HANDLE_CAPACITY:
            self._handles.popitem(last=False)

    async def perform_request(
        self,
        method: str,
        url: str,
        params: Any = None,
        body: Any = None,
        **kwargs: Any,
    ) -> Any:
        await self._maybe_refresh()
        check_request(
            method, url, body, self._registry.snapshot(), params=params, handles=self._handles
        )
        result = await self._inner.perform_request(method, url, params=params, body=body, **kwargs)
        self._remember(result)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def install_project_guard(client: Any, registry: Any, *, refresh: Any = None) -> None:
    """Idempotently wrap ``client.transport`` with the project guard."""
    if getattr(client.transport, 'project_guarded', False):
        return
    client.transport = ProjectGuardedTransport(client.transport, registry, refresh)


class _RegistryReader:
    """The two calls :class:`~src.services.projects.registry.ProjectRegistry`
    makes, sent straight to the *unguarded* inner transport so reading the
    registry never re-enters the guard that needs it."""

    def __init__(self, transport: Any) -> None:
        self._transport = transport

    async def get(self, *, index: str, id: str) -> Any:  # noqa: A002 - mirrors the client API
        return await self._transport.perform_request('GET', f'/{index}/_doc/{quote(id, safe="")}')

    async def search(self, *, index: str, body: Any) -> Any:
        return await self._transport.perform_request('POST', f'/{index}/_search', body=body)


def make_script_opensearch(hosts: list[str], **kwargs: Any) -> Any:
    """The one factory for script / worker OpenSearch clients: an
    ``AsyncOpenSearch`` with the project guard installed. Its registry is
    refreshed on use, at most once a second, so a long-running worker
    learns about projects created after it started."""
    from opensearchpy import AsyncOpenSearch

    from src.clients.opensearch.serializer import OrjsonSerializer
    from src.services.projects.registry import ProjectRegistry

    kwargs.setdefault('use_ssl', False)
    kwargs.setdefault('serializer', OrjsonSerializer())
    client = AsyncOpenSearch(hosts=hosts, **kwargs)
    reader = _RegistryReader(client.transport)
    registry = ProjectRegistry(lambda: reader)
    install_project_guard(client, registry, refresh=registry.ensure_fresh)
    return client


async def make_curation_opensearch() -> Any:
    """The raw ``AsyncOpenSearch`` behind the shared client
    (``src.core.dependencies.get_opensearch()``). The guard is installed
    when that client is constructed; this only unwraps it."""
    from src.core.dependencies import get_opensearch

    wrapper = await get_opensearch()
    raw: Any = getattr(wrapper, 'client', wrapper)
    if not getattr(raw.transport, 'project_guarded', False):
        from src.services.projects.registry import get_project_registry

        install_project_guard(raw, get_project_registry())
    return raw
