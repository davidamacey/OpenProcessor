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
import contextvars
import json
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote

from prometheus_client import Counter

from src.config.project_context import current_project, try_current_project
from src.config.projects import project_index_prefix
from src.core.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from src.config.projects import ProjectRecord

logger = get_logger(__name__)


class CrossProjectAccess(RuntimeError):  # noqa: N818 - name fixed by the projects plan (§2.4)
    """A request reaches an index the bound project does not own, or has a
    shape the guard does not allow."""


class ProjectReadOnly(RuntimeError):  # noqa: N818 - name fixed by the projects plan (§2.4)
    """A write was attempted while bound read-only (e.g. to an archived
    project, or a combine source read via ``bind_project(..., read_only=True)``)."""


OP_CROSS_PROJECT_ACCESS = Counter(
    'op_cross_project_access_total',
    'OpenSearch requests the project guard refused.',
    ['bound', 'target'],
)
_cross_project_access_count = 0


def cross_project_access_count() -> int:
    """Process-wide count of refused requests (the same events the
    ``op_cross_project_access_total`` counter records, unlabelled)."""
    return _cross_project_access_count


_REGISTRY_ADMIN: contextvars.ContextVar[bool] = contextvars.ContextVar(
    'op_registry_admin', default=False
)


@contextmanager
def bind_registry_admin() -> Generator[None, None, None]:
    """Allow writes to the ``op_projects`` registry index and the one
    cross-project metadata read (``_cat/indices`` doc counts) for the
    duration of the block. Only project lifecycle code and the global
    project list enter it."""
    token = _REGISTRY_ADMIN.set(True)
    try:
        yield
    finally:
        _REGISTRY_ADMIN.reset(token)


_GLOBAL_CONFIGS_READ: contextvars.ContextVar[bool] = contextvars.ContextVar(
    'op_global_configs_read', default=False
)


@contextmanager
def global_configs_read() -> Generator[None, None, None]:
    """Allow READS of the ``op_global_configs`` registry (W9: the VLM
    endpoint registry) while a project is bound. Project-scoped code
    (a run's ``?vlm=``, the worker's per-project runtime) resolves the
    deployment-wide endpoints through the config store, which enters this
    for its own reads only; a write to that index while bound is still
    refused (global routes run unbound), and so is any raw access outside
    the block."""
    token = _GLOBAL_CONFIGS_READ.set(True)
    try:
        yield
    finally:
        _GLOBAL_CONFIGS_READ.reset(token)


def _projects_index() -> str:
    return os.environ.get('OP_PROJECTS_INDEX', 'op_projects')


def _global_configs_index() -> str:
    return os.environ.get('OP_GLOBAL_CONFIGS_INDEX', 'op_global_configs')


# --- The allowlist -----------------------------------------------------------

_READ = frozenset({'GET', 'HEAD'})
_READ_POST = frozenset({'GET', 'POST'})

# Cluster-level facts with no index and no document content.
_GLOBAL_SHAPES: frozenset[tuple[str, tuple[str, ...]]] = frozenset(
    {
        ('GET', ()),
        ('HEAD', ()),
        ('GET', ('_cluster', 'health')),
        ('GET', ('_cluster', 'settings')),
        ('GET', ('_nodes', 'stats', 'jvm')),
    }
)

# ``/{index}/<rest>``: rest -> allowed methods. ``*`` stands for one doc id.
_INDEX_SHAPES: dict[tuple[str, ...], frozenset[str]] = {
    (): frozenset({'HEAD', 'GET', 'PUT', 'DELETE'}),
    ('_doc',): frozenset({'POST'}),
    ('_doc', '*'): frozenset({'GET', 'HEAD', 'PUT', 'POST', 'DELETE'}),
    ('_create', '*'): frozenset({'PUT', 'POST'}),
    ('_update', '*'): frozenset({'POST'}),
    ('_source', '*'): _READ,
    ('_search',): _READ_POST,
    ('_count',): _READ_POST,
    ('_mget',): _READ_POST,
    ('_msearch',): _READ_POST,
    ('_bulk',): frozenset({'POST', 'PUT'}),
    ('_update_by_query',): frozenset({'POST'}),
    ('_delete_by_query',): frozenset({'POST'}),
    ('_refresh',): _READ_POST,
    ('_mapping',): frozenset({'GET', 'PUT', 'POST'}),
    ('_settings',): frozenset({'GET', 'PUT'}),
    ('_stats',): frozenset({'GET'}),
}

# Index-less multi-doc calls: every body line / doc names its own index.
_MULTI_DOC_SHAPES: dict[str, frozenset[str]] = {
    '_bulk': frozenset({'POST', 'PUT'}),
    '_mget': _READ_POST,
    '_msearch': _READ_POST,
}

# Actions whose body is a query (scanned for any index it names). Document
# writes, mappings and settings carry data, not index references.
_QUERY_ACTIONS = frozenset({'_search', '_count', '_update_by_query', '_delete_by_query', '_mget'})
_READ_ACTIONS = frozenset({'_search', '_count', '_mget', '_msearch', '_refresh', '_source'})
_INDEX_REF_KEYS = frozenset({'index', '_index', 'indices'})
# msearch header keys (a header may override the URL's index with either
# ``index`` or ``indices``; both are read, anything else is refused).
_MSEARCH_HEADER_KEYS = frozenset(
    {'index', 'indices', 'preference', 'routing', 'request_cache', 'search_type'}
)
_FORBIDDEN_NAME_CHARS = frozenset('*?:<>|"\\ #')


def _refuse(reason: str, *, target: str = '') -> CrossProjectAccess:
    global _cross_project_access_count  # noqa: PLW0603 - process-wide rejection counter

    bound = try_current_project()
    slug = bound.record.slug if bound is not None else '-'
    _cross_project_access_count += 1
    OP_CROSS_PROJECT_ACCESS.labels(bound=slug, target=target or '-').inc()
    logger.error('cross_project_access', bound=slug, target=target, reason=reason)
    return CrossProjectAccess(f"project guard refused a request (bound '{slug}'): {reason}")


def _segments(url: str) -> list[str]:
    return [unquote(part) for part in url.split('?', 1)[0].split('/') if part]


def _json_value(body: Any) -> Any:
    """``body`` as parsed JSON: dicts/lists as-is (opensearch-py passes
    ``mget``/``search`` bodies to the transport unserialized), str/bytes
    parsed. Unparseable text is refused rather than skipped."""
    if body is None or isinstance(body, (dict, list)):
        return body
    text = body.decode('utf-8', 'replace') if isinstance(body, bytes) else str(body)
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except ValueError as exc:
        raise _refuse('request body is not JSON') from exc


def _ndjson_lines(body: Any) -> list[Any]:
    if body is None:
        return []
    if isinstance(body, list):
        return body
    text = body.decode('utf-8', 'replace') if isinstance(body, bytes) else str(body)
    lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            lines.append(json.loads(line))
        except ValueError as exc:
            raise _refuse('multi-doc body line is not JSON') from exc
    return lines


def _index_refs(value: Any) -> list[str]:
    """Every string under an ``index``/``_index``/``indices`` key anywhere
    in a query body (terms lookups, ``more_like_this`` docs, ``mget``
    docs, ...)."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _INDEX_REF_KEYS:
                if isinstance(item, str):
                    found.extend(item.split(','))
                elif isinstance(item, list):
                    found.extend(x for x in item if isinstance(x, str))
            found.extend(_index_refs(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_index_refs(item))
    return found


def _check_name(name: str) -> None:
    if (
        not name
        or name.startswith(('_', '-', '+', '.'))
        or any(ch in _FORBIDDEN_NAME_CHARS for ch in name)
    ):
        raise _refuse(f'{name!r} is not a concrete index name', target=name)


def owner_slug(index: str, snapshot: Mapping[str, ProjectRecord]) -> str | None:
    """The project slug that owns ``index``, or ``None`` for an index
    owned by no project (``visual_search_*``, ``op_projects``, ...)."""
    for slug, record in snapshot.items():
        if index in record.resources.indexes.values():
            return slug
    return None


def _check_targets(
    targets: list[str], *, write: bool, snapshot: Mapping[str, ProjectRecord]
) -> None:
    """Every target must be a concrete name this bind may touch."""
    if not targets:
        raise _refuse('the request names no index (it would reach every index)')
    bound = try_current_project()
    admin = _REGISTRY_ADMIN.get()
    for name in targets:
        _check_name(name)
        owner = owner_slug(name, snapshot)
        if owner is None:
            if name == _projects_index():
                if write and not admin:
                    raise _refuse(
                        'the project registry is writable only by lifecycle code', target=name
                    )
                continue
            if name.startswith(project_index_prefix()):
                raise _refuse(f'{name!r} belongs to no known project', target=name)
            if bound is not None and name == _global_configs_index() and not write:
                if _GLOBAL_CONFIGS_READ.get():
                    continue
                raise _refuse(
                    f'{name!r} is readable while bound only through the config store', target=name
                )
            if bound is not None:
                raise _refuse(f'{name!r} is not an index of the bound project', target=name)
            continue
        if bound is None:
            current_project()  # raises ProjectNotBound
        assert bound is not None
        if owner != bound.record.slug:
            raise _refuse(f'{name!r} belongs to project {owner!r}', target=name)
        if bound.read_only and write:
            raise ProjectReadOnly(
                f"project '{bound.record.slug}' is bound read-only; refusing a write to '{name}'"
            )


def _handle_ids(segments: list[str], body: Any, params: Any) -> list[str]:
    ids: list[str] = []
    if len(segments) > 2:
        ids.extend(segments[2].split(','))
    payload = _json_value(body)
    raw = (payload or {}).get('scroll_id') if isinstance(payload, dict) else None
    if raw is None and isinstance(params, dict):
        raw = params.get('scroll_id')
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8', 'replace')
    if isinstance(raw, str):
        ids.extend(raw.split(','))
    elif isinstance(raw, list):
        ids.extend(str(x) for x in raw)
    return ids


def _check_handles(ids: list[str], handles: Mapping[str, str | None] | None) -> None:
    """A scroll, point-in-time or task id is usable only by the project
    whose request opened it (or, unbound, by global work)."""
    if not ids:
        raise _refuse('a scroll/task call must name the handle it continues')
    bound = try_current_project()
    slug = bound.record.slug if bound is not None else None
    for handle in ids:
        if handle == '_all':
            raise _refuse('clearing every scroll is never allowed')
        if handles is None or handle not in handles:
            raise _refuse('unknown scroll/task handle', target=handle[:32])
        if handles[handle] != slug:
            raise _refuse('the scroll/task handle belongs to another project', target=handle[:32])


def check_request(  # noqa: PLR0911 - one branch per allowlisted shape family
    method: str,
    url: str,
    body: Any,
    snapshot: Mapping[str, ProjectRecord],
    *,
    params: Any = None,
    handles: Mapping[str, str | None] | None = None,
) -> None:
    """Raise unless ``method``/``url``/``body`` is an allowlisted shape
    whose every index belongs to the bound project (or, unbound, to no
    project). ``handles`` maps scroll/task ids this client handed out to
    the slug that opened them."""
    method = method.upper()
    segs = _segments(url)
    head = segs[0] if segs else ''

    if not head.startswith('_') and segs:
        if segs[1:] == ['_search', 'point_in_time'] and method == 'POST':
            # Opens a point-in-time; its id is remembered as this bind's handle.
            _check_targets(head.split(','), write=False, snapshot=snapshot)
            return
        rest = tuple(segs[1:2]) + (('*',) if len(segs) > 2 else ())
        if len(segs) > 3 or method not in _INDEX_SHAPES.get(rest, frozenset()):
            raise _refuse(f'{method} /{head}/{"/".join(segs[1:])} is not an allowed request shape')
        action = rest[0] if rest else ''
        targets = head.split(',')
        write = _is_write(method, action)
        if action in _QUERY_ACTIONS or action in {'_msearch', '_bulk'}:
            targets = targets + _body_targets(action, body)
        if not rest and method == 'PUT':
            payload = _json_value(body)
            if isinstance(payload, dict) and payload.get('aliases'):
                raise _refuse('creating an index with aliases is never allowed', target=head)
        _check_targets(targets, write=write, snapshot=snapshot)
        return

    key = (method, tuple(segs))
    if key in _GLOBAL_SHAPES:
        return
    if head in _MULTI_DOC_SHAPES and len(segs) == 1:
        if method not in _MULTI_DOC_SHAPES[head]:
            raise _refuse(f'{method} /{head} is not an allowed request shape')
        _check_targets(
            _body_targets(head, body, require_each=True),
            write=_is_write(method, head),
            snapshot=snapshot,
        )
        return
    if segs == ['_search'] and method in _READ_POST:
        # The only index-less search: a point-in-time search, which reaches
        # exactly the index its (remembered) PIT was opened on.
        payload = _json_value(body)
        pit = payload.get('pit') if isinstance(payload, dict) else None
        pit_id = pit.get('id') if isinstance(pit, dict) else None
        if not isinstance(pit_id, str):
            raise _refuse('a search must name its index (it would reach every index)')
        if _index_refs(payload):
            raise _refuse('a point-in-time search may not name other indexes')
        _check_handles([pit_id], handles)
        return
    if segs == ['_search', 'point_in_time'] and method == 'DELETE':
        payload = _json_value(body)
        raw = payload.get('pit_id') if isinstance(payload, dict) else None
        ids = [raw] if isinstance(raw, str) else [str(x) for x in raw or []]
        _check_handles(ids, handles)
        return
    if segs[:2] == ['_search', 'scroll'] and len(segs) <= 3 and method in {'GET', 'POST', 'DELETE'}:
        _check_handles(_handle_ids(segs, body, params), handles)
        return
    if head == '_tasks' and len(segs) == 2 and method == 'GET':
        _check_handles([segs[1]], handles)
        return
    if segs[:3] == ['_plugins', '_knn', 'warmup'] and len(segs) == 4 and method == 'GET':
        _check_targets(segs[3].split(','), write=False, snapshot=snapshot)
        return
    if segs[:2] == ['_cat', 'indices'] and len(segs) == 3 and method == 'GET':
        if not _REGISTRY_ADMIN.get():
            raise _refuse('_cat/indices is only for the global project list')
        for name in segs[2].split(','):
            _check_name(name)
        return
    raise _refuse(f'{method} {url.split("?", 1)[0]} is not an allowed request shape')


def _is_write(method: str, action: str) -> bool:
    if method in ('GET', 'HEAD'):
        return False
    if method == 'POST':
        return action not in _READ_ACTIONS
    return True


def _body_targets(action: str, body: Any, *, require_each: bool = False) -> list[str]:
    """Indexes a request body names. ``require_each``: the URL names no
    index, so every line / doc must name its own."""
    targets: list[str] = []
    if action == '_bulk':
        lines = _ndjson_lines(body)
        i = 0
        while i < len(lines):
            line = lines[i]
            op = next(iter(line), None) if isinstance(line, dict) and len(line) == 1 else None
            if op not in ('index', 'create', 'update', 'delete') or not isinstance(line[op], dict):
                raise _refuse('a bulk action line is malformed')
            name = line[op].get('_index')
            if name:
                targets.append(name)
            elif require_each:
                raise _refuse('a bulk action names no index')
            i += 1 if op == 'delete' else 2
        return targets
    if action == '_msearch':
        lines = _ndjson_lines(body)
        for header in lines[0::2]:
            if not isinstance(header, dict):
                raise _refuse('msearch header is not an object')
            unknown = set(header) - _MSEARCH_HEADER_KEYS
            if unknown:
                raise _refuse(f'msearch header key(s) {sorted(unknown)} are not allowed')
            named = [
                part
                for key in ('index', 'indices')
                for value in [header.get(key)]
                if value
                for part in (value.split(',') if isinstance(value, str) else value)
            ]
            if named:
                targets.extend(named)
            elif require_each:
                raise _refuse('an msearch header names no index')
        for query in lines[1::2]:
            targets.extend(_index_refs(query))
        return targets
    payload = _json_value(body)
    if action == '_mget' and require_each:
        docs = (payload or {}).get('docs') if isinstance(payload, dict) else None
        if not docs or any(not isinstance(d, dict) or not d.get('_index') for d in docs):
            raise _refuse('an mget doc names no index')
    return targets + _index_refs(payload)


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

    from src.services.projects.registry import ProjectRegistry

    kwargs.setdefault('use_ssl', False)
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
    raw = getattr(wrapper, 'client', wrapper)
    if not getattr(raw.transport, 'project_guarded', False):
        from src.services.projects.registry import get_project_registry

        install_project_guard(raw, get_project_registry())
    return raw
