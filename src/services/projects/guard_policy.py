"""The project guard's allowlist and the verdict on a request.

Split out of ``guard.py``: the registry-admin and global-configs-read
bindings, the allowlisted request shapes, and :func:`check_request`, which
raises unless a request is an allowlisted shape whose every index belongs
to the bound project.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from src.config.env import env_str
from src.config.project_context import current_project, try_current_project
from src.config.projects import project_index_prefix
from src.services.projects.guard_errors import ProjectReadOnly, _refuse
from src.services.projects.guard_parse import (
    _body_targets,
    _handle_ids,
    _index_refs,
    _json_value,
    _segments,
)


if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from src.config.projects import ProjectRecord


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
    """Allow READS of the ``op_global_configs`` registry (the VLM
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
    return env_str('OP_PROJECTS_INDEX', 'op_projects')


def _global_configs_index() -> str:
    return env_str('OP_GLOBAL_CONFIGS_INDEX', 'op_global_configs')


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


_FORBIDDEN_NAME_CHARS = frozenset('*?:<>|"\\ #')


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
    for name in dict.fromkeys(targets):  # a bulk names the same index once per document
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
