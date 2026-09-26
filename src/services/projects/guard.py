"""The OpenSearch project guard: a transport-level wrapper that refuses
any call whose target index belongs to a project other than the one
currently bound (see
``docs/design/openprocessor_internal/projects_plan.md`` §2.4).

Installed once, on the one shared ``AsyncOpenSearch`` client the curation
subsystem uses (``src.core.dependencies.get_opensearch()``'s
``.client``), via :func:`make_curation_opensearch`.

Scripts and workers build their own clients through
:func:`make_script_opensearch`, which installs the same guard. A static
test (``tests/projects/test_no_frozen_project_config.py``) forbids
constructing ``AsyncOpenSearch(...)`` / ``OpenSearch(...)`` anywhere in
``src/`` or ``scripts/`` except these factories.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from src.config.project_context import current_project
from src.core.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.config.projects import ProjectRecord

logger = get_logger(__name__)

# Actions that never write, even under POST (OpenSearch overloads POST for
# read-only search-shaped calls). Anything else under POST, plus every PUT
# and DELETE, is treated as a write for the read-only-binding check.
_READ_ONLY_ACTIONS = frozenset(
    {
        '_search',
        '_msearch',
        '_count',
        '_mget',
        '_field_caps',
        '_mapping',
        '_settings',
        '_cat',
        '_stats',
        '_alias',
        '_aliases',
        '_validate',
        '_explain',
        '_rank_eval',
        '_termvectors',
        '_mtermvectors',
    }
)
_MULTI_DOC_ACTIONS = frozenset({'_bulk', '_msearch', '_mget'})


class CrossProjectAccess(RuntimeError):  # noqa: N818 - name fixed by the projects plan (§2.4)
    """A call's target index belongs to a project other than the bound
    one."""


class ProjectReadOnly(RuntimeError):  # noqa: N818 - name fixed by the projects plan (§2.4)
    """A write was attempted while bound read-only (e.g. to an archived
    project, or a combine source read via ``bind_project(..., read_only=True)``)."""


_cross_project_access_count = 0


def cross_project_access_count() -> int:
    """Process-wide counter of rejected cross-project accesses -- stands
    in for the plan's ``op_cross_project_access_total`` Prometheus
    counter until this module is wired into the metrics registry."""
    return _cross_project_access_count


def _split_url(url: str) -> tuple[list[str] | None, str]:
    """``/op_items/_doc/abc`` -> (['op_items'], '_doc'). ``/a,b/_msearch``
    -> (['a', 'b'], '_msearch'). ``/_cluster/health`` -> (None,
    '_cluster') -- a leading ``_`` segment means "no index target"."""
    parts = [p for p in url.split('/') if p]
    if not parts:
        return None, ''
    if parts[0] == '_all':
        return ['_all'], (parts[1] if len(parts) > 1 and parts[1].startswith('_') else '')
    if parts[0].startswith('_'):
        return None, parts[0]
    indices = parts[0].split(',')
    action = next((p for p in parts[1:] if p.startswith('_')), '')
    return indices, action


def _is_write(method: str, action: str) -> bool:
    method = method.upper()
    if method == 'GET':
        return False
    if method in ('PUT', 'DELETE'):
        return True
    if method == 'POST':
        return action not in _READ_ONLY_ACTIONS
    return False


def _decode_body(body: Any) -> str:
    if body is None:
        return ''
    if isinstance(body, bytes):
        return body.decode('utf-8', errors='replace')
    return str(body)


def _indices_from_bulk_body(text: str) -> set[str]:
    indices: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        for action_key in ('index', 'create', 'update', 'delete'):
            meta = doc.get(action_key)
            if isinstance(meta, dict) and meta.get('_index'):
                indices.add(meta['_index'])
    return indices


def _indices_from_msearch_body(text: str) -> set[str]:
    indices: set[str] = set()
    lines = [line for line in text.splitlines() if line.strip()]
    # msearch alternates header/query lines; only headers carry `index`.
    for line in lines[0::2]:
        try:
            header = json.loads(line)
        except ValueError:
            continue
        target = header.get('index')
        if target is None:
            continue
        if isinstance(target, str):
            indices.update(target.split(','))
        else:
            indices.update(target)
    return indices


def _indices_from_mget_body(text: str) -> set[str]:
    indices: set[str] = set()
    try:
        body = json.loads(text) if text else {}
    except ValueError:
        return indices
    for doc in body.get('docs', []) or []:
        if isinstance(doc, dict) and doc.get('_index'):
            indices.add(doc['_index'])
    return indices


def _indices_from_multi_doc_body(action: str, body: Any) -> set[str]:
    text = _decode_body(body)
    if action == '_bulk':
        return _indices_from_bulk_body(text)
    if action == '_msearch':
        return _indices_from_msearch_body(text)
    if action == '_mget':
        return _indices_from_mget_body(text)
    return set()


def owner_slug(index: str, snapshot: Mapping[str, ProjectRecord]) -> str | None:
    """The project slug that owns ``index``, or ``None`` for an index
    owned by no project (``visual_search_*``, ``op_projects``, ...)."""
    for slug, record in snapshot.items():
        if index in record.resources.indexes.values():
            return slug
    return None


def check_request(
    method: str,
    url: str,
    body: Any,
    snapshot: Mapping[str, ProjectRecord],
) -> None:
    """Raise if ``method``/``url``/``body`` touches an index this bind may
    not touch. No-op for global (non-project) endpoints and unowned
    indexes."""
    global _cross_project_access_count  # noqa: PLW0603 - process-wide rejection counter

    indices, action = _split_url(url)
    if indices is None and action not in _MULTI_DOC_ACTIONS:
        return

    target_indices: set[str] = set()
    for pattern in indices or []:
        if pattern in ('*', '_all') or pattern.startswith('*'):
            raise CrossProjectAccess(f'wildcard/_all index pattern {pattern!r} is never allowed')
        target_indices.add(pattern)
    if action in _MULTI_DOC_ACTIONS:
        target_indices |= _indices_from_multi_doc_body(action, body)

    owned_targets = [(idx, owner_slug(idx, snapshot)) for idx in target_indices]
    owned_targets = [(idx, owner) for idx, owner in owned_targets if owner is not None]
    if not owned_targets:
        return

    bound = current_project()  # raises ProjectNotBound if nothing is bound

    is_write = _is_write(method, action)
    for idx, owner in owned_targets:
        if owner != bound.record.slug:
            _cross_project_access_count += 1
            logger.error(
                'cross_project_access bound=%s target_project=%s target_index=%s',
                bound.record.slug,
                owner,
                idx,
            )
            raise CrossProjectAccess(
                f"bound project '{bound.record.slug}' may not access "
                f"'{idx}' (owned by project '{owner}')"
            )
        if bound.read_only and is_write:
            raise ProjectReadOnly(
                f"project '{bound.record.slug}' is bound read-only; refusing a write to '{idx}'"
            )


class ProjectGuardedTransport:
    """Wraps an ``opensearchpy`` transport's ``perform_request`` with
    :func:`check_request`. Installed once per client by
    :func:`install_project_guard`. ``prime`` (optional) refreshes the
    registry once before the first checked call -- a script's client
    has no background poll loop to do it."""

    def __init__(self, inner: Any, registry: Any, prime: Any = None) -> None:
        self._inner = inner
        self._registry = registry
        self._prime = prime

    async def perform_request(
        self,
        method: str,
        url: str,
        params: Any = None,
        body: Any = None,
        **kwargs: Any,
    ) -> Any:
        if self._prime is not None:
            prime, self._prime = self._prime, None
            await prime()
        check_request(method, url, body, self._registry.snapshot())
        return await self._inner.perform_request(method, url, params=params, body=body, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def install_project_guard(client: Any, registry: Any, *, prime: Any = None) -> None:
    """Idempotently wrap ``client.transport`` with the project guard."""
    if isinstance(client.transport, ProjectGuardedTransport):
        return
    client.transport = ProjectGuardedTransport(client.transport, registry, prime)


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
    read once, lazily, before the first request."""
    from opensearchpy import AsyncOpenSearch

    from src.services.projects.registry import ProjectRegistry

    kwargs.setdefault('use_ssl', False)
    client = AsyncOpenSearch(hosts=hosts, **kwargs)
    reader = _RegistryReader(client.transport)
    registry = ProjectRegistry(lambda: reader)
    install_project_guard(client, registry, prime=registry.ensure_fresh)
    return client


async def make_curation_opensearch() -> Any:
    """The one factory for curation OpenSearch access: fetches the shared
    client (``src.core.dependencies.get_opensearch()``) and ensures the
    project guard is installed on it, then returns the raw
    ``AsyncOpenSearch`` instance."""
    from src.core.dependencies import get_opensearch
    from src.services.projects.registry import get_project_registry

    wrapper = await get_opensearch()
    raw = getattr(wrapper, 'client', wrapper)
    install_project_guard(raw, get_project_registry())
    return raw
