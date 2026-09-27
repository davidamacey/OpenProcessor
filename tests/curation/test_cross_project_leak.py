"""The isolation proof (projects_plan.md §10 P1): every scoped curation
route, called as project ``beta``, touches only ``beta``'s OpenSearch
indexes and never returns another project's ids or class names.

Also asserted: every served URL in a response (image, crop, artifact
links) is under ``beta``'s own prefix, so a client following it never
lands in another project.

Setup: three projects -- ``default`` (the env-named indexes), ``alpha``
and ``beta`` -- on a fake OpenSearch transport behind the real project
guard, each seeded with its own distinct items, images, labels and
classes. The routes are enumerated from the app's route table, so a route
added later is covered automatically; a new path parameter or a new
streaming route fails the test until it is mapped below.

Recording happens *before* the guard decides (``check_request`` is
wrapped), so an access the guard refused still counts as a leak attempt,
and so does one a route swallowed in a broad ``except``.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess  # nosec B404 - only patched to refuse, never called
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from fastapi.testclient import TestClient
from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError


if TYPE_CHECKING:
    from pathlib import Path


API = '/curation'
SCOPED = f'{API}/projects/{{project}}'

# Every path parameter a scoped route may carry, filled with *beta's* ids.
# A route with a parameter missing here fails the test ("unmapped route").
ROUTE_PARAMS: dict[str, str] = {
    'crop_id': 'beta-item-0001',
    'image_id': 'beta-img-0001',
    'class_id': '1',
    'cluster_id': '1',
    'job_id': 'beta-job-0001',
    'campaign_id': 'beta-campaign-0001',
    'name': 'beta-model',
    'model_name': 'beta-model',
    'tab': 'uncertainty',
    'alias': 'beta-source',
    'artifact': 'results.csv',
}

# Long-lived SSE streams: never called here (they would not return); the
# per-project event filtering they serve is proven by
# tests/projects/test_event_hub_project_filter.py.
STREAMING_ROUTES: frozenset[str] = frozenset(
    {
        f'{SCOPED}/events',
        f'{SCOPED}/pipeline/events',
    }
)

_ISOLATION_ERRORS = ('CrossProjectAccess', 'ProjectNotBound', 'internal_isolation_error')


def _docs(slug: str, class_name: str) -> dict[str, dict[str, dict[str, Any]]]:
    """One project's seed data, keyed by IndexRole value -> doc id -> doc.
    Every id and the class name embed ``slug`` so a leak is greppable."""
    item_id = f'{slug}-item-0001'
    image_id = f'{slug}-img-0001'
    item = {
        'crop_id': item_id,
        'image_id': image_id,
        'image_path': f'/data/{slug}/{image_id}.jpg',
        'class_id': 1,
        'class_name': class_name,
        'class_source': 'human',
        'validated': True,
        'bbox': [0.1, 0.1, 0.5, 0.5],
        'cluster_id': 1,
    }
    return {
        'items': {item_id: item},
        'images': {image_id: {'image_id': image_id, 'image_path': item['image_path']}},
        'labels_confirmed': {
            f'{slug}-label-0001': {'crop_id': item_id, 'class_id': 1, 'class_name': class_name}
        },
        'classes': {'1': {'class_id': 1, 'class_name': class_name}},
    }


class _FakeTransport:
    """The bottom of the fake: answers OpenSearch REST calls from a per-
    index doc store. Queries are not evaluated -- a search returns every
    doc of the index it names -- which is exactly what a leak test wants:
    any index a route reaches shows up in its response."""

    def __init__(self, store: dict[str, dict[str, dict[str, Any]]]) -> None:
        self.store = store

    def _hits(self, indices: list[str]) -> list[dict[str, Any]]:
        return [
            {'_index': idx, '_id': doc_id, '_source': doc, '_seq_no': 1, '_primary_term': 1}
            for idx in indices
            for doc_id, doc in self.store.get(idx, {}).items()
        ]

    async def perform_request(  # noqa: PLR0911 - one branch per OpenSearch endpoint shape
        self,
        method: str,
        url: str,
        params: Any = None,  # noqa: ARG002
        body: Any = None,
        **_kwargs: Any,
    ) -> Any:
        parts = [p for p in url.split('?')[0].split('/') if p]
        indices = parts[0].split(',') if parts and not parts[0].startswith('_') else []
        action = next((p for p in parts[1:] if p.startswith('_')), parts[0] if parts else '')
        if method == 'HEAD':
            return True
        if action == '_doc' and method == 'GET':
            doc_id = parts[2] if len(parts) > 2 else ''
            doc = self.store.get(indices[0], {}).get(doc_id)
            if doc is None:
                raise NotFoundError(404, 'not_found', {'found': False})
            return {
                '_index': indices[0],
                '_id': doc_id,
                'found': True,
                '_source': doc,
                '_seq_no': 1,
                '_primary_term': 1,
            }
        if action in ('_search', '_msearch') and parts[:2] != ['_search', 'scroll']:
            hits = self._hits(indices)
            resp = {
                'hits': {'total': {'value': len(hits), 'relation': 'eq'}, 'hits': hits},
                'aggregations': {},
            }
            if action == '_msearch':
                return {'responses': [resp]}
            return resp
        if action == '_count':
            return {'count': len(self._hits(indices))}
        if action == '_mget':
            return {'docs': self._mget(indices, body)}
        if action == '_bulk':
            return {'errors': False, 'items': []}
        if action in ('_update_by_query', '_delete_by_query'):
            return {'updated': 0, 'deleted': 0, 'total': 0, 'failures': []}
        if action == '_mapping':
            return {idx: {'mappings': {'properties': {}}} for idx in indices}
        if action in ('_doc', '_update', '_create'):
            return {'_id': parts[-1], 'result': 'updated', '_seq_no': 2, '_primary_term': 1}
        if parts[:1] == ['_cat']:
            return []
        return {'hits': {'total': {'value': 0}, 'hits': []}, 'acknowledged': True}

    def _mget(self, indices: list[str], body: Any) -> list[dict[str, Any]]:
        text = body.decode() if isinstance(body, bytes) else body
        spec = json.loads(text) if isinstance(text, str) else (text or {})
        out = []
        for entry in spec.get('docs', []):
            idx = entry.get('_index') or (indices[0] if indices else '')
            doc = self.store.get(idx, {}).get(entry.get('_id'))
            out.append(
                {
                    '_index': idx,
                    '_id': entry.get('_id'),
                    'found': doc is not None,
                    '_source': doc or {},
                    '_seq_no': 1,
                    '_primary_term': 1,
                }
            )
        return out

    async def close(self) -> None:
        return None


def _record(slug: str, **resources: Any) -> Any:
    from src.config.projects import ProjectRecord

    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug.title(),
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        **resources,
    )


def _write_registry(path: Path, class_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({'version': 1, 'classes': [{'id': 1, 'name': class_name}]}), encoding='utf-8'
    )


@pytest.fixture
def isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The real app, three seeded projects, a recording guard, and no way
    out of the process (no network, no subprocess)."""
    import src.config.curation as curation_config_mod
    from src.clients import curation_opensearch
    from src.config.curation import IndexRole, base_curation_config
    from src.config.projects import resources_for_new
    from src.core.dependencies import app_state, get_async_triton
    from src.routers.curation import _common
    from src.services.projects import guard, registry as registry_mod
    from src.services.projects.registry import ProjectRegistry, default_project_record

    for name, sub in {
        'STATE_DIR': 'state',
        'EXPORT_ROOT': 'default/exports',
        'UPLOAD_ROOT': 'default/uploads',
        'REGISTRY_PATH': 'default/class_registry.json',
        'BAKEOFF_EVAL_ROOT': 'default/bakeoff_eval',
        'CROP_CACHE_DIR': 'crop_cache',
        'TRAIN_JOBS_DIR': 'jobs',
        'BAKEOFF_JOBS_DIR': 'state/bakeoff_jobs',
        'PROJECTS_DATA_ROOT': 'projects',
    }.items():
        monkeypatch.setenv(f'OP_{name}', str(tmp_path / sub))
    monkeypatch.setenv('OP_EVENT_BUS', 'process')
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    monkeypatch.setattr(curation_opensearch, '_registries', {})
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', set())

    base = base_curation_config()
    records = {'default': default_project_record()}
    for slug in ('alpha', 'beta'):
        records[slug] = _record(slug, resources=resources_for_new(slug, base))

    class_names = {'default': 'default_cardinal', 'alpha': 'alpha_zebra', 'beta': 'beta_heron'}
    store: dict[str, dict[str, dict[str, Any]]] = {}
    for slug, record in records.items():
        for role_value, docs in _docs(slug, class_names[slug]).items():
            store[record.resources.indexes[IndexRole(role_value)]] = docs
        _write_registry(record.resources.class_registry_path, class_names[slug])

    registry = ProjectRegistry(lambda: None)
    registry._by_slug = {s: r for s, r in records.items() if s != 'default'}
    registry._revision = 1

    async def _fresh(self: Any) -> None:
        return None

    monkeypatch.setattr(ProjectRegistry, 'ensure_fresh', _fresh)
    registry_mod.set_project_registry(registry)

    raw = AsyncOpenSearch(hosts=['http://127.0.0.1:9'])
    raw.transport = _FakeTransport(store)  # type: ignore[assignment]
    guard.install_project_guard(raw, registry)

    from src.clients.opensearch import OpenSearchClient

    wrapper = OpenSearchClient(hosts=['http://127.0.0.1:9'])
    wrapper.client = raw
    monkeypatch.setattr(app_state, '_opensearch_client', wrapper)

    accesses: list[tuple[str | None, str, str, str]] = []  # (bound, method, url, index)
    real_check = guard.check_request

    def _recording_check(method: str, url: str, body: Any, snapshot: Any) -> None:
        from src.config.project_context import try_current_project

        bound = try_current_project()
        indices, action = guard._split_url(url)
        targets = set(indices or [])
        if action in guard._MULTI_DOC_ACTIONS:
            targets |= guard._indices_from_multi_doc_body(action, body)
        slug = bound.record.slug if bound is not None else None
        accesses.extend((slug, method, url, index) for index in targets)
        real_check(method, url, body, snapshot)

    monkeypatch.setattr(guard, 'check_request', _recording_check)

    def _no_network(*_a: Any, **_k: Any) -> Any:
        raise httpx.ConnectError('network disabled in the leak test')

    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', _no_network)
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', _no_network)

    def _no_subprocess(*_a: Any, **_k: Any) -> Any:
        raise OSError('subprocesses disabled in the leak test')

    monkeypatch.setattr(subprocess, 'Popen', _no_subprocess)
    monkeypatch.setattr(subprocess, 'run', _no_subprocess)

    import asyncio

    async def _no_exec(*_a: Any, **_k: Any) -> Any:
        raise OSError('subprocesses disabled in the leak test')

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', _no_exec)
    monkeypatch.setattr(asyncio, 'create_subprocess_shell', _no_exec)

    class _DeadTriton:
        async def is_server_live(self) -> bool:
            return False

        def __getattr__(self, name: str) -> Any:
            raise ConnectionError(f'triton disabled in the leak test ({name})')

    from src.main import app

    app.dependency_overrides[get_async_triton] = lambda: _DeadTriton()
    try:
        yield app, records, accesses
    finally:
        app.dependency_overrides.pop(get_async_triton, None)
        registry_mod.set_project_registry(None)


def _served_urls(value: Any) -> list[str]:
    """Every string in a JSON body that *is* a curation URL (starts with
    the API prefix) -- served image/crop/artifact links, not copy that
    merely mentions a path."""
    if isinstance(value, str):
        return [value] if value.startswith(f'{API}/') else []
    if isinstance(value, dict):
        return [u for v in value.values() for u in _served_urls(v)]
    if isinstance(value, list):
        return [u for v in value for u in _served_urls(v)]
    return []


def _scoped_routes(app: Any) -> list[tuple[str, str]]:
    """Every (method, path template) mounted under the scoped prefix."""
    out: list[tuple[str, str]] = []
    for route in app.routes:
        path = getattr(route, 'path', '')
        if not path.startswith(SCOPED):
            continue
        out.extend((method, path) for method in sorted(route.methods or ()) if method != 'HEAD')
    return out


def _fill(path: str) -> str:
    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == 'project':
            return 'beta'
        if name not in ROUTE_PARAMS:
            raise AssertionError(f'unmapped route {path}: add {name!r} to ROUTE_PARAMS')
        return ROUTE_PARAMS[name]

    return re.sub(r'\{(\w+)(?::\w+)?\}', _sub, path)


def _is_streaming(app: Any, path: str) -> bool:
    for route in app.routes:
        if getattr(route, 'path', '') == path:
            annotation = inspect.signature(route.endpoint).return_annotation
            return 'StreamingResponse' in str(annotation)
    return False


def test_every_scoped_route_stays_inside_the_bound_project(isolated_app: Any) -> None:
    app, records, accesses = isolated_app
    routes = _scoped_routes(app)
    assert len(routes) > 100, f'expected the full scoped surface, got {len(routes)} routes'

    streaming_unmapped = sorted({p for _m, p in routes if _is_streaming(app, p)} - STREAMING_ROUTES)
    assert not streaming_unmapped, f'unmapped streaming route(s): {streaming_unmapped}'

    beta_indexes = set(records['beta'].resources.indexes.values())
    foreign_indexes = {
        name: slug
        for slug, record in records.items()
        if slug != 'beta'
        for name in record.resources.indexes.values()
    }
    foreign_markers = (
        *(f'{slug}{kind}' for slug in ('alpha', 'default') for kind in ('-item', '-img', '-label')),
        'alpha_zebra',
        'default_cardinal',
    )

    leaks: list[str] = []
    beta_touching_routes = 0
    client = TestClient(app, raise_server_exceptions=False)
    for method, template in routes:
        if template in STREAMING_ROUTES:
            continue
        url = _fill(template)
        before = len(accesses)
        kwargs: dict[str, Any] = {} if method in ('GET', 'DELETE') else {'json': {}}
        response = client.request(method, url, **kwargs)
        route_accesses = accesses[before:]

        for bound, _verb, os_url, index in route_accesses:
            if index in foreign_indexes:
                leaks.append(
                    f'{method} {template}: bound={bound} reached {foreign_indexes[index]!r} '
                    f'index {index} ({os_url})'
                )
            if bound != 'beta':
                leaks.append(f'{method} {template}: OpenSearch call bound to {bound!r}')
        if any(index in beta_indexes for _b, _v, _u, index in route_accesses):
            beta_touching_routes += 1

        body = response.text
        leaks.extend(
            f'{method} {template}: response carries {marker!r}'
            for marker in foreign_markers
            if marker in body
        )
        if response.headers.get('content-type', '').startswith('application/json'):
            leaks.extend(
                f"{method} {template}: served URL {url!r} is not under beta's prefix"
                for url in _served_urls(response.json())
                if url != f'{API}/projects/beta' and not url.startswith(f'{API}/projects/beta/')
            )
        if response.status_code >= 500 and any(e in body for e in _ISOLATION_ERRORS):
            leaks.append(
                f'{method} {template}: isolation error {response.status_code} {body[:200]}'
            )

    assert not leaks, 'cross-project leak(s):\n' + '\n'.join(sorted(set(leaks)))
    # Not vacuous: a healthy share of the surface really read beta's data.
    assert beta_touching_routes >= 20, f'only {beta_touching_routes} routes reached beta indexes'


_VOLATILE_KEYS = re.compile(r'(_at|_ms|_s|^ts|^as_of|request_id|timestamp|duration.*|elapsed.*)$')


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if not _VOLATILE_KEYS.search(k)}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def test_unscoped_alias_serves_the_default_project(isolated_app: Any) -> None:
    """Back-compat: every parameterless scoped GET answers identically at
    ``/curation/X`` (the hidden alias) and ``/curation/projects/default/X``
    -- today's clients keep working against today's indexes."""
    app, _records, accesses = isolated_app
    client = TestClient(app, raise_server_exceptions=False)
    compared = 0
    for method, template in _scoped_routes(app):
        suffix = template[len(SCOPED) :]
        if method != 'GET' or '{' in suffix or template in STREAMING_ROUTES or not suffix:
            continue
        before = len(accesses)
        scoped = client.get(f'{API}/projects/default{suffix}')
        alias = client.get(f'{API}{suffix}')
        assert all(bound == 'default' for bound, *_ in accesses[before:]), suffix
        if suffix == '/stats':
            # P3: /stats is a global_router route (its own bind_path_project
            # dependency, per projects_plan.md §4) rather than part of the
            # scoped/alias double-mount, so the (P1, soon-removed per the
            # no-back-compat owner decision) alias never serves it.
            assert alias.status_code == 404, suffix
            continue
        assert alias.status_code == scoped.status_code, suffix
        if suffix in ('/health', '/events'):
            continue  # the global routes deliberately win at these two paths
        if scoped.headers.get('content-type', '').startswith('application/json'):
            assert _stable(alias.json()) == _stable(scoped.json()), suffix
        compared += 1
    assert compared >= 30
