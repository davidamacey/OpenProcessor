"""The per-project index bootstrap skips the mapping migrations when the indexes already
carry the current schema marker (#376).

Every uvicorn worker bootstraps every active project at startup, and each recycled
worker does it again. The full sequence is an exists check per index plus some sixty
additive ``PUT _mapping`` calls, all no-ops on a migrated index. After a full run the
bootstrap stamps each index's mapping ``_meta`` with a fingerprint of the index bodies
and the migration list; a later process reads the marker of every index in one request
and, when it is current, runs no migration. A new project, a project whose indexes
predate the marker, a missing index or a changed schema still get the full run.

The fake client checks every request against the project guard, as the guarded
transport would.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from src.clients.curation_opensearch.ensure_overlay_fields import inner_result_window
from src.clients.curation_opensearch.lifecycle import INDEX_BODIES, curation_index_names
from src.config import get_curation_config, index_name
from src.services.curation import index_bootstrap
from src.services.projects.guard_policy import check_request
from src.services.projects.registry import get_project_registry


class _NotFoundError(Exception):
    status_code = 404


class _Indices:
    def __init__(self, cluster: _Cluster) -> None:
        self._c = cluster

    async def get(self, *, index: str, **params: Any) -> dict[str, Any]:
        self._c.request('GET', f'/{index}', params=params)
        out: dict[str, Any] = {}
        for name in index.split(','):
            if name not in self._c.indexes:
                if not params.get('ignore_unavailable'):
                    raise _NotFoundError(name)
                continue
            idx = self._c.indexes[name]
            out[name] = {
                'mappings': {'_meta': idx['_meta'], 'properties': idx['properties']},
                'settings': {'index': dict(idx['settings'])},
            }
        return out

    async def exists(self, *, index: str) -> bool:
        self._c.request('HEAD', f'/{index}')
        return index in self._c.indexes

    async def create(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self._c.request('PUT', f'/{index}', body=body)
        self._c.indexes[index] = {
            'properties': copy.deepcopy(body.get('mappings', {}).get('properties', {})),
            'settings': {},
            '_meta': {},
        }
        return {'acknowledged': True}

    async def put_mapping(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self._c.request('PUT', f'/{index}/_mapping', body=body)
        if index not in self._c.indexes:
            raise _NotFoundError(index)
        self._c.indexes[index]['properties'].update(copy.deepcopy(body.get('properties', {})))
        if '_meta' in body:
            self._c.indexes[index]['_meta'] = dict(body['_meta'])
        return {'acknowledged': True}

    async def get_mapping(self, *, index: str) -> dict[str, Any]:
        self._c.request('GET', f'/{index}/_mapping')
        return {index: {'mappings': {'properties': self._c.indexes[index]['properties']}}}

    async def get_settings(self, *, index: str) -> dict[str, Any]:
        self._c.request('GET', f'/{index}/_settings')
        return {index: {'settings': {'index': dict(self._c.indexes[index]['settings'])}}}

    async def put_settings(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self._c.request('PUT', f'/{index}/_settings', body=body)
        self._c.indexes[index]['settings'].update(
            {k: str(v) for k, v in body.get('index', {}).items()}
        )
        return {'acknowledged': True}


class _Cluster:
    """The indexes of one OpenSearch cluster, shared by every worker process."""

    def __init__(self) -> None:
        self.indexes: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self.indices = _Indices(self)

    def request(self, method: str, url: str, *, body: Any = None, params: Any = None) -> None:
        check_request(method, url, body, get_project_registry().snapshot(), params=params)
        self.calls.append((method, url))

    async def count(self, *, index: str) -> dict[str, int]:
        self.request('GET', f'/{index}/_count')
        return {'count': 1}

    def mapping_puts(self) -> int:
        return sum(1 for m, u in self.calls if m == 'PUT' and u.endswith('/_mapping'))


@pytest.fixture(autouse=True)
def _fresh_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', set())


def _new_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another worker process: nothing bootstrapped in memory yet."""
    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', set())


def _slug() -> str:
    return get_curation_config().project_slug


def _marker(cluster: _Cluster, name: str) -> Any:
    return cluster.indexes[name]['_meta'].get(index_bootstrap.SCHEMA_META_KEY)


@pytest.mark.asyncio
async def test_a_new_project_is_created_migrated_and_stamped() -> None:
    cluster = _Cluster()

    assert await index_bootstrap.ensure_indexes(cluster) is True

    names = curation_index_names()
    assert set(cluster.indexes) == set(names)
    assert cluster.mapping_puts() > len(names)
    for name in names:
        assert _marker(cluster, name) == index_bootstrap.schema_fingerprint()
    assert _slug() in index_bootstrap._INDEXES_BOOTSTRAPPED


@pytest.mark.asyncio
async def test_a_later_process_reads_the_marker_and_runs_no_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = _Cluster()
    await index_bootstrap.ensure_indexes(cluster)
    cold = len(cluster.calls)
    cluster.calls.clear()
    _new_process(monkeypatch)

    assert await index_bootstrap.ensure_indexes(cluster) is True

    # One read of every index's marker and settings, and the classes self-heal count.
    assert [m for m, _ in cluster.calls] == ['GET', 'GET']
    assert cluster.calls[0][1] == '/' + ','.join(curation_index_names())
    assert cluster.calls[1][1].endswith('/_count')
    assert cold >= 30, f'the full bootstrap made {cold} calls'
    assert _slug() in index_bootstrap._INDEXES_BOOTSTRAPPED
    items = get_curation_config().items_index
    assert inner_result_window(items) == int(get_curation_config().region_max_boxes_per_write)


@pytest.mark.asyncio
async def test_indexes_from_before_the_marker_are_migrated_then_stamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = _Cluster()
    await index_bootstrap.ensure_indexes(cluster)
    for idx in cluster.indexes.values():
        idx['_meta'] = {}
        idx['properties'].pop('ingest_write_id', None)
    cluster.calls.clear()
    _new_process(monkeypatch)

    await index_bootstrap.ensure_indexes(cluster)

    assert cluster.mapping_puts() > len(cluster.indexes)
    images = get_curation_config().images_index
    assert 'ingest_write_id' in cluster.indexes[images]['properties']
    assert _marker(cluster, images) == index_bootstrap.schema_fingerprint()


@pytest.mark.asyncio
async def test_a_changed_schema_runs_the_migrations_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = _Cluster()
    await index_bootstrap.ensure_indexes(cluster)
    cluster.calls.clear()
    _new_process(monkeypatch)
    monkeypatch.setattr(index_bootstrap, 'schema_fingerprint', lambda: 'next-release')

    await index_bootstrap.ensure_indexes(cluster)

    assert cluster.mapping_puts() > len(cluster.indexes)
    assert {_marker(cluster, n) for n in cluster.indexes} == {'next-release'}


@pytest.mark.asyncio
async def test_a_missing_index_is_recreated(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = _Cluster()
    await index_bootstrap.ensure_indexes(cluster)
    classes = get_curation_config().classes_index
    del cluster.indexes[classes]
    _new_process(monkeypatch)

    await index_bootstrap.ensure_indexes(cluster)

    assert classes in cluster.indexes
    assert _marker(cluster, classes) == index_bootstrap.schema_fingerprint()


@pytest.mark.asyncio
async def test_a_lowered_inner_result_window_is_raised_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = _Cluster()
    await index_bootstrap.ensure_indexes(cluster)
    items = get_curation_config().items_index
    cluster.indexes[items]['settings']['max_inner_result_window'] = '1'
    cluster.calls.clear()
    _new_process(monkeypatch)

    await index_bootstrap.ensure_indexes(cluster)

    wanted = int(get_curation_config().region_max_boxes_per_write)
    assert int(cluster.indexes[items]['settings']['max_inner_result_window']) == wanted
    assert ('PUT', f'/{items}/_settings') in cluster.calls


@pytest.mark.asyncio
async def test_a_failed_migration_stamps_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failed(_os: Any) -> dict[str, Any]:
        return {'acknowledged': False, 'error': 'mapping conflict'}

    monkeypatch.setattr(index_bootstrap, 'ensure_items_score_fields', failed)
    cluster = _Cluster()

    assert await index_bootstrap.ensure_indexes(cluster) is False

    assert all(not idx['_meta'] for idx in cluster.indexes.values())
    assert _slug() not in index_bootstrap._INDEXES_BOOTSTRAPPED


@pytest.mark.asyncio
async def test_an_unreadable_marker_falls_back_to_the_full_bootstrap() -> None:
    cluster = _Cluster()

    async def down(**_kw: Any) -> Any:
        msg = 'connection reset'
        raise ConnectionError(msg)

    cluster.indices.get = down  # type: ignore[method-assign]

    assert await index_bootstrap.ensure_indexes(cluster) is True
    assert set(cluster.indexes) == set(curation_index_names())


def test_every_migration_field_is_in_the_index_bodies() -> None:
    """The fingerprint covers the bodies, so a migration may only add what a new index
    gets from its body: a migration field missing there would never change the marker."""
    import asyncio

    cluster = _Cluster()
    for name in curation_index_names():
        cluster.indexes[name] = {'properties': {}, 'settings': {}, '_meta': {}}
    for _, migrate in index_bootstrap._migrations():
        asyncio.run(migrate(cluster))
    cfg = get_curation_config()
    bodies: dict[str, dict[str, Any]] = {}
    for role, body in INDEX_BODIES.items():
        bodies.setdefault(index_name(cfg, role), {}).update(body['mappings']['properties'])
    for name, idx in cluster.indexes.items():
        for field, spec in idx['properties'].items():
            assert bodies[name].get(field) == spec, f'{name}.{field} is not in its body'
