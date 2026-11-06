"""An in-memory OpenSearch cluster for the index-lifecycle tests: creates, mapping and settings
puts, the schema stamp, close and open, and every write recorded so a test can assert on them."""

from __future__ import annotations

import copy
import unicodedata
from typing import Any


class NotFoundError(Exception):
    status_code = 404


class FakeIndices:
    def __init__(self, cluster: FakeCluster) -> None:
        self._c = cluster

    def _record(self, call: str, index: str) -> None:
        self._c.calls.append((call, index))

    async def exists(self, *, index: str) -> bool:
        self._record('indices.exists', index)
        return index in self._c.indexes

    async def create(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self._record('indices.create', index)
        self._c.writes.append(('create', index))
        if index in self._c.indexes:
            raise RuntimeError('resource_already_exists_exception')
        settings = {
            k: str(v)
            for k, v in body.get('settings', {}).get('index', {}).items()
            if k != 'analysis'
        }
        self._c.indexes[index] = {
            'properties': copy.deepcopy(body.get('mappings', {}).get('properties', {})),
            'settings': settings,
            'analysis': copy.deepcopy(
                body.get('settings', {}).get('index', {}).get('analysis', {})
            ),
            '_meta': {},
            'body': copy.deepcopy(body),
        }
        return {'acknowledged': True}

    async def get(self, *, index: str, **params: Any) -> dict[str, Any]:
        self._record('indices.get', index)
        out: dict[str, Any] = {}
        for name in index.split(','):
            if name not in self._c.indexes:
                if not params.get('ignore_unavailable'):
                    raise NotFoundError(name)
                continue
            idx = self._c.indexes[name]
            out[name] = {
                'mappings': {'_meta': idx['_meta'], 'properties': idx['properties']},
                'settings': {'index': dict(idx['settings'])},
            }
        return out

    async def get_mapping(self, *, index: str, **params: Any) -> dict[str, Any]:
        self._record('indices.get_mapping', index)
        out: dict[str, Any] = {}
        for name in index.split(','):
            if name not in self._c.indexes:
                if not params.get('ignore_unavailable'):
                    raise NotFoundError(name)
                continue
            meta = self._c.indexes[name]['_meta']
            out[name] = {'mappings': {'_meta': dict(meta)}} if meta else {}
        return out

    async def put_mapping(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self._record('indices.put_mapping', index)
        self._c.writes.append(('put_mapping', index))
        if index not in self._c.indexes:
            raise NotFoundError(index)
        idx = self._c.indexes[index]
        idx['properties'].update(copy.deepcopy(body.get('properties', {})))
        if '_meta' in body:
            idx['_meta'] = dict(body['_meta'])
        return {'acknowledged': True}

    async def get_settings(self, *, index: str) -> dict[str, Any]:
        self._record('indices.get_settings', index)
        if index not in self._c.indexes:
            raise NotFoundError(index)
        idx = self._c.indexes[index]
        index_settings: dict[str, Any] = dict(idx['settings'])
        if idx['analysis']:
            index_settings['analysis'] = copy.deepcopy(idx['analysis'])
        return {index: {'settings': {'index': index_settings}}}

    async def put_settings(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self._record('indices.put_settings', index)
        self._c.writes.append(('put_settings', index))
        if self._c.refuse_analysis and 'analysis' in body.get('index', {}):
            raise RuntimeError('illegal_argument_exception: settings refused')
        idx = self._c.indexes[index]
        for key, value in body.get('index', {}).items():
            if key == 'analysis':
                if index not in self._c.closed:
                    raise RuntimeError('analysis settings need a closed index')
                idx['analysis'] = copy.deepcopy(value)
            else:
                idx['settings'][key] = str(value)
        return {'acknowledged': True}

    async def close(self, *, index: str) -> dict[str, Any]:
        self._record('indices.close', index)
        self._c.writes.append(('close', index))
        self._c.closed.add(index)
        return {'acknowledged': True}

    async def open(self, *, index: str) -> dict[str, Any]:
        self._record('indices.open', index)
        self._c.writes.append(('open', index))
        self._c.closed.discard(index)
        return {'acknowledged': True}

    async def stats(self, *, index: str, metric: str) -> dict[str, Any]:  # noqa: ARG002
        self._record('indices.stats', index)
        primaries = {'docs': {'count': 3}, 'store': {'size_in_bytes': 4096}}
        return {'_all': {'primaries': primaries}}

    async def analyze(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self._record('indices.analyze', index)
        if body['normalizer'] not in self._c.indexes[index]['analysis'].get('normalizer', {}):
            raise RuntimeError('no such normalizer')
        folded = unicodedata.normalize('NFKD', body['text']).encode('ascii', 'ignore').decode()
        return {'tokens': [{'token': ''.join(c for c in folded if c.isalnum()).upper()}]}


class FakeClusterApi:
    def __init__(self, cluster: FakeCluster) -> None:
        self._c = cluster

    async def health(self, *, index: str, wait_for_status: str, timeout: str) -> dict[str, Any]:  # noqa: ARG002
        self._c.calls.append(('cluster.health', index))
        return {'status': 'yellow', 'timed_out': False}


class FakeTransport:
    def __init__(self, cluster: FakeCluster) -> None:
        self._c = cluster

    async def perform_request(self, method: str, url: str, **_: Any) -> dict[str, Any]:
        self._c.calls.append((f'transport {method}', url))
        return {}


class FakeCluster:
    """One cluster. ``writes`` lists only the requests that change it."""

    def __init__(self) -> None:
        self.indexes: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self.writes: list[tuple[str, str]] = []
        self.closed: set[str] = set()
        self.refuse_analysis = False
        self.indices = FakeIndices(self)
        self.cluster = FakeClusterApi(self)
        self.transport = FakeTransport(self)

    def is_open(self, index: str) -> bool:
        return index not in self.closed


class FakeGlobalClient:
    """The ``OpenSearchClient`` of the deployment-wide ``visual_search_*`` indexes."""

    NAMES = ('visual_search_global', 'visual_search_vehicles')

    def __init__(self, cluster: FakeCluster) -> None:
        self._c = cluster

    async def create_all_indexes(self, force_recreate: bool = False) -> dict[str, bool]:  # noqa: ARG002
        for name in self.NAMES:
            if name not in self._c.indexes:
                self._c.indexes[name] = {
                    'properties': {},
                    'settings': {},
                    'analysis': {},
                    '_meta': {},
                }
                self._c.writes.append(('create', name))
        return dict.fromkeys(self.NAMES, True)
