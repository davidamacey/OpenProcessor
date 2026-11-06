"""The committed 0.6.0 fixture as a fake OpenSearch plus a directory tree.

``install(tmp_path)`` writes the fixture files under ``tmp_path`` (replacing ``{ROOT}``), and
returns a :class:`V060Install` with the fake client, the layout and a storage store. The fake
implements only what the upgrade issues and records every call, so a test can assert the export
issued reads only.
"""

from __future__ import annotations

import base64
import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opensearchpy.exceptions import NotFoundError

from src.clients.storage.roots import RootedStore
from src.services.upgrade.from_060.admin_client import ControlReader, ControlWriter
from src.services.upgrade.from_060.layout import Layout
from src.services.upgrade.from_060.sources import JOB_FAMILIES


FIXTURE = Path(__file__).parent / 'fixtures' / 'v060'
READ_CALLS = frozenset(
    {
        'get',
        'search',
        'count',
        'tasks.get',
        'indices.exists',
        'indices.get_mapping',
        'indices.get_settings',
    }
)


def _iso_to_epoch(stamp: str) -> float:
    import datetime as dt

    return dt.datetime.fromisoformat(stamp).timestamp()


def matches(query: dict[str, Any], doc: dict[str, Any]) -> bool:
    """The few query shapes the upgrade sends: match_all, exists, bool must / must_not."""
    if 'match_all' in query:
        return True
    if 'exists' in query:
        return doc.get(query['exists']['field']) is not None
    clause = query['bool']
    return all(matches(q, doc) for q in clause.get('must', [])) and not any(
        matches(q, doc) for q in clause.get('must_not', [])
    )


class _Indices:
    def __init__(self, fake: FakeOpenSearch) -> None:
        self._fake = fake

    async def exists(self, index: str) -> bool:
        self._fake.calls.append('indices.exists')
        return index in self._fake.indexes

    async def get_mapping(self, index: str) -> dict[str, Any]:
        self._fake.calls.append('indices.get_mapping')
        return {index: {'mappings': copy.deepcopy(self._fake.index(index)['mappings'])}}

    async def get_settings(self, index: str) -> dict[str, Any]:
        self._fake.calls.append('indices.get_settings')
        settings = copy.deepcopy(self._fake.index(index)['settings'])
        settings.setdefault('index', {}).update(
            uuid='abc', creation_date='1', version={'created': '1'}, provided_name=index
        )
        return {index: {'settings': settings}}

    async def create(self, index: str, body: dict[str, Any]) -> None:
        self._fake.calls.append('indices.create')
        if index in self._fake.indexes:
            raise ValueError(f'{index} exists')
        self._fake.indexes[index] = {
            'mappings': body['mappings'],
            'settings': body['settings'],
            'docs': {},
        }

    async def delete(self, index: str) -> None:
        self._fake.calls.append('indices.delete')
        if index not in self._fake.indexes:
            raise NotFoundError(404, 'index_not_found_exception', index)
        del self._fake.indexes[index]

    async def refresh(self, index: str) -> None:  # noqa: ARG002  # mirrors the client signature
        self._fake.calls.append('indices.refresh')


class _Tasks:
    def __init__(self, fake: FakeOpenSearch) -> None:
        self._fake = fake

    async def get(self, task_id: str) -> dict[str, Any]:  # noqa: ARG002  # mirrors the client signature
        self._fake.calls.append('tasks.get')
        return {'completed': True, 'response': {'updated': self._fake.last_updated}}


@dataclass
class FakeOpenSearch:
    indexes: dict[str, dict[str, Any]]
    calls: list[str] = field(default_factory=list)
    last_updated: int = 0

    def __post_init__(self) -> None:
        self.indices = _Indices(self)
        self.tasks = _Tasks(self)

    def index(self, name: str) -> dict[str, Any]:
        if name not in self.indexes:
            raise NotFoundError(404, 'index_not_found_exception', name)
        return self.indexes[name]

    def docs(self, name: str) -> dict[str, Any]:
        docs: dict[str, Any] = self.index(name)['docs']
        return docs

    async def get(self, index: str, id: str) -> dict[str, Any]:  # noqa: A002
        self.calls.append('get')
        docs = self.docs(index)
        if id not in docs:
            raise NotFoundError(404, 'not_found', id)
        return {'_id': id, '_source': copy.deepcopy(docs[id])}

    async def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append('search')
        ids = list(self.docs(index))
        start = body['search_after'][0] + 1 if 'search_after' in body else 0
        page = ids[start : start + body['size']]
        hits = [
            {'_id': i, '_source': copy.deepcopy(self.docs(index)[i]), 'sort': [ids.index(i)]}
            for i in page
        ]
        return {'hits': {'hits': hits}}

    async def count(self, index: str, body: dict[str, Any]) -> dict[str, int]:
        self.calls.append('count')
        return {'count': sum(1 for d in self.docs(index).values() if matches(body['query'], d))}

    async def update_by_query(self, index: str, body: dict[str, Any], **_: Any) -> dict[str, str]:
        self.calls.append('update_by_query')
        assert 'embedding_state' in body['script']['source']
        n = 0
        for doc in self.docs(index).values():
            if matches(body['query'], doc):
                doc['embedding_state'] = 'embedded'
                n += 1
        self.last_updated = n
        return {'task': 'fake:1'}

    async def bulk(self, body: list[Any], refresh: bool = False) -> dict[str, Any]:  # noqa: ARG002
        self.calls.append('bulk')
        for action, source in zip(body[::2], body[1::2], strict=True):
            meta = action['index']
            self.docs(meta['_index'])[meta['_id']] = source
        return {'errors': False, 'items': []}


def layout_under(jobs_root: Path, model_repo: Path) -> Layout:
    """The default 0.6.0 layout below one jobs directory."""
    return Layout(
        job_family_dirs={
            name: jobs_root / Path(default).name for name, _var, default in JOB_FAMILIES
        },
        combine_dir=jobs_root / 'combine',
        model_repo=model_repo,
    )


@dataclass
class V060Install:
    root: Path
    os: FakeOpenSearch
    layout: Layout
    store: RootedStore

    @property
    def reader(self) -> ControlReader:
        return ControlReader(self.os)

    @property
    def writer(self) -> ControlWriter:
        return ControlWriter(self.os)

    def path(self, rel: str) -> Path:
        return self.root / rel


def _substitute(value: Any, root: str) -> Any:
    if isinstance(value, str):
        return value.replace('{ROOT}', root)
    if isinstance(value, list):
        return [_substitute(v, root) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, root) for k, v in value.items()}
    return value


def install(root: Path) -> V060Install:
    """The fixture, materialized under ``root``."""
    raw = json.loads((FIXTURE / 'opensearch.json').read_text(encoding='utf-8'))
    indexes = _substitute(raw, str(root))
    for item in json.loads((FIXTURE / 'files.json').read_text(encoding='utf-8')):
        path = root / _substitute(item['path'], str(root))
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (
            base64.b64decode(item['data']) if item['enc'] == 'b64' else item['data'].encode('utf-8')
        )
        path.write_bytes(data)
        stamp = _iso_to_epoch(item['mtime'])
        os.utime(path, (stamp, stamp))
    (root / 'storage' / 'projects').mkdir(parents=True, exist_ok=True)
    store = RootedStore(
        {
            'projects': root / 'storage' / 'projects',
            'upgrade': root / 'storage' / 'upgrade',
            'models': root / 'models',
        }
    )
    return V060Install(
        root=root,
        os=FakeOpenSearch(indexes),
        layout=layout_under(root / 'jobs', root / 'models'),
        store=store,
    )
