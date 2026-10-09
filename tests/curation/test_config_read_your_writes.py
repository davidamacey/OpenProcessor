"""Config-store writes are read-your-writes (#196): every mutation asks
OpenSearch to make itself visible to search before answering, so a clone
followed at once by a GET on another worker never sees a 404."""

from __future__ import annotations

from typing import Any

import pytest
from opensearchpy.exceptions import NotFoundError

from src.services.config_store import index as config_index
from src.services.config_store.index import activate, delete_config, save_config


class _LaggyClient:
    """Realtime GET, but search only sees docs once a write asked for
    ``refresh`` or an explicit ``indices.refresh`` ran (models NRT lag)."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.searchable: set[str] = set()
        self.writes: list[tuple[str, str, Any]] = []
        self.indices = self
        self._seq = 0

    async def refresh(self, index: str) -> None:
        del index
        self.searchable = set(self.docs)

    def _after_write(self, op: str, id_: str, refresh: Any) -> None:
        self.writes.append((op, id_, refresh))
        if refresh in ('wait_for', True, 'true'):
            self.searchable = set(self.docs)

    async def get(self, index: str, id: str, ignore: Any = None) -> dict[str, Any]:  # noqa: A002
        del index, ignore
        if id not in self.docs:
            raise NotFoundError(404, 'nf', {})
        return {'_source': dict(self.docs[id]), '_seq_no': self._seq, '_primary_term': 1}

    async def index(self, index: str, id: str, body: dict[str, Any], **kw: Any) -> None:  # noqa: A002
        del index
        self._seq += 1
        self.docs[id] = dict(body)
        self._after_write('index', id, kw.get('refresh'))

    async def update(self, index: str, id: str, body: dict[str, Any], **kw: Any) -> None:  # noqa: A002
        del index
        if id in self.docs:
            self.docs[id]['config_revision'] += 1
        else:
            self.docs[id] = dict(body['upsert'])
        self._after_write('update', id, kw.get('refresh'))

    async def delete(self, index: str, id: str, **kw: Any) -> None:  # noqa: A002
        del index
        self.docs.pop(id, None)
        self._after_write('delete', id, kw.get('refresh'))

    async def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        del index, body
        return {'hits': {'hits': []}}


@pytest.fixture(autouse=True)
def _clear_absent() -> None:
    config_index._absent_until.clear()


@pytest.mark.asyncio
async def test_save_config_writes_are_visible_to_search_without_a_reader_refresh() -> None:
    client = _LaggyClient()
    await save_config(client, 'idx', kind='prompt_pack', name='p', body={}, expected_revision=None)
    assert {'pack:p', 'pack:p@1', 'meta:config_revision'} <= client.searchable
    assert all(refresh == 'wait_for' for _, _, refresh in client.writes), client.writes


@pytest.mark.asyncio
async def test_delete_and_activate_writes_ask_for_refresh() -> None:
    client = _LaggyClient()
    await save_config(client, 'idx', kind='prompt_pack', name='p', body={}, expected_revision=None)
    client.writes.clear()
    await activate(client, 'idx', axis='prompt_pack', name='p', revision=1, expected_active=None)
    await delete_config(client, 'idx', kind='prompt_pack', name='p', expected_revision=1)
    assert client.writes
    assert all(refresh == 'wait_for' for _, _, refresh in client.writes), client.writes
    assert 'pack:p' not in client.searchable
