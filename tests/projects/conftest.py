"""Fixtures shared by tests/projects/*.

Tests here that exercise binding itself opt out of tests/conftest.py's
autouse default-project bind with ``@pytest.mark.unbound``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


class _FakeIndices:
    def __init__(self) -> None:
        self.created: dict[str, dict[str, Any]] = {}

    async def exists(self, *, index: str) -> bool:
        return index in self.created

    async def create(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.created[index] = body
        return {'acknowledged': True}


class FakeRegistryOpenSearch:
    """Just enough of AsyncOpenSearch for ``ProjectRegistry``/
    ``bootstrap_default_project``: ``get``/``index``/``search`` on one
    flat doc store, keyed by id, with seq_no OCC like the real thing."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.seq: dict[str, int] = {}
        self.indices = _FakeIndices()

    async def get(self, *, index: str, id: str) -> dict[str, Any]:  # noqa: A002, ARG002
        await asyncio.sleep(0)  # a real read yields; lets concurrent writers interleave
        if id not in self.docs:
            return {'found': False, '_id': id}
        return {
            'found': True,
            '_id': id,
            '_source': self.docs[id],
            '_seq_no': self.seq[id],
            '_primary_term': 1,
        }

    async def index(
        self,
        *,
        index: str,  # noqa: ARG002 - mirrors the client signature
        id: str,  # noqa: A002
        body: dict[str, Any],
        op_type: str | None = None,
        if_seq_no: int | None = None,
        if_primary_term: int | None = None,  # noqa: ARG002 - one term in the fake
    ) -> dict[str, Any]:
        from opensearchpy.exceptions import ConflictError

        if op_type == 'create' and id in self.docs:
            raise ConflictError(409, 'version_conflict_engine_exception', {})
        if if_seq_no is not None and self.seq.get(id) != if_seq_no:
            raise ConflictError(409, 'version_conflict_engine_exception', {})
        self.docs[id] = body
        self.seq[id] = self.seq.get(id, 0) + 1
        return {'_id': id, 'result': 'created'}

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        prefix = (((body.get('query') or {}).get('prefix') or {}).get('_id')) or ''
        hits: list[dict[str, Any]] = [
            {'_id': doc_id, '_source': doc}
            for doc_id, doc in sorted(self.docs.items())
            if doc_id.startswith(prefix)
        ]
        after = body.get('search_after')
        if after:
            hits = [h for h in hits if h['_source']['slug'] > after[0]]
        hits = hits[: body.get('size', 10)]
        for hit in hits:
            hit['sort'] = [hit['_source']['slug']]
        return {'hits': {'hits': hits}}


@pytest.fixture
def fake_registry_client() -> FakeRegistryOpenSearch:
    return FakeRegistryOpenSearch()
