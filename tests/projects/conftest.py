"""Fixtures shared by tests/projects/*.

Deliberately does NOT reuse tests/conftest.py's autouse default-project
bind (that fixture does not exist yet -- it lands with commit 4's route
mounting); every test here binds explicitly.
"""

from __future__ import annotations

from typing import Any

import pytest


class FakeRegistryOpenSearch:
    """Just enough of AsyncOpenSearch for ``ProjectRegistry``/
    ``bootstrap_default_project``: ``get``/``index``/``search`` on one
    flat doc store, keyed by id."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    async def get(self, *, index: str, id: str) -> dict[str, Any]:  # noqa: A002, ARG002
        if id not in self.docs:
            return {'found': False, '_id': id}
        return {'found': True, '_id': id, '_source': self.docs[id]}

    async def index(self, *, index: str, id: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: A002, ARG002
        self.docs[id] = body
        return {'_id': id, 'result': 'created'}

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        prefix = (((body.get('query') or {}).get('prefix') or {}).get('_id')) or ''
        hits = [
            {'_id': doc_id, '_source': doc}
            for doc_id, doc in self.docs.items()
            if doc_id.startswith(prefix)
        ]
        return {'hits': {'hits': hits}}


@pytest.fixture
def fake_registry_client() -> FakeRegistryOpenSearch:
    return FakeRegistryOpenSearch()
