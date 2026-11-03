"""Creating and stamping a project's indexes overlaps the OpenSearch round trips (#439).

Each create or mapping put is a cluster-state update, a second or more on a loaded node;
six of them one after the other made ``POST /projects`` take 13 to 20 s.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.clients.curation_opensearch.lifecycle import create_curation_indexes, curation_index_names
from src.services.curation import index_bootstrap


class _Barrier:
    """Every call waits until ``n`` calls are in flight; a sequential caller times out."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.inflight = 0
        self.all_in = asyncio.Event()
        self.calls: list[str] = []

    async def arrive(self, index: str) -> None:
        self.calls.append(index)
        self.inflight += 1
        if self.inflight >= self.n:
            self.all_in.set()
        await asyncio.wait_for(self.all_in.wait(), timeout=2)


class _Indices:
    def __init__(self, barrier: _Barrier) -> None:
        self.barrier = barrier

    async def exists(self, **_: Any) -> bool:
        return False

    async def create(self, index: str, **_: Any) -> dict[str, bool]:
        await self.barrier.arrive(index)
        return {'acknowledged': True}

    async def put_mapping(self, index: str, **_: Any) -> dict[str, bool]:
        await self.barrier.arrive(index)
        return {'acknowledged': True}


class _Client:
    def __init__(self, barrier: _Barrier) -> None:
        self.indices = _Indices(barrier)


@pytest.mark.asyncio
async def test_indexes_are_created_concurrently() -> None:
    names = curation_index_names()
    barrier = _Barrier(len(names))
    results = await create_curation_indexes(_Client(barrier))  # type: ignore[arg-type]
    assert results == dict.fromkeys(names, True)
    assert sorted(barrier.calls) == sorted(names)


@pytest.mark.asyncio
async def test_the_schema_stamp_is_written_concurrently() -> None:
    names = curation_index_names()
    barrier = _Barrier(len(names))
    await index_bootstrap._stamp_schema(_Client(barrier), names)
    assert sorted(barrier.calls) == sorted(names)
