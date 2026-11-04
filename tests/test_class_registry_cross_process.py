"""Concurrent class writes lose no class and reuse no id (audit CC-2, Q-P1.3).

Eight spawned processes write one registry document through a stand-in that enforces
``if_seq_no`` atomically, as the server does. Before the document store the same race on
``class_registry.json`` lost a class in 20 of 20 runs.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from pathlib import Path
from typing import Any

import pytest
from _registry_doc_fake import FakeDocOpenSearch

from src.clients.curation_opensearch.registry import (
    MAX_REGISTRY_CLASSES,
    ClassRegistry,
    ClassRegistryError,
)
from src.core.exceptions import RequestRejectedError


def make_store(tmp_path: Path) -> ClassRegistry:
    return ClassRegistry(
        index='op_x__settings',
        classes_index='op_x__classes',
        legacy_path=tmp_path / 'class_registry.json',
        project='x',
    )


def _worker(
    docs: Any, seqs: Any, lock: Any, legacy: str, name: str, barrier: Any, shared: str
) -> None:
    store = ClassRegistry(
        index='op_x__settings',
        classes_index='op_x__classes',
        legacy_path=Path(legacy),
        project='x',
    )
    client = FakeDocOpenSearch(docs, seqs, lock)

    async def run() -> None:
        barrier.wait()
        await store.add_class(client, name)
        await store.ensure_class(client, shared)

    asyncio.run(run())


@pytest.mark.parametrize('repeat', range(5))
def test_eight_processes_add_eight_distinct_classes(tmp_path: Path, repeat: int) -> None:
    ctx = mp.get_context('spawn')
    with ctx.Manager() as manager:
        docs, seqs, lock = manager.dict(), manager.dict(), manager.Lock()
        barrier = ctx.Barrier(8)
        procs = [
            ctx.Process(
                target=_worker,
                args=(
                    docs,
                    seqs,
                    lock,
                    str(tmp_path / 'class_registry.json'),
                    f'c{i}',
                    barrier,
                    'shared',
                ),
            )
            for i in range(8)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
        assert all(p.exitcode == 0 for p in procs)

        snap = asyncio.run(make_store(tmp_path).snapshot(FakeDocOpenSearch(docs, seqs, lock)))
    names = sorted(c.class_name for c in snap.classes)
    assert names == sorted([f'c{i}' for i in range(8)] + ['shared'])
    ids = [c.class_id for c in snap.classes]
    assert len(set(ids)) == 9
    assert snap.next_class_id == 9


@pytest.mark.asyncio
async def test_ensure_class_twice_concurrently_returns_one_id(tmp_path: Path) -> None:
    store, client = make_store(tmp_path), FakeDocOpenSearch()
    (a, _), (b, _) = await asyncio.gather(
        store.ensure_class(client, 'wheel'), store.ensure_class(client, 'wheel')
    )
    assert a == b
    assert [c.class_name for c in (await store.snapshot(client)).classes] == ['wheel']


@pytest.mark.asyncio
async def test_add_class_past_the_bound_is_registry_full(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr('src.clients.curation_opensearch.registry.MAX_REGISTRY_CLASSES', 2)
    store, client = make_store(tmp_path), FakeDocOpenSearch()
    await store.add_class(client, 'a')
    await store.add_class(client, 'b')
    with pytest.raises(RequestRejectedError) as exc:
        await store.add_class(client, 'c')
    assert exc.value.status_code == 409
    assert exc.value.detail['error'] == 'registry_full'
    assert MAX_REGISTRY_CLASSES == 10_000


@pytest.mark.asyncio
async def test_six_forced_conflicts_give_registry_conflict(tmp_path: Path) -> None:
    store, client = make_store(tmp_path), FakeDocOpenSearch()
    await store.add_class(client, 'seed')
    client.force_conflicts = 6
    with pytest.raises(RequestRejectedError) as exc:
        await store.add_class(client, 'late')
    assert exc.value.detail['error'] == 'registry_conflict'


@pytest.mark.asyncio
async def test_conflicts_below_the_limit_retry_and_succeed(tmp_path: Path) -> None:
    store, client = make_store(tmp_path), FakeDocOpenSearch()
    await store.add_class(client, 'seed')
    client.force_conflicts = 4
    assert await store.add_class(client, 'late') == 1


@pytest.mark.asyncio
async def test_sequential_adds_give_ids_zero_to_n(tmp_path: Path) -> None:
    store, client = make_store(tmp_path), FakeDocOpenSearch()
    ids = [await store.add_class(client, f'k{i}') for i in range(6)]
    assert ids == list(range(6))


@pytest.mark.asyncio
async def test_duplicate_active_name_is_refused_by_add_not_ensure(tmp_path: Path) -> None:
    store, client = make_store(tmp_path), FakeDocOpenSearch()
    await store.add_class(client, 'dup')
    with pytest.raises(ClassRegistryError):
        await store.add_class(client, 'dup')
    assert (await store.ensure_class(client, 'dup'))[1] is False


@pytest.mark.asyncio
async def test_ids_are_never_reused_after_merge(tmp_path: Path) -> None:
    store, client = make_store(tmp_path), FakeDocOpenSearch()
    a = await store.add_class(client, 'a')
    b = await store.add_class(client, 'b')
    await store.merge_class(client, b, a)
    assert await store.add_class(client, 'c') == 2
