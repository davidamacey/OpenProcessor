"""The class registry document against the real server, which enforces ``if_seq_no`` itself.

The harness runs the API with one worker (some cancel endpoints and the SSE hub keep in-process
state), so the cross-process race runs at the store level: 8 spawned processes call ``add_class``
with distinct names and then ``ensure_class`` with one shared name, straight against the
harness OpenSearch. Then one HTTP sequence checks the routes read the same document.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import uuid
from pathlib import Path
from typing import Any

import pytest
from opensearchpy import AsyncOpenSearch

from src.clients.curation_opensearch.registry import REGISTRY_DOC_ID, ClassRegistry

from .conftest import INDEXES, OPENSEARCH_URL


pytestmark = pytest.mark.live

REPEATS = 5
PROCESSES = 8


def _store() -> ClassRegistry:
    return ClassRegistry(
        index=INDEXES['settings'],
        classes_index=INDEXES['classes'],
        legacy_path=Path('/nonexistent/class_registry.json'),
        project='default',
    )


def _worker(prefix: str, number: int, barrier: Any) -> None:
    async def run() -> None:
        client = AsyncOpenSearch(hosts=[OPENSEARCH_URL])
        try:
            store = _store()
            barrier.wait()
            await store.add_class(client, f'{prefix}_{number}')
            await store.ensure_class(client, f'{prefix}_shared')
        finally:
            await client.close()

    asyncio.run(run())


def _race(prefix: str) -> None:
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(PROCESSES)
    procs = [ctx.Process(target=_worker, args=(prefix, i, barrier)) for i in range(PROCESSES)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]


async def _snapshot() -> Any:
    client = AsyncOpenSearch(hosts=[OPENSEARCH_URL])
    try:
        return await _store().snapshot(client)
    finally:
        await client.close()


@pytest.mark.parametrize('repeat', range(REPEATS))
def test_eight_processes_add_eight_distinct_classes_and_one_shared(repeat: int) -> None:
    prefix = f'live_race_{uuid.uuid4().hex[:8]}'

    _race(prefix)

    doc = asyncio.run(_snapshot())
    mine = [c for c in doc.classes if c.class_name.startswith(prefix)]
    assert len(mine) == PROCESSES + 1, [c.class_name for c in mine]
    assert len({c.class_id for c in mine}) == PROCESSES + 1
    assert len({c.class_id for c in doc.classes}) == len(doc.classes), 'an id was handed out twice'


def test_the_routes_read_and_write_the_same_document(api_client: Any, opensearch: Any) -> None:
    created = api_client.post('/classes', json={'name': 'live_doc_probe', 'group': 'harness'})
    assert created.status_code == 201, created.text
    class_id = created.json()['class_id']

    updated = api_client.put(f'/classes/{class_id}', json={'hotkey_letter': 'q', 'group': 'moved'})
    assert updated.status_code == 200, updated.text

    listed = {c['class_id']: c for c in api_client.get('/classes').json()['classes']}
    assert listed[class_id]['hotkey_letter'] == 'q'
    assert listed[class_id]['group'] == 'moved'

    stored = opensearch.get(f'/{INDEXES["settings"]}/_doc/{REGISTRY_DOC_ID}').json()
    entries = {c['class_id']: c for c in stored['_source']['class_registry']['classes']}
    assert entries[class_id]['class_name'] == 'live_doc_probe'
    assert entries[class_id]['hotkey_letter'] == 'q'
