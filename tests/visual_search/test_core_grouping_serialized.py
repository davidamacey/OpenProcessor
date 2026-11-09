"""Two concurrent core ingest batches that hold one near-duplicate pair end in one group.

Before ND-C1 each request grouped its own images (``search``, then ``update``): two requests that
each saw the other's image ungrouped each created a group (#473 part b). Now the request only
writes a ``near_dup_pending`` row (level ``core``, no project) and the single-writer
``near_dup_core_group`` job groups the pending rows, so the pair lands in one group whatever the
requests' interleaving.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest
from fixtures.job_rows import sql

from src.db.session import async_session_factory, core_tx
from src.services.near_duplicates import core_group, core_pending
from src.services.visual_search import VisualSearchService
from visual_search.core_fakes import INDEX, FakeCoreOpenSearch


pytestmark = [pytest.mark.db, pytest.mark.asyncio]

A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
B = np.array([0.9999, 0.0141, 0.0], dtype=np.float32)
B = B / np.linalg.norm(B)
VECTORS = {b'bytes-a': A, b'bytes-b': B}


@pytest.fixture
def opensearch(monkeypatch: pytest.MonkeyPatch) -> FakeCoreOpenSearch:
    fake = FakeCoreOpenSearch()

    def infer(images: list[bytes], **_kw: Any) -> list[dict[str, Any]]:
        return [
            {'num_dets': 0, 'num_faces': 0, 'global_embedding': VECTORS[i].tolist()} for i in images
        ]

    monkeypatch.setattr('src.services.visual_search.ingest_batch.infer_batch', infer)
    monkeypatch.setattr('src.clients.triton_client.get_sync_triton_client', lambda _url: object())
    return fake


def _service(fake: FakeCoreOpenSearch) -> VisualSearchService:
    service = VisualSearchService.__new__(VisualSearchService)
    service.opensearch = fake  # type: ignore[assignment]
    service.inference = None  # type: ignore[assignment]
    return service


async def _ingest(service: VisualSearchService, data: bytes, image_id: str) -> dict[str, Any]:
    return await service.ingest_batch(
        [(data, image_id, None)],
        skip_duplicates=False,
        enable_ocr=False,
        enable_detection=False,
        enable_faces=False,
        enable_clip=True,
    )


def _pending(pg: Any, prefix: str) -> list[tuple[str, str]]:
    return sql(
        pg,
        "SELECT member_id, reason FROM near_dup_pending WHERE level = 'core' "
        'AND member_id LIKE %s ORDER BY member_id',
        f'{prefix}%',
    )


async def _drain(pg: Any, fake: FakeCoreOpenSearch) -> Any:
    job = sql(
        pg,
        'INSERT INTO jobs (project_id, kind, queue, status, max_queue_wait, requested_by) '
        "VALUES (NULL, 'noop', 'jobs', 'queued', interval '1 hour', 'test') RETURNING id",
    )[0][0]

    @asynccontextmanager
    async def scope() -> Any:
        async with async_session_factory()() as session, core_tx(session, 'test drain'):
            yield session

    env = core_group.core_env(fake, scope=scope, job_id=job)  # type: ignore[arg-type]
    return await core_group.drain(env)


def _groups_of(pg: Any, *members: str) -> int:
    """How many distinct dedup groups of the current core run hold these members."""
    return sql(
        pg,
        'SELECT count(DISTINCT m.group_id) FROM near_dup_members m '
        'JOIN near_dup_runs r ON r.id = m.run_id '
        "WHERE r.level = 'core' AND r.status = 'current' AND m.purpose = 'dedup' "
        'AND m.member_id = ANY(%s)',
        list(members),
    )[0][0]


@pytest.fixture
def ids() -> Any:
    """Image ids no other test of the module shares (the database lives for the module)."""
    tag = uuid.uuid4().hex[:8]
    return lambda name: f'{tag}-{name}'


async def test_two_concurrent_batches_with_a_shared_pair_give_one_group(
    pg, bound_async_engine, opensearch, ids
) -> None:
    a, b = ids('a'), ids('b')
    service = _service(opensearch)
    async with asyncio.TaskGroup() as group:
        group.create_task(_ingest(service, b'bytes-a', a))
        group.create_task(_ingest(service, b'bytes-b', b))

    prefix = a.split('-')[0]
    assert _pending(pg, prefix) == sorted([(a, 'added'), (b, 'added')])
    assert all('near_dup_group_id' not in d for d in opensearch.docs[INDEX].values())

    drained = await _drain(pg, opensearch)

    docs = opensearch.docs[INDEX]
    assert docs[a]['near_dup_group_id'] == docs[b]['near_dup_group_id']
    assert sorted(d['near_dup_is_representative'] for d in docs.values()) == [False, True]
    assert drained.claimed == 2
    assert _pending(pg, prefix) == []
    assert _groups_of(pg, a, b) == 1


C = np.array([0.0, 1.0, 0.0], dtype=np.float32)
B2 = 0.99995 * A + 0.01 * C
VECTORS[b'bytes-b2'] = B2 / np.linalg.norm(B2)
VECTORS[b'bytes-c'] = C


async def test_a_later_batch_joins_the_group_and_an_unrelated_image_stays_alone(
    pg, bound_async_engine, opensearch, ids
) -> None:
    a, b, b2, c = ids('a'), ids('b'), ids('b2'), ids('c')
    service = _service(opensearch)
    await _ingest(service, b'bytes-a', a)
    await _ingest(service, b'bytes-b', b)
    await _drain(pg, opensearch)
    first = opensearch.docs[INDEX][a]['near_dup_group_id']

    await _ingest(service, b'bytes-b2', b2)
    await _ingest(service, b'bytes-c', c)
    second = await _drain(pg, opensearch)

    docs = opensearch.docs[INDEX]
    pair = (a, b, b2)
    assert {docs[i]['near_dup_group_id'] for i in pair} == {first}
    assert 'near_dup_group_id' not in docs[c]
    assert {docs[i]['near_dup_group_size'] for i in pair} == {3}
    assert all(isinstance(docs[i]['near_dup_cosine_to_rep'], float) for i in pair)
    assert sum(docs[i]['near_dup_is_representative'] for i in pair) == 1
    assert second.claimed == 2
    assert _groups_of(pg, *pair) == 1


async def test_a_pending_image_without_a_vector_stays_pending_and_does_not_spin(
    pg, bound_async_engine, opensearch, ids
) -> None:
    ghost = ids('never-indexed')
    await core_pending.record_pending([ghost], 'added')

    drained = await _drain(pg, opensearch)

    assert (drained.runs, drained.missing_vectors) == (1, 1)
    assert _pending(pg, ghost) == [(ghost, 'added')]
    sql(pg, 'DELETE FROM near_dup_pending WHERE member_id = %s', ghost)
