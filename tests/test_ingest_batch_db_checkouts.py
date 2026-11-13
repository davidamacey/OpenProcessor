"""One ingest batch holds a small bounded number of Postgres connections at once.

The API pool has 16 connections per process and no overflow. Every image of a batch used to open
its own sessions (placement, content claim, near-duplicate pending rows) 16 at a time, so a few
concurrent batches exhausted the pool and the route answered 503 ``database_busy``.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest

from src.db import checkout_budget as budget
from src.services.curation import ingest_batch, ingest_index
from src.services.curation.ingest_models import IngestResult
from src.services.near_duplicates import content_key, pending


# The offline conftest swaps these for in-memory fakes per test; keep the real ones.
_REAL_CLAIM = content_key.claim_bound
_REAL_PLACE = ingest_index.place_vectors


class _Pool:
    """Counts sessions open at once, as a pool's checked-out connections."""

    def __init__(self) -> None:
        self.open = 0
        self.peak = 0

    def factory(self) -> Any:
        pool = self

        @asynccontextmanager
        async def session() -> Any:
            pool.open += 1
            pool.peak = max(pool.peak, pool.open)
            try:
                await asyncio.sleep(0.005)  # a statement in flight
                yield object()
            finally:
                pool.open -= 1

        return lambda: session()


@pytest.fixture
def pool(monkeypatch: pytest.MonkeyPatch) -> _Pool:
    pool = _Pool()
    project = uuid.uuid4()

    async def bound() -> uuid.UUID:
        return project

    @asynccontextmanager
    async def tx(_session: Any, _project: Any) -> Any:
        yield _session

    import src.db.session as db_session
    from src.services.config_store import bound_project

    monkeypatch.setattr(db_session, 'async_session_factory', pool.factory)
    monkeypatch.setattr(db_session, 'project_tx', lambda s, p: tx(s, p))
    monkeypatch.setattr(bound_project, 'bound_project_id', bound)
    for module in (content_key, pending):
        monkeypatch.setattr(module, 'async_session_factory', pool.factory)
        monkeypatch.setattr(module, 'project_tx', lambda s, p: tx(s, p))
        monkeypatch.setattr(module, 'bound_project_id', bound)

    async def decide(*_a: Any, **_k: Any) -> content_key.Verdict:
        return content_key.Verdict(None, 'k', None)

    async def enqueue(*_a: Any, **_k: Any) -> int:
        return 1

    async def place(*_a: Any, **_k: Any) -> list[Any]:
        return []

    async def insert(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(content_key, 'claim_bound', _REAL_CLAIM)
    monkeypatch.setattr(ingest_index, 'place_vectors', _REAL_PLACE)
    monkeypatch.setattr(content_key, '_decide', decide)
    monkeypatch.setattr(content_key.repo, 'insert_content', insert)
    monkeypatch.setattr(content_key, 'content_image_id', lambda _k: 'img')
    monkeypatch.setattr(pending, 'enqueue_pending', enqueue)
    from src.services.clustering import assign

    monkeypatch.setattr(assign, 'place_new', place)
    return pool


async def _write_phases() -> None:
    """The three per-image database phases of an ingest, in their real order."""
    await ingest_index.place_vectors(['i'], __import__('numpy').zeros((1, 2), dtype='float32'))
    await content_key.claim_bound('k', 'img', '/p.jpg', b'x')
    await pending.enqueue_pending_bound('item', ['c'], 'added')


@pytest.mark.asyncio
async def test_sixteen_images_share_the_batch_budget(pool: _Pool) -> None:
    with budget.checkout_budget(ingest_batch.BATCH_DB_CONNECTIONS):
        await asyncio.gather(*[_write_phases() for _ in range(16)])
    assert 1 <= pool.peak <= ingest_batch.BATCH_DB_CONNECTIONS
    assert pool.open == 0


@pytest.mark.asyncio
async def test_run_ingest_batch_runs_its_images_under_the_budget(
    pool: _Pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from src.services.curation import ingest_gpu_decode

    async def not_ready(_s: Any) -> bool:
        return False

    monkeypatch.setattr(ingest_gpu_decode, 'gpu_decode_ready', not_ready)

    class Service:
        detector = SimpleNamespace(secondary_profile=None)
        config = SimpleNamespace(ingest_gpu_decode=False, ingest_gpu_min_megapixels=0.0)
        policy = SimpleNamespace(embedding=SimpleNamespace(mode='all'))

        async def _check_duplicates(self, entries: Any) -> list[None]:
            return [None] * len(entries)

        async def ingest_one(self, _data: bytes, path: str, **_kw: Any) -> IngestResult:
            await _write_phases()
            return IngestResult(status='success', image_id=path, image_path=path)

        async def refresh_indexes(self) -> None:
            return None

    async def no_prefill(*_a: Any, **_k: Any) -> tuple[dict, dict, dict, dict]:
        return {}, {}, {}, {}

    monkeypatch.setattr(ingest_batch, '_prefill_detections', no_prefill)
    images = [bytes([i]) * 64 for i in range(16)]
    result = await ingest_batch.run_ingest_batch(
        cast('Any', Service()),
        images,
        [f'/{i}.jpg' for i in range(16)],
    )
    assert all(r.status == 'success' for r in result.results)
    assert 1 <= pool.peak <= ingest_batch.BATCH_DB_CONNECTIONS
