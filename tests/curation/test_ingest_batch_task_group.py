"""The ingest data plane fans out in a ``TaskGroup``: a failure cancels the rest, never leaves a task.

``asyncio.gather`` leaves its siblings running when one raises (Python docs, "Task Groups"). In
``run_ingest_batch`` a byte-identical copy waits on its representative's future; when the
representative's ``ingest_one`` raised, that future was never resolved and the copy's task stayed
pending after the request had already failed. An all-success batch is unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from curation.ingest_fakes import _jpeg_bytes, _make_service
from src.utils.task_group import run_all, run_settled


async def _pending_tasks(owner: str) -> list[asyncio.Task[Any]]:
    """Tasks still running whose coroutine was defined in ``owner`` (the writer's own drain task
    outlives a batch by design)."""
    await asyncio.sleep(0)
    return [
        t
        for t in asyncio.all_tasks()
        if not t.done() and owner in getattr(t.get_coro(), '__qualname__', '')
    ]


@pytest.mark.asyncio
async def test_a_failing_representative_leaves_no_pending_follower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _jpeg_bytes(seed=1)
    svc, _os, _ = _make_service()

    async def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError('ingest_one failed')

    monkeypatch.setattr(svc, 'ingest_one', boom)

    with pytest.raises(RuntimeError, match='ingest_one failed'):
        await svc.ingest_batch([first, first], ['/a.jpg', '/a-copy.jpg'])

    assert await _pending_tasks('run_ingest_batch') == []


@pytest.mark.asyncio
async def test_an_all_success_batch_is_unchanged() -> None:
    first, second = _jpeg_bytes(seed=1), _jpeg_bytes(seed=2)
    svc, os_fake, _ = _make_service()

    batch = await svc.ingest_batch([first, second, first], ['/a.jpg', '/b.jpg', '/a-copy.jpg'])

    assert [r.status for r in batch.results] == ['success', 'success', 'duplicate']
    assert batch.results[2].image_id == batch.results[0].image_id
    assert len(os_fake.images) == 2
    assert await _pending_tasks('run_ingest_batch') == []


@pytest.mark.asyncio
async def test_run_all_keeps_order_and_a_failure_cancels_the_siblings() -> None:
    async def slow(n: int) -> int:
        await asyncio.sleep(0.01 * (3 - n))
        return n

    assert await run_all([slow(n) for n in range(3)]) == [0, 1, 2]

    async def boom() -> None:
        raise ValueError('child')

    async def waits() -> None:
        await asyncio.sleep(10)

    with pytest.raises(ValueError, match='child'):
        await run_all([waits(), boom(), waits()])
    assert await _pending_tasks('waits') == []


@pytest.mark.asyncio
async def test_run_settled_returns_a_failure_as_a_value_and_runs_every_child() -> None:
    async def ok() -> str:
        return 'ok'

    async def bad() -> str:
        raise KeyError('x')

    out = await run_settled([ok(), bad(), ok()])

    assert out[0] == out[2] == 'ok'
    assert isinstance(out[1], KeyError)
