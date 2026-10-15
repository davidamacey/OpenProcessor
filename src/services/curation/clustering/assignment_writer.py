"""Concurrent, bounded write-back of cluster assignments.

Every assignment is the same guarded painless update as before
(:func:`~cluster_write_guard._guarded_class_cluster_write`), so a document a human
owns is still left alone and the stored result of each document is identical. What
changes is how the batches travel: up to ``concurrency`` bulk requests are in flight
at once instead of one after another, and a bulk answered with HTTP 429 (or items
rejected with 429) is retried with a growing delay instead of being dropped.

A document id appears in exactly one batch of a pass, so the order the batches finish
in does not change what is stored.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import TransportError

from src.config.curation_accessors import items_index
from src.core.logging import get_logger
from src.services.curation.clustering.cluster_write_guard import (
    _guarded_class_cluster_write,
    _log_bulk_write_errors,
)


if TYPE_CHECKING:
    from collections.abc import Iterable

    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


# At most this many documents in one bulk request. A single 100k-document request exceeds the
# client's default timeout, and a request must be small enough that several run at once.
BULK_CHUNK = 1000
# Fewest documents in a request when a small pool is split to keep every writer busy.
BULK_CHUNK_MIN = 250
# Bulk requests in flight at once. The node's write threads, not the client, are the limit:
# on one node, 4 writers reach about 3x and 8 about 4x of one writer, and more add little.
BULK_CONCURRENCY = 8
# 429 handling: attempts after the first, and the first delay (doubles per attempt).
BULK_RETRIES = 6
BULK_BACKOFF_S = 0.5

# (document id, cluster id to store, distance or None)
Assignment = tuple[str, int, float | None]
BodyFn = Callable[[Sequence[Any]], list[dict[str, Any]]]


def _body(batch: Sequence[Assignment]) -> list[dict[str, Any]]:
    index = items_index()
    body: list[dict[str, Any]] = []
    for crop_id, cid, dist in batch:
        body.append({'update': {'_index': index, '_id': crop_id}})
        body.append(_guarded_class_cluster_write(cid, dist))
    return body


def _status(item: dict[str, Any]) -> int | None:
    action: dict[str, Any] = next(iter(item.values()), {})
    return action.get('status')


def bulk_item_failed(item: dict[str, Any]) -> bool:
    """True when a bulk response item carries an ``error`` for its action."""
    action: dict[str, Any] = next(iter(item.values()), {}) or {}
    return bool(action.get('error'))


async def send_batch(
    client: AsyncOpenSearch,
    op: str,
    batch: Sequence[Any],
    body_fn: BodyFn = _body,
) -> int:
    """Bulk-write one batch (``body_fn`` turns items into actions), retrying what OpenSearch
    rejects with 429; returns the number of items that failed.

    Any other item failure is logged and the pass proceeds (as it always has); any other
    request-level error propagates.
    """
    pending = list(batch)
    for attempt in range(BULK_RETRIES + 1):
        last = attempt == BULK_RETRIES
        try:
            resp = await client.bulk(body=body_fn(pending), refresh=False)
        except TransportError as exc:
            if exc.status_code != 429 or last:
                raise
            await asyncio.sleep(BULK_BACKOFF_S * 2**attempt)
            continue
        if not resp.get('errors'):
            return 0
        rejected: list[Any] = []
        other: list[dict[str, Any]] = []
        for assignment, item in zip(pending, resp.get('items') or [], strict=False):
            if _status(item) == 429 and not last:
                rejected.append(assignment)
            else:
                other.append(item)
        _log_bulk_write_errors(op, {'items': other})
        failed = sum(1 for item in other if bulk_item_failed(item))
        if not rejected:
            return failed
        pending = rejected
        await asyncio.sleep(BULK_BACKOFF_S * 2**attempt)
    return 0


class AssignmentWriter:
    """``async with`` a writer, ``await submit(batch)`` per batch, leave the block to drain.

    ``submit`` waits while ``concurrency`` batches are already in flight, and raises the
    first failure of an earlier batch. Leaving the block waits for the rest; leaving it on an
    error (including a cancel raised by the caller) cancels what is still in flight.
    """

    def __init__(
        self,
        client: AsyncOpenSearch,
        op: str,
        *,
        concurrency: int = BULK_CONCURRENCY,
        body_fn: BodyFn = _body,
    ) -> None:
        self._client = client
        self._op = op
        self._body_fn = body_fn
        self.n_failed = 0
        self._slots = asyncio.Semaphore(max(1, concurrency))
        self._tasks: set[asyncio.Task[None]] = set()
        self._error: BaseException | None = None
        self.n_written = 0

    async def __aenter__(self) -> AssignmentWriter:
        return self

    async def __aexit__(self, exc_type: object, *_: object) -> None:
        if exc_type is not None:
            await self._cancel()
            return
        await self._drain()

    async def submit(self, batch: Sequence[Any]) -> None:
        await self._slots.acquire()
        if self._error is not None:
            self._slots.release()
            raise self._error
        task = asyncio.create_task(self._send(batch))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _send(self, batch: Sequence[Any]) -> None:
        try:
            self.n_failed += await send_batch(self._client, self._op, batch, self._body_fn)
            self.n_written += len(batch)
        except Exception as exc:
            self._error = self._error or exc
        finally:
            self._slots.release()

    async def _drain(self) -> None:
        tasks = list(self._tasks)
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            await self._cancel()
            raise
        if self._error is not None:
            raise self._error

    async def _cancel(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def batch_size(n_rows: int, concurrency: int = BULK_CONCURRENCY) -> int:
    """Documents per request: ``BULK_CHUNK``, or less for a pool too small to keep
    ``concurrency`` requests busy (never below ``BULK_CHUNK_MIN``)."""
    return max(BULK_CHUNK_MIN, min(BULK_CHUNK, -(-n_rows // max(1, concurrency))))


def batches(rows: Iterable[Any], size: int = BULK_CHUNK) -> Iterable[list[Any]]:
    """``rows`` in order, ``size`` at a time."""
    batch: list[Any] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
