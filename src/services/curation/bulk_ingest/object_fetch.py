"""Fetch a batch of objects with bounded in-flight bytes and classified failures.

A request names a batch of object references; :func:`held_fetch` reserves the
batch's encoded size from a process-wide :class:`~src.utils.cpu_pool.InflightGate`
(the gate class that bounds decoded pixels, here counting encoded MiB) *before* any
download starts, downloads on a dedicated thread pool, and holds the reservation
until the caller has finished with the bytes. One reservation per batch, not per
object, so two batches can never each hold part of the capacity and wait for the
rest. A batch larger than the whole capacity is clamped and runs alone.

An object that is too large, gone, forbidden or unreachable is an outcome of its own
(``error_kind`` per ``docs/design/ingest_surfaces_plan.md``), never a failed batch.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.curation.bulk_ingest.object_store import ObjectRef, ObjectStoreError, S3Config
from src.services.curation.ingest_models import ERROR_KIND_TOO_LARGE
from src.utils.cpu_pool import InflightGate, mib


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.services.curation.bulk_ingest.object_store import ObjectStore


@dataclass(frozen=True)
class FetchOutcome:
    requested: ObjectRef
    actual: ObjectRef | None = None
    data: bytes | None = None
    error: str | None = None
    error_kind: str | None = None


_gates: dict[int, InflightGate] = {}
_pools: dict[int, ThreadPoolExecutor] = {}


def fetch_gate(config: S3Config) -> InflightGate:
    pid = os.getpid()
    if pid not in _gates:
        _gates[pid] = InflightGate(config.inflight_mb)
    return _gates[pid]


def _pool(config: S3Config) -> ThreadPoolExecutor:
    pid = os.getpid()
    if pid not in _pools:
        _pools[pid] = ThreadPoolExecutor(
            max_workers=config.fetch_workers, thread_name_prefix='op-s3-'
        )
    return _pools[pid]


async def _size(store: ObjectStore, ref: ObjectRef, config: S3Config) -> ObjectRef | FetchOutcome:
    if ref.size is not None:
        return ref
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_pool(config), store.head, ref)
    except ObjectStoreError as exc:
        return FetchOutcome(ref, error=str(exc), error_kind=exc.error_kind)


async def _get(store: ObjectStore, ref: ObjectRef, config: S3Config) -> FetchOutcome:
    loop = asyncio.get_running_loop()
    try:
        actual, data = await loop.run_in_executor(_pool(config), store.get, ref)
    except ObjectStoreError as exc:
        return FetchOutcome(ref, error=str(exc), error_kind=exc.error_kind)
    if len(data) > config.max_object_bytes:
        return FetchOutcome(
            ref,
            error=f'{len(data)} bytes exceeds OP_S3_MAX_OBJECT_MB',
            error_kind=ERROR_KIND_TOO_LARGE,
        )
    return FetchOutcome(ref, actual=actual, data=data)


@asynccontextmanager
async def held_fetch(
    store: ObjectStore, refs: list[ObjectRef], config: S3Config
) -> AsyncIterator[list[FetchOutcome]]:
    """Yield one outcome per ref, in order. The batch's bytes stay reserved until the
    ``async with`` block exits."""
    sized = await asyncio.gather(*(_size(store, ref, config) for ref in refs))
    outcomes: list[FetchOutcome | None] = [None] * len(refs)
    to_fetch: list[tuple[int, ObjectRef]] = []
    for i, found in enumerate(sized):
        if isinstance(found, FetchOutcome):
            outcomes[i] = found
        elif (found.size or 0) > config.max_object_bytes:
            outcomes[i] = FetchOutcome(
                refs[i],
                error=f'{found.size} bytes exceeds OP_S3_MAX_OBJECT_MB ({config.max_object_mb})',
                error_kind=ERROR_KIND_TOO_LARGE,
            )
        else:
            to_fetch.append(
                (
                    i,
                    ObjectRef(
                        found.bucket,
                        found.key,
                        refs[i].etag or found.etag,
                        found.version_id or refs[i].version_id,
                        found.size,
                    ),
                )
            )
    units = sum(mib(ref.size) for _, ref in to_fetch)
    if not to_fetch:
        yield [o for o in outcomes if o is not None]
        return
    async with fetch_gate(config).hold(units):
        fetched = await asyncio.gather(*(_get(store, ref, config) for _, ref in to_fetch))
        for (i, _), outcome in zip(to_fetch, fetched, strict=True):
            outcomes[i] = outcome
        yield [o for o in outcomes if o is not None]


__all__ = ['FetchOutcome', 'fetch_gate', 'held_fetch']
