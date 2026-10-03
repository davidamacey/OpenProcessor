"""Resume the ingest-time open-vocabulary pass after a restart.

Ingest stamps new images ``open_vocab_status: pending`` and drains them in a
background task of the API process (:mod:`~src.services.curation.open_vocab_ingest`);
a restart, or a segmenter outage that ended the drain, leaves them pending.
:func:`sweep_pending_open_vocab` picks up the ones nobody is working on (a
``pending`` stamp older than :func:`open_vocab_stale_after_s`, or without a stamp) and runs
them through the same drain, so a pending image finishes without an operator.
One sweeper per project runs at a time across API workers (a non-blocking file
lock held for the sweep); a pass over an image is idempotent, so the one race
left (an old drain still running past the grace period) only costs segmenter time.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.curation.dataset_import.limits import (
    open_vocab_stale_after_s,
    open_vocab_sweep_interval_s,
)
from src.services.curation.job_lock import exclusive_start_lock
from src.services.curation.open_vocab_ingest import drain
from src.services.curation.reprocess_open_vocab import current_active_set
from src.services.curation.reprocess_targets import scan_items
from src.services.detection.segmenter_http import segment_image_http


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from opensearchpy import AsyncOpenSearch

    from src.services.curation.ingest import CurationIngestService

    ServiceFactory = Callable[[AsyncOpenSearch], Awaitable[CurationIngestService]]

logger = get_logger(__name__)

#: Images one sweep takes on; the next tick takes the next batch.
SWEEP_BATCH = 500


def _stale(stamp: Any, cutoff: datetime) -> bool:
    if not isinstance(stamp, str):
        return True
    try:
        return datetime.fromisoformat(stamp) < cutoff
    except ValueError:
        return True


async def sweep_pending_open_vocab(
    opensearch: AsyncOpenSearch, service_factory: ServiceFactory, *, now: datetime | None = None
) -> int:
    """Drain the bound project's stale ``pending`` images. Returns how many it
    took on; ``0`` (and ``pending`` is left alone) when no set is active, the
    active set does not opt in with ``run_on_ingest``, or nothing is stale."""
    active = await current_active_set(opensearch)
    if active is None or not active[0].run_on_ingest:
        return 0
    cutoff = (now or datetime.now(UTC)) - timedelta(seconds=open_vocab_stale_after_s())
    pending = await scan_items(
        opensearch,
        {'term': {'open_vocab_status': 'pending'}},
        index=get_curation_config().images_index,
        includes=['open_vocab_status_at'],
        id_field='image_id',
        max_docs=SWEEP_BATCH,
    )
    ids = [i for i, src in pending if _stale(src.get('open_vocab_status_at'), cutoff)][:SWEEP_BATCH]
    if not ids:
        return 0
    ov, revision = active
    await drain(
        opensearch, await service_factory(opensearch), ov, revision, ids, segment_image_http
    )
    return len(ids)


async def default_service(opensearch: AsyncOpenSearch) -> CurationIngestService:
    from src.routers.curation._common import get_class_registry
    from src.routers.curation.ingest import _get_ingest_service

    return await _get_ingest_service(opensearch, get_class_registry())


async def _sweep_all_projects(service_factory: ServiceFactory) -> None:
    from src.services.curation.reprocess_job import jobs_root
    from src.services.projects.bootstrap import for_each_project
    from src.services.projects.guard import make_curation_opensearch

    for slug in for_each_project():
        try:
            with exclusive_start_lock(jobs_root() / 'open_vocab_sweep.lock') as mine:
                if not mine:
                    continue
                taken = await sweep_pending_open_vocab(
                    await make_curation_opensearch(), service_factory
                )
            if taken:
                logger.info('open_vocab_sweep_done', project=slug, images=taken)
        except Exception as exc:
            logger.warning('open_vocab_sweep_failed', project=slug, error=str(exc))


async def _loop(interval_s: int, service_factory: ServiceFactory) -> None:
    while True:
        await asyncio.sleep(interval_s)
        await _sweep_all_projects(service_factory)


def start_open_vocab_sweeper(
    service_factory: ServiceFactory = default_service,
) -> asyncio.Task[None] | None:
    """The periodic sweep as a task the caller cancels at shutdown; ``None``
    when ``OP_OPEN_VOCAB_SWEEP_S`` is 0."""
    interval = open_vocab_sweep_interval_s()
    if interval <= 0:
        return None
    return asyncio.create_task(_loop(interval, service_factory))


async def stop_open_vocab_sweeper(task: asyncio.Task[None] | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


__all__ = [
    'default_service',
    'start_open_vocab_sweeper',
    'stop_open_vocab_sweeper',
    'sweep_pending_open_vocab',
]
