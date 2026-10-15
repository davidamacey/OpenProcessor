"""The ``segment`` action: re-queue the run's items for the region worker, wait for it.

The worker is a separate process. The run stamps each item it re-queues with its id
(``region_run_id``, kept as provenance) and how the worker verifies the new boxes
(``region_run_verify``, cleared by the worker when it writes the item), then sits in
``waiting_region`` until no item of the run is pending. A validated set is never
re-queued (the lock rule), so it is not waited for.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.config import RegionStatus, get_curation_config
from src.config.region_fields import get_region_fields
from src.services.curation.assist.models import SegmentResult
from src.services.curation.region_requeue import apply_requeue_ids
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.curation.file_job import FileJob

POLL_S = 2.0
WAIT_MAX_S = 6 * 3600.0
"""A region worker that never answers must not hold the run forever."""
_PAGE = 500


async def requeue_for_run(
    opensearch: AsyncOpenSearch, ids: list[str], *, run_id: str, verify: str
) -> int:
    """Re-queue ``ids`` for detection with ``run_id`` and ``verify`` stamped on them;
    how many were queued."""
    queued = 0
    for start in range(0, len(ids), _PAGE):
        result = await apply_requeue_ids(
            opensearch,
            ids[start : start + _PAGE],
            target=RegionStatus.PENDING_DETECTION,
            clear_detection=True,
            stamp={'region_run_id': run_id, 'region_run_verify': verify},
        )
        queued += result['updated']
    return queued


async def _count(opensearch: AsyncOpenSearch, run_id: str, statuses: list[str]) -> int:
    F = get_region_fields()
    query: dict[str, Any] = {
        'bool': {
            'filter': [
                {'term': {'region_run_id': run_id}},
                {'terms': {F.status: statuses}},
            ]
        }
    }
    resp = await opensearch.count(index=get_curation_config().items_index, body={'query': query})
    return int(resp.get('count', 0))


async def wait_for_region(
    job: FileJob, opensearch: AsyncOpenSearch, run_id: str, queued: int
) -> SegmentResult:
    """Wait until none of the run's ``queued`` items is pending, or the run is cancelled
    or the wait limit passes (the remainder is reported as ``pending``)."""
    pending_statuses = [
        RegionStatus.PENDING_DETECTION.value,
        RegionStatus.PENDING_VERIFICATION.value,
    ]
    waited = 0.0
    while True:
        pending = await _count(opensearch, run_id, pending_statuses) if queued else 0
        failed = await _count(opensearch, run_id, [RegionStatus.DETECTION_FAILED.value])
        done = max(0, queued - pending)
        job.update(
            progress={'processed': done, 'total': queued},
            updated_at=now_iso(),
        )
        job.touch_heartbeat()
        if pending == 0 or job.cancel_requested() or waited >= WAIT_MAX_S:
            return SegmentResult(queued=queued, done=done - failed, pending=pending, failed=failed)
        await asyncio.sleep(POLL_S)
        waited += POLL_S


__all__ = ['POLL_S', 'WAIT_MAX_S', 'requeue_for_run', 'wait_for_region']
