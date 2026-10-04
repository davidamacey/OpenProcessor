"""One-time-per-start repair of projects created by an earlier release.

The API runs many uvicorn workers and each runs its startup bootstrap, so a
repair that rewrites index data must not run in all of them at once (that
flooded a small-heap node with ``update_by_query`` tasks, most rejected 429).
The per-project lock is a non-blocking ``flock``: the worker that gets it
repairs the project, the rest skip it. Every step is idempotent and cheap
when there is nothing to do, so a worker that starts after the winner
finished simply finds nothing left.

Repairs, per project:

* load the config snapshot, then register the active region profile's class
  (the process-local snapshot is empty at bootstrap, so the profile read by
  the plain startup seed was always "none");
* record ``embedded`` on items that have a vector but no ``embedding_state``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.core.logging import get_logger
from src.services.curation.job_lock import exclusive_start_lock


logger = get_logger(__name__)

_RETRY_ATTEMPTS = 4
_RETRY_BASE_DELAY_S = 2.0
_TOO_MANY_REQUESTS = 429


async def _backfill_with_retry(client: Any, index: str) -> int:
    """The embedded-state backfill, retried with backoff while OpenSearch
    answers 429 (the node is busy, not the request wrong)."""
    from src.services.curation.embedding_state import backfill_embedded_state

    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return await backfill_embedded_state(client, index)
        except Exception as exc:
            if getattr(exc, 'status_code', None) != _TOO_MANY_REQUESTS or (
                attempt == _RETRY_ATTEMPTS - 1
            ):
                raise
            await asyncio.sleep(_RETRY_BASE_DELAY_S * 2**attempt)
    return 0  # unreachable: the last attempt returns or raises


async def _ensure_active_region_class(client: Any) -> int | None:
    from src.services.config_store import get_config_store
    from src.services.curation.region_class import ensure_region_class

    await get_config_store().refresh(client)
    return ensure_region_class()


async def repair_bound_project(client: Any, slug: str, lock_file: Any, items_index: str) -> None:
    """Repair the bound project unless another process already holds its
    lock. Logs a summary at info only when rows were repaired (warning on a
    failed step, debug otherwise); a failed step is logged and left for the
    next start."""
    with exclusive_start_lock(lock_file) as acquired:
        if not acquired:
            return
        summary: dict[str, Any] = {}
        try:
            summary['region_class_id'] = await _ensure_active_region_class(client)
        except Exception as exc:
            summary['region_class_error'] = str(exc)
        try:
            summary['embedded_state_backfilled'] = await _backfill_with_retry(client, items_index)
        except Exception as exc:
            summary['embedded_state_error'] = str(exc)
        if any(k.endswith('_error') for k in summary):
            level = logger.warning
        elif summary.get('embedded_state_backfilled'):
            level = logger.info
        else:
            # Nothing to repair: a worker that starts after the lock is
            # released re-checks and must not repeat the line per project.
            level = logger.debug
        level('legacy_project_repair', project=slug, **summary)


async def repair_legacy_projects(client: Any) -> None:
    from src.config import get_curation_config
    from src.config.project_context import current_project
    from src.services.projects.bootstrap import for_each_project

    for slug in for_each_project():
        state_dir = current_project().record.resources.project_state_dir
        await repair_bound_project(
            client, slug, state_dir / 'legacy_repair.lock', get_curation_config().items_index
        )
