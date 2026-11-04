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
* pin a project with no VLM activation doc to what it used to inherit
  (:mod:`~src.services.projects.vlm_default`);
* record ``embedded`` on items that have a vector but no ``embedding_state``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.core.logging import get_logger
from src.domain.job_lease import LeaseKind
from src.services.curation.lease_heartbeat import leased
from src.services.projects.vlm_default import pin_legacy_vlm


logger = get_logger(__name__)

_RETRY_ATTEMPTS = 4
_RETRY_BASE_DELAY_S = 2.0
_TOO_MANY_REQUESTS = 429


async def _backfill_with_retry(client: Any, index: str) -> int:
    """The embedded-state backfill, retried with backoff while OpenSearch
    answers 429 (the node is busy, not the request wrong)."""
    from src.domain.embedding_state import backfill_embedded_state

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


class _HeldElsewhereError(Exception):
    """Another process holds the project's repair lease."""


async def repair_bound_project(client: Any, slug: str, items_index: str) -> None:
    """Repair the bound project unless another process already holds its ``legacy_repair``
    lease (32 workers start at once: one wins, the rest skip). Logs a summary at info only
    when rows were repaired (warning on a failed step, debug otherwise); a failed step is
    logged and left for the next start."""
    try:
        async with leased(
            client,
            LeaseKind.LEGACY_REPAIR,
            None,
            f'repair-{slug}',
            busy=lambda _holder: _HeldElsewhereError(),
        ):
            await _repair_steps(client, slug, items_index)
    except _HeldElsewhereError:
        return


async def _repair_steps(client: Any, slug: str, items_index: str) -> None:
    summary: dict[str, Any] = {}
    steps = (
        ('region_class_id', 'region_class_error', lambda: _ensure_active_region_class(client)),
        ('vlm_pinned', 'vlm_pin_error', lambda: pin_legacy_vlm(client)),
        (
            'embedded_state_backfilled',
            'embedded_state_error',
            lambda: _backfill_with_retry(client, items_index),
        ),
    )
    for key, error_key, step in steps:
        try:
            summary[key] = await step()
        except Exception as exc:
            summary[error_key] = str(exc)
    if any(k.endswith('_error') for k in summary):
        level = logger.warning
    elif summary.get('embedded_state_backfilled') or summary.get('vlm_pinned'):
        level = logger.info
    else:
        # Nothing to repair: a worker that starts after the lease is released re-checks and
        # must not repeat the line per project.
        level = logger.debug
    level('legacy_project_repair', project=slug, **summary)


async def repair_legacy_projects(client: Any) -> None:
    from src.config import get_curation_config
    from src.services.projects.bootstrap import for_each_project

    for slug in for_each_project():
        await repair_bound_project(client, slug, get_curation_config().items_index)
