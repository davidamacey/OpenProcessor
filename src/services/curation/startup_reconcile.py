"""Startup repair of file-backed jobs left active by a killed process.

Each job module's ``reconcile_orphaned_jobs()`` turns a stale ``running``
state into ``interrupted`` for the bound project (see
:mod:`~src.services.curation.job_reconcile`); this runs every module for
every project, best-effort and isolated per module.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from src.clients.job_lease import release, takeover_expired
from src.core.logging import get_logger
from src.domain.job_lease import Acquired, LeaseKind


logger = get_logger(__name__)

JOB_MODULES: tuple[tuple[str, LeaseKind], ...] = (
    ('src.services.curation.item_scores.job', LeaseKind.ITEM_SCORES),
    ('src.services.curation.selection.job', LeaseKind.SELECTION),
    ('src.services.curation.embedding_viz', LeaseKind.EMBEDDING_VIZ),
    ('src.services.curation.autolabel.job', LeaseKind.AUTOLABEL),
    ('src.services.curation.probe_job', LeaseKind.PROBE),
    ('src.services.curation.reprocess_job', LeaseKind.REPROCESS),
    ('src.services.curation.assist.job', LeaseKind.ASSIST_RUN),
    ('src.services.curation.dataset_import.runner', LeaseKind.DATASET_IMPORT),
    ('src.services.training.promote_job', LeaseKind.PROMOTE),
)


async def reconcile_all_jobs(opensearch: Any) -> None:
    """Repair every module's orphaned jobs, for every project.

    Each repair is a lease takeover attempt: a job whose lease is live (a sibling process
    that just started it, with no heartbeat file yet) is left alone, and when 32 workers
    start at once one wins each lease and the rest see it busy and skip. A lease that is
    absent (a job from before leases) or expired belongs to a dead holder, so the module's
    own repair runs under the lease taken here.
    """
    from src.services.projects.bootstrap import for_each_project

    for name, kind in JOB_MODULES:
        module = importlib.import_module(name)
        for slug in for_each_project():
            try:
                if await _reconcile_under_lease(opensearch, module, kind):
                    logger.warning('orphaned_job_reconciled', module=name, project=slug)
            except Exception as exc:
                logger.warning(
                    'orphaned_job_reconcile_skipped', module=name, project=slug, error=str(exc)
                )


async def _reconcile_under_lease(opensearch: Any, module: Any, kind: LeaseKind) -> bool:
    taken = await takeover_expired(opensearch, kind, None, f'reconcile-{os.getpid()}')
    if not isinstance(taken, Acquired):
        return False
    try:
        return bool(module.reconcile_orphaned_jobs())
    finally:
        await release(opensearch, taken.lease)
