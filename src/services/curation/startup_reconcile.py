"""Startup repair of file-backed jobs left active by a killed process.

Each job module's ``reconcile_orphaned_jobs()`` turns a stale ``running``
state into ``interrupted`` for the bound project (see
:mod:`~src.services.curation.job_reconcile`); this runs every module for
every project, best-effort and isolated per module.
"""

from __future__ import annotations

import importlib

from src.core.logging import get_logger


logger = get_logger(__name__)

JOB_MODULES = (
    'src.services.curation.item_scores.job',
    'src.services.curation.selection.job',
    'src.services.curation.embedding_viz',
    'src.services.curation.autolabel.job',
    'src.services.curation.probe_job',
    'src.services.curation.reprocess_job',
    'src.services.curation.dataset_import.runner',
    'src.services.training.promote_job',
)


def reconcile_all_jobs() -> None:
    from src.services.projects.bootstrap import for_each_project
    from src.services.projects.combine.store import reconcile_orphaned_jobs

    for name in JOB_MODULES:
        module = importlib.import_module(name)
        for slug in for_each_project():
            try:
                if module.reconcile_orphaned_jobs():
                    logger.warning('orphaned_job_reconciled', module=name, project=slug)
            except Exception as exc:
                logger.warning(
                    'orphaned_job_reconcile_skipped', module=name, project=slug, error=str(exc)
                )
    # Combine jobs are global (not nested under one project), so once.
    try:
        if reconcile_orphaned_jobs():
            logger.warning('orphaned_combine_jobs_reconciled')
    except Exception as exc:
        logger.warning('orphaned_combine_reconcile_skipped', error=str(exc))
