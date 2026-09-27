"""Every project's training and bake-off job dirs, for the GPU arbiter.

Each project (``default`` included) writes job files only to its own
``train_jobs_dir`` / ``bakeoff_jobs_dir`` (``src.config.projects.
resources_for_new``), so "is anything running" means scanning all of
them. Reads the project registry snapshot; ``default`` falls back to a
freshly computed record when the registry hasn't refreshed yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path

    from src.config.projects import ProjectRecord


_SCANNED_STATUSES = frozenset({'active', 'archived'})


def _default_project_record() -> ProjectRecord:
    from src.config.curation import base_curation_config
    from src.config.projects import DEFAULT_SLUG, new_project_record
    from src.services.projects.registry import get_project_registry

    return get_project_registry().get(DEFAULT_SLUG) or new_project_record(
        DEFAULT_SLUG, base_curation_config()
    )


def _scanned_records() -> dict[str, ProjectRecord]:
    from src.config.projects import DEFAULT_SLUG
    from src.services.projects.registry import get_project_registry

    return {
        DEFAULT_SLUG: _default_project_record(),
        **{
            slug: record
            for slug, record in get_project_registry().snapshot().items()
            if slug != DEFAULT_SLUG and record.status in _SCANNED_STATUSES
        },
    }


def all_train_jobs_dirs() -> dict[str, Path]:
    """``{project_slug: train_jobs_dir}`` for ``default`` plus every
    active/archived project."""
    return {slug: r.resources.train_jobs_dir for slug, r in _scanned_records().items()}


def all_bakeoff_jobs_dirs() -> dict[str, Path]:
    """``{project_slug: bakeoff_jobs_dir}`` for ``default`` plus every
    active/archived project -- the dirs the bake-off router enqueues into."""
    return {slug: r.resources.bakeoff_jobs_dir for slug, r in _scanned_records().items()}


def bakeoff_active(*, jobs_dir: Path | None = None) -> bool:
    """True if a bake-off job is queued or running (job.json still present)
    in any project's queue, or in ``jobs_dir`` alone when given. A missing
    dir means nothing queued."""
    targets = [jobs_dir] if jobs_dir is not None else list(all_bakeoff_jobs_dirs().values())
    for target in targets:
        try:
            if any(target.glob('*.job.json')):
                return True
        except OSError:
            continue
    return False


__all__ = ['all_bakeoff_jobs_dirs', 'all_train_jobs_dirs', 'bakeoff_active']
