"""Every jobs dir the global GPU arbiter must watch (projects_plan.md §5.3).

The arbiter is one process-global loop, but training and bake-off jobs are
written into the bound project's own dirs, so its active-run scans need
the union over every active and archived project.
"""

from __future__ import annotations

from pathlib import Path

from src.config import get_gpu_arbiter_config


def all_bakeoff_jobs_dirs() -> list[Path]:
    """``GpuArbiterConfig.bakeoff_jobs_dir`` plus every active/archived
    project's own ``bakeoff_jobs_dir`` (``default`` included), deduplicated.
    Reads the registry's in-process snapshot; no I/O."""
    from src.services.projects.registry import get_project_registry

    dirs: list[Path] = []
    configured = get_gpu_arbiter_config().bakeoff_jobs_dir
    if configured:
        dirs.append(Path(configured))
    for record in get_project_registry().snapshot().values():
        if record.status in ('active', 'archived'):
            candidate = Path(record.resources.bakeoff_jobs_dir)
            if candidate not in dirs:
                dirs.append(candidate)
    return dirs


def _resolve_train_jobs_dir() -> Path:
    """The *default* project's jobs dir (P1R §6.1/D-A: an ordinary
    registered project -- read from the registry, else build fresh)."""
    from src.config.curation import base_curation_config
    from src.config.project_context import bind_project
    from src.config.projects import DEFAULT_SLUG, new_project_record
    from src.services.projects.registry import get_project_registry
    from src.services.training.jobs import _resolve_jobs_dir

    record = get_project_registry().get(DEFAULT_SLUG) or new_project_record(
        DEFAULT_SLUG, base_curation_config()
    )
    with bind_project(record):
        return _resolve_jobs_dir()


def all_train_jobs_dirs() -> dict[str, Path]:
    """``{project_slug: train_jobs_dir}`` for the default dir plus every
    active/archived project (projects_plan.md §5.3) -- the arbiter stays
    one global process but its active-run scan must see every project's
    dir. Reads the registry's in-process snapshot; no I/O here."""
    from src.config.projects import DEFAULT_SLUG
    from src.services.projects.registry import get_project_registry

    dirs: dict[str, Path] = {DEFAULT_SLUG: _resolve_train_jobs_dir()}
    for slug, record in get_project_registry().snapshot().items():
        if slug != DEFAULT_SLUG and record.status in ('active', 'archived'):
            dirs[slug] = record.resources.train_jobs_dir
    return dirs


__all__ = ['all_bakeoff_jobs_dirs', 'all_train_jobs_dirs']
