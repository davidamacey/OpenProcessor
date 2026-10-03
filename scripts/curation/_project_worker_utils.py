"""Per-project discovery helpers shared by the multi-project curation
workers (``vlm_worker.py``, ``cluster_refresh_daemon.py`` and the
detection worker in ``scripts/curation/worker/``).

A worker process is not bound to one project: each cycle it discovers
the active projects, skips paused ones, and binds each project only
around that project's own work (projects_plan.md §5.1/§5.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.services.projects.script_binding import only_project


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord

PIPELINE_PAUSED_FLAG_NAME = 'pipeline_paused.flag'
REGION_STAGE_PAUSED_FLAG_NAME = 'region_stage_paused.flag'


def is_project_paused(record: ProjectRecord) -> bool:
    """A project's pipeline is paused while
    ``<project_state_dir>/pipeline_paused.flag`` exists. Workers skip a
    paused project's fetches and keep serving the others; the global GPU
    pause sentinel still pauses everything."""
    return (Path(record.resources.project_state_dir) / PIPELINE_PAUSED_FLAG_NAME).exists()


def is_region_stage_paused(record: ProjectRecord) -> bool:
    """The project's region stage alone is paused while
    ``<project_state_dir>/region_stage_paused.flag`` exists: the region worker
    fetches nothing for it and releases items it already holds before their
    segmenter call, so they stay ``pending_detection``. Every other stage
    keeps running."""
    return (Path(record.resources.project_state_dir) / REGION_STAGE_PAUSED_FLAG_NAME).exists()


async def unpaused_projects(registry: Any, only_slug: str | None) -> list[ProjectRecord]:
    """The active, unpaused projects this worker serves this cycle
    (just ``only_slug`` when the worker runs with ``--project``)."""
    await registry.ensure_fresh()
    return [
        record
        for record in only_project(registry.active_projects(), only_slug)
        if not is_project_paused(record)
    ]


def rotated(records: list[ProjectRecord], start: int) -> list[ProjectRecord]:
    """``records`` starting at ``start`` (mod len): the round-robin order
    for one cycle; callers advance ``start`` by one per cycle."""
    if not records:
        return []
    start %= len(records)
    return records[start:] + records[:start]


def curation_api_prefix() -> str:
    """The deployment's curation API mount (a global setting)."""
    from src.config.curation import base_curation_config

    return base_curation_config().api_prefix.rstrip('/')


def scoped_url(api: str, api_prefix: str, slug: str, path: str) -> str:
    """``{api}{api_prefix}/projects/{slug}{path}``: the project-scoped
    route a script calls (it has no request context for
    ``project_api_base()``)."""
    return f'{api}{api_prefix}/projects/{slug}{path}'
