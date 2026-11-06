"""Which projects have a VLM to call, decided once per worker cycle (#207).

The VLM worker used to find pending crops in every project and only learn
from the route's 409 ``vlm_not_configured`` that the project's VLM is off.
This asks the same facts the route uses (the deployment-wide endpoint
registry plus the project's own activation, ``vlm_configured``) so a project
without an active VLM costs no search and no route call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scripts.curation._project_worker_utils import unpaused_projects
from src.config.project_context import bind_project


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


# While no project has an active VLM the producer sleeps ``--poll-interval``,
# doubling each consecutive such cycle up to this many seconds, so an
# unconfigured deployment costs a state read per project a minute, not a
# search and a route call per project every few seconds.
IDLE_BACKOFF_MAX_S = 60.0


def idle_backoff_s(poll_interval: float, idle_cycles: int) -> float:
    """Sleep after ``idle_cycles`` (>= 1) consecutive cycles with no active
    VLM: ``poll_interval`` doubling per cycle, capped at ``IDLE_BACKOFF_MAX_S``
    (never below ``poll_interval`` itself; 0 cycles means a normal poll)."""
    result: float = min(
        poll_interval * 2 ** max(0, idle_cycles - 1), max(poll_interval, IDLE_BACKOFF_MAX_S)
    )
    return result


async def projects_with_active_vlm(registry: Any, only_slug: str | None) -> list[ProjectRecord]:
    """The unpaused projects (see ``unpaused_projects``) whose VLM is active.

    The registry is refreshed once, then each project's activation is read
    under its own binding. Fail closed: a project whose state cannot be read,
    or whose activation cannot be resolved, is skipped this cycle."""
    from src.services.labeling import vlm_endpoints

    active: list[ProjectRecord] = []
    for record in await unpaused_projects(registry, only_slug):
        try:
            with bind_project(record):
                await vlm_endpoints.refresh_vlm_state()
                if vlm_endpoints.vlm_configured():
                    active.append(record)
        except Exception as exc:
            print(f'[vlm-worker] project={record.slug} VLM state unreadable, skipping: {exc}')
    return active


async def project_has_item_classes(record: ProjectRecord) -> bool:
    """Whether the project's class registry has a class an item can be labeled as.

    A project without one cannot be labeled (the route answers 409 ``no_classes``),
    so the producer must not claim its crops. The registry snapshot is one version check
    when unchanged, so asking once per project per cycle is cheap and a new class is seen
    at once. A project whose registry cannot be read has none to offer: it is skipped."""
    from sqlalchemy.exc import SQLAlchemyError

    from src.db.repos.classes import ClassRegistryError
    from src.services.classes.registry import get_class_registry
    from src.services.curation.region_class import item_classes

    with bind_project(record):
        try:
            registry = await get_class_registry().snapshot()
        except (ClassRegistryError, SQLAlchemyError) as exc:
            print(f'[vlm-worker] project={record.slug} class registry unreadable: {exc}')
            return False
        return bool(item_classes(registry.classes))
