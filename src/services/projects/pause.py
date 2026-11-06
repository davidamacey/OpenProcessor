"""Pause and resume a project's workers: one column, ``projects.paused_at``.

The registry snapshot carries it, so every worker's next discovery cycle sees the change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.db.repos import projects as repo
from src.services.projects.records import record_from_row
from src.services.projects.registry import get_project_registry
from src.services.projects.store import run_repo


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


async def set_paused(slug: str, *, paused: bool) -> ProjectRecord:
    """Idempotent: pausing a paused project keeps the first pause time."""
    row = await run_repo(
        'pause a project' if paused else 'resume a project',
        repo.set_paused,
        slug,
        paused=paused,
        by='api',
    )
    await get_project_registry().ensure_fresh()
    return record_from_row(row)
