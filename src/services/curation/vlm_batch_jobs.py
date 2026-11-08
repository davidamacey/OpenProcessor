"""Queue a VLM batch job for the bound project (the route half; the work is ``vlm_batch``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.project_context import current_project_id
from src.services.jobs.service import start_job


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession


async def start(session: AsyncSession, kind: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Queue ``kind``; ``{'job_id', 'status'}`` for the 202 body."""
    result = await start_job(session, kind, current_project_id(), params=params)
    return {'job_id': str(result.job.id), 'status': result.job.status}
