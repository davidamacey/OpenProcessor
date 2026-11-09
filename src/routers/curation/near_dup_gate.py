"""Route dependency: refuse while the near-duplicate groups are stale (409 ``near_dup_stale``)."""

from __future__ import annotations

from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the annotation
from src.services.near_duplicates.holdout import require_fresh


async def require_near_dup_fresh(session: DbSessionDep) -> None:
    await require_fresh(session, current_project_id())
