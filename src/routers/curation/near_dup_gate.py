"""Route dependencies: refuse (409 ``near_dup_stale``) only where a holdout is or would be frozen."""

from __future__ import annotations

from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the annotation
from src.services.near_duplicates.holdout import require_freezable, require_fresh_when_frozen


async def require_near_dup_fresh(session: DbSessionDep) -> None:
    """Export gate: stale groups matter only once a holdout is frozen."""
    await require_fresh_when_frozen(session, current_project_id())


async def require_near_dup_freezable(session: DbSessionDep) -> None:
    """Freeze gate: a re-freeze needs fresh groups, a first freeze needs a current item run."""
    await require_freezable(session, current_project_id())
