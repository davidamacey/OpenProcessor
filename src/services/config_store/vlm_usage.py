"""Which projects run which VLM endpoint.

The registry is deployment-wide but activation is per project, so "is this endpoint in use" is
answered from every project's LIVE ``vlm`` activation in the database, never a cached snapshot: a
delete must not race a stale view of another project's activation.
"""

from __future__ import annotations

from src.db.repos import config as repo
from src.services.config_store.documents import tx


async def activations_by_project() -> dict[str, repo.ActiveRef | None]:
    """``slug -> vlm activation`` (``None`` = the project never activated one, so it runs the
    ``env`` built-in) for every project that is not being deleted. Raises when the database
    cannot be read: callers use this to REFUSE a delete, so an unreadable project must block it."""
    async with tx('who runs a vlm endpoint') as session:
        return await repo.list_axis_state(session, 'vlm')


def slugs_running(activations: dict[str, repo.ActiveRef | None], name: str) -> list[str]:
    """Projects whose activation names ``name``. ``env`` is also what a
    project that never activated one runs."""
    out: list[str] = []
    for slug, ref in activations.items():
        if ref is None:
            active = 'env' if name == 'env' else None
        else:
            active = ref.name
        if active == name:
            out.append(slug)
    return out


__all__ = ['activations_by_project', 'slugs_running']
