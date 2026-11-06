"""The project's shared strategy defaults: cached read and optimistic partial-merge write.

The defaults live in the ``defaults`` column of ``project_settings``, whose single ``revision``
guards every settings write (``expected_revision``; a stale one is a
:class:`~src.db.repos.config.RevisionConflictError`, the route's 409). Reads are on nearly every
strategy-scoring request, so they are cached for a few seconds per process; a write invalidates
this process's entry at once, and another process serves the old value for at most the TTL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cachetools import TTLCache

from src.db.repos import settings as settings_repo
from src.db.repos.config import RevisionConflictError
from src.db.session import async_session_factory, project_tx
from src.services.config_store.bound_project import bound_project_id


if TYPE_CHECKING:
    import uuid


_SETTINGS_CACHE_TTL_SECONDS = 5.0
_cache: TTLCache[uuid.UUID, dict[str, Any]] = TTLCache(maxsize=64, ttl=_SETTINGS_CACHE_TTL_SECONDS)


def _envelope(row: settings_repo.SettingsRow | None) -> dict[str, Any]:
    if row is None:
        return {'defaults': {}, 'updated_at': None, 'updated_by': None, 'revision': 0}
    return {
        'defaults': {k: v for k, v in row.defaults.items() if v is not None},
        'updated_at': row.updated_at.isoformat(),
        'updated_by': row.updated_by,
        'revision': row.revision,
    }


async def get_curation_settings(*, fresh: bool = False) -> dict[str, Any]:
    """The bound project's defaults envelope. A project with no settings row (nothing ever
    written) answers the empty envelope at revision ``0``. ``fresh`` skips the process cache
    (the route a client reads its ``revision`` from)."""
    project_id = await bound_project_id()
    if not fresh and (cached := _cache.get(project_id)) is not None:
        return cached
    async with async_session_factory()() as session, project_tx(session, project_id):
        envelope = _envelope(await settings_repo.get_settings(session, project_id))
    _cache[project_id] = envelope
    return envelope


async def update_curation_settings(
    defaults: dict[str, str | None], *, expected_revision: int
) -> dict[str, Any]:
    """Merge ``defaults`` into the stored map (``None`` clears an axis) when ``expected_revision``
    is the stored revision; otherwise :class:`~src.db.repos.config.RevisionConflictError`."""
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        current = await settings_repo.get_settings(session, project_id, for_update=True)
        merged = dict(current.defaults) if current is not None else {}
        for axis, value in defaults.items():
            if value is None:
                merged.pop(axis, None)
            else:
                merged[axis] = value
        row = await settings_repo.put_settings(
            session, project_id, expected_revision=expected_revision, changes={'defaults': merged}
        )
    _cache.pop(project_id, None)
    return _envelope(row)


__all__ = ['RevisionConflictError', 'get_curation_settings', 'update_curation_settings']
