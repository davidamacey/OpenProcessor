"""The bound project's ``location_policy`` (``project_settings`` column), cached for a few seconds.

Ingest reads it once per image batch and the readers once per request; a change writes through
:func:`set_location_policy` (see ``location_purge`` for what a switch starts), which drops this
process's entry at once. Another process serves the old value for at most the TTL, which is why
the purge waits that long before it scans (``src/services/source_backfill/location_purge.py``).
A project with no settings row has the default (``store``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cachetools import TTLCache

from src.db.repos import settings as settings_repo
from src.db.session import async_session_factory, project_tx
from src.domain.location_policy import DEFAULT_LOCATION_POLICY
from src.services.config_store.bound_project import bound_project_id


if TYPE_CHECKING:
    import uuid

POLICY_CACHE_TTL_S = 5.0
_cache: TTLCache[uuid.UUID, tuple[str, int]] = TTLCache(maxsize=64, ttl=POLICY_CACHE_TTL_S)


async def read_location_policy(*, fresh: bool = False) -> tuple[str, int]:
    """``(location_policy, settings revision)`` of the bound project. ``fresh`` skips the cache."""
    project_id = await bound_project_id()
    if not fresh and (cached := _cache.get(project_id)) is not None:
        return cached
    async with async_session_factory()() as session, project_tx(session, project_id):
        row = await settings_repo.get_settings(session, project_id)
    value = (DEFAULT_LOCATION_POLICY, 0) if row is None else (row.location_policy, row.revision)
    _cache[project_id] = value
    return value


async def get_location_policy() -> str:
    """The policy to write and serve under."""
    return (await read_location_policy())[0]


async def write_location_policy(policy: str, *, expected_revision: int) -> tuple[str, int]:
    """Store ``policy`` when ``expected_revision`` is the settings revision (else
    :class:`~src.db.repos.config.RevisionConflictError`); returns the policy and its new revision."""
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        row = await settings_repo.put_settings(
            session,
            project_id,
            expected_revision=expected_revision,
            changes={'location_policy': policy},
        )
    _cache.pop(project_id, None)
    return row.location_policy, row.revision


__all__ = [
    'POLICY_CACHE_TTL_S',
    'get_location_policy',
    'read_location_policy',
    'write_location_policy',
]
