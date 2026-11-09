"""Clone-source resolution shared by the four revisioned-config clone routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.routers.curation._api_error import api_error, unknown_revision_error


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


async def resolve_clone_source[R](
    *,
    name: str,
    revision: int | None,
    source: str | None,
    what: str,
    get_revision: Callable[[str, int], Awaitable[R | None]],
    build: Callable[..., R | None],
) -> R:
    """The record a clone copies from. ``source='stored'`` plus a revision must
    hit that stored revision (404 otherwise); a revision with no source prefers
    the stored one and falls back to the live record; a named ``source`` must
    match the live record's source."""
    if revision is not None and source in (None, 'stored'):
        stored = await get_revision(name, revision)
        if stored is not None:
            return stored
        if source == 'stored':
            raise unknown_revision_error(name, revision)
    record = build(name, revision=revision)
    if record is None or (source is not None and getattr(record, 'source', None) != source):
        raise api_error(404, 'not_found', f'{name!r} is not a known {what}')
    return record
