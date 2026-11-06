"""The database id of the project a request or worker is bound to.

Every config, settings and state row is keyed by the project's id, while the rest of the code
holds the slug. A slug maps to one id for the life of the row, so a positive answer is cached
for the process; an unknown slug raises (fail closed) and is never cached, so a project created
by another process is found on the next call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from src.config.project_context import current_project
from src.db.models.projects import Project
from src.db.session import async_session_factory, deployment_tx


if TYPE_CHECKING:
    import uuid


_IDS: dict[str, uuid.UUID] = {}


class UnknownProjectError(LookupError):
    """No project row carries this slug."""


async def project_id_for(slug: str) -> uuid.UUID:
    cached = _IDS.get(slug)
    if cached is not None:
        return cached
    async with async_session_factory()() as session, deployment_tx(session, 'resolve a slug'):
        found = (
            await session.execute(select(Project.id).where(Project.slug == slug))
        ).scalar_one_or_none()
    if found is None:
        msg = f'no project {slug!r}'
        raise UnknownProjectError(msg)
    _IDS[slug] = found
    return found


async def bound_project_id() -> uuid.UUID:
    """The id of the bound project; ``ProjectNotBound`` when nothing is bound."""
    return await project_id_for(current_project().record.slug)
