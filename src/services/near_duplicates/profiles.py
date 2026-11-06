"""The project's near-duplicate profile: seeded from the code defaults, read fail-closed.

A project is created with revision 1 of the profile ``default`` active, so the cuts equal
today's behaviour (0.98 cosine) until the owner edits them. Reading never falls back to a code
default: a project with no active profile refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.db.repos import config as repo
from src.schemas.curation.near_duplicates import NearDupProfileBody


if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

KIND = 'near_dup_profile'
AXIS = 'near_dup'
DEFAULT_NAME = 'default'


class NoActiveProfileError(LookupError):
    """The project has no active near-duplicate profile; nothing groups until one is activated."""


@dataclass(frozen=True)
class ActiveProfile:
    config_id: uuid.UUID
    revision: int
    body: NearDupProfileBody


async def seed_default_profile(session: AsyncSession, project_id: uuid.UUID) -> None:
    """Create revision 1 of ``default`` from the code defaults and activate it."""
    doc = await repo.save_config(
        session,
        project_id=project_id,
        kind=KIND,
        name=DEFAULT_NAME,
        body=NearDupProfileBody().model_dump(mode='json'),
        expected_revision=None,
        description='The code defaults: 0.98 cosine on the frame and on the crop.',
    )
    await repo.activate(
        session,
        project_id=project_id,
        axis=AXIS,
        config_id=doc.id,
        revision=doc.revision,
        expected_active=None,
    )


async def active_profile(session: AsyncSession, project_id: uuid.UUID) -> ActiveProfile:
    ref = await repo.get_activation(session, project_id, AXIS)
    if ref is None or ref.state != 'active' or ref.config_id is None or ref.revision is None:
        msg = 'no near-duplicate profile is active for this project'
        raise NoActiveProfileError(msg)
    doc = await repo.get_config_by_id(session, ref.config_id, ref.revision)
    if doc is None:
        msg = f'the active near-duplicate profile {ref.name!r} revision {ref.revision} is gone'
        raise NoActiveProfileError(msg)
    return ActiveProfile(doc.id, doc.revision, NearDupProfileBody.model_validate(doc.body))
