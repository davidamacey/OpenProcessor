"""Config store repository, part 1: errors, row types, the digest, scope check and epochs.

Everything here is shared by :mod:`src.db.repos.config` (documents, probes, deployment settings, the
upgrade load) and :mod:`src.db.repos.config_activations`; callers import the names from
``src.db.repos.config``, which re-exports them.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.config import ACTIVATION_AXES, CONFIG_KINDS, DEPLOYMENT_CONFIG_KINDS
from src.db.models.projects import DeploymentCounter, Project


if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

GLOBAL_EPOCH_COUNTER = 'global_config_epoch'
OFF_CONFIG_ID = uuid.UUID(int=0)
AXIS_KIND: dict[str, str] = dict(ACTIVATION_AXES)
DEFAULT_ACTOR = 'system'


class RevisionConflictError(Exception):
    """A save, delete or settings write named a revision that is no longer current."""

    def __init__(self, current_revision: int | None) -> None:
        self.current_revision = current_revision
        super().__init__(f'revision conflict (current_revision={current_revision})')


class ActiveConflictError(Exception):
    """An activation named an ``expected_active`` that is no longer the active one."""

    def __init__(self, current: dict[str, Any] | None) -> None:
        self.current = current
        super().__init__(f'active conflict (current={current})')


class AxisKindError(ValueError):
    """The document's kind does not belong on that activation axis."""


class ConfigInUseError(Exception):
    """A document that an activation still points at cannot be deleted."""


class ConfigScopeError(ValueError):
    """The kind and the project do not agree (a deployment kind with a project, or the reverse)."""


@dataclass(frozen=True)
class ConfigDoc:
    """A document and the body of one of its revisions (the current one unless asked for another)."""

    id: uuid.UUID
    project_id: uuid.UUID | None
    kind: str
    name: str
    revision: int
    body: dict[str, Any]
    description: str
    created_at: dt.datetime
    updated_at: dt.datetime
    updated_by: str
    cloned_from_id: uuid.UUID | None
    current_revision: int


@dataclass(frozen=True)
class RevisionInfo:
    revision: int
    created_at: dt.datetime
    created_by: str


@dataclass(frozen=True)
class ActiveRef:
    """What an axis points at. ``state`` is ``active`` or ``off``; a never-activated axis has no ref."""

    axis: str
    state: str
    config_id: uuid.UUID | None = None
    kind: str | None = None
    name: str | None = None
    revision: int | None = None
    activated_at: dt.datetime | None = None
    activated_by: str | None = None

    def as_expected(self) -> dict[str, Any]:
        """The ``{name, revision}`` shape of the wire's ``expected_active``."""
        return {'name': self.name, 'revision': self.revision}


@dataclass(frozen=True)
class ActivationEvent:
    id: int
    axis: str
    from_config_id: uuid.UUID | None
    from_revision: int | None
    to_config_id: uuid.UUID
    to_revision: int
    reason: str
    at: dt.datetime
    by: str


@dataclass(frozen=True)
class Probe:
    config_id: uuid.UUID
    revision: int
    probed_at: dt.datetime
    ok: bool
    result: dict[str, Any]


@dataclass(frozen=True)
class ProbeLoad:
    revision: int
    probed_at: dt.datetime
    ok: bool
    result: dict[str, Any]


@dataclass(frozen=True)
class RuntimeReport:
    project_id: uuid.UUID
    process: str
    host: str
    config_epoch: int
    applied_at: dt.datetime


@dataclass(frozen=True)
class ConfigRevisionLoad:
    revision: int
    body: dict[str, Any]
    created_at: dt.datetime
    created_by: str = 'upgrade'


@dataclass(frozen=True)
class ConfigDocumentLoad:
    """A 0.6.0 document and every revision it had; the last revision is the current one."""

    project_id: uuid.UUID | None
    kind: str
    name: str
    revisions: Sequence[ConfigRevisionLoad]
    description: str = ''
    id: uuid.UUID | None = None
    cloned_from_id: uuid.UUID | None = None
    created_at: dt.datetime | None = None
    updated_at: dt.datetime | None = None
    updated_by: str = 'upgrade'


@dataclass(frozen=True)
class ActivationEventLoad:
    axis: str
    to_config_id: uuid.UUID
    to_revision: int
    at: dt.datetime
    from_config_id: uuid.UUID | None = None
    from_revision: int | None = None
    reason: str = 'upgrade'
    by: str = 'upgrade'


@dataclass(frozen=True)
class VlmEndpointLoad:
    """One deployment-wide endpoint, its revisions, its probes and (optionally) its last probe."""

    document: ConfigDocumentLoad
    probes: Sequence[ProbeLoad] = field(default_factory=tuple)


def body_sha256(body: Any) -> str:
    """The digest stored with every revision: sha256 of the canonical JSON (sorted keys, no spaces)."""
    canonical = json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def check_scope(kind: str, project_id: uuid.UUID | None) -> None:
    if kind not in CONFIG_KINDS:
        msg = f'unknown config kind {kind!r}'
        raise ConfigScopeError(msg)
    if (kind in DEPLOYMENT_CONFIG_KINDS) != (project_id is None):
        msg = f'{kind} is {"deployment-wide" if kind in DEPLOYMENT_CONFIG_KINDS else "per project"}'
        raise ConfigScopeError(msg)


async def bump_epoch(session: AsyncSession, project_id: uuid.UUID | None) -> int:
    """Advance the counter every process polls: the project's ``config_epoch`` or the global one."""
    if project_id is None:
        counter = (
            pg_insert(DeploymentCounter)
            .values(name=GLOBAL_EPOCH_COUNTER, value=1)
            .on_conflict_do_update(
                index_elements=[DeploymentCounter.name],
                set_={'value': DeploymentCounter.value + 1},
            )
            .returning(DeploymentCounter.value)
        )
        return int((await session.execute(counter)).scalar_one())
    row = await session.execute(
        update(Project)
        .where(Project.id == project_id)
        .values(config_epoch=Project.config_epoch + 1)
        .returning(Project.config_epoch)
    )
    epoch = row.scalar_one_or_none()
    if epoch is None:
        msg = f'no project {project_id}'
        raise LookupError(msg)
    return int(epoch)


async def get_epoch(session: AsyncSession, project_id: uuid.UUID | None) -> int:
    """The current epoch; ``0`` while the global counter has never been bumped."""
    if project_id is None:
        value = (
            await session.execute(
                select(DeploymentCounter.value).where(
                    DeploymentCounter.name == GLOBAL_EPOCH_COUNTER
                )
            )
        ).scalar_one_or_none()
        return int(value or 0)
    value = (
        await session.execute(select(Project.config_epoch).where(Project.id == project_id))
    ).scalar_one_or_none()
    if value is None:
        msg = f'no project {project_id}'
        raise LookupError(msg)
    return int(value)
