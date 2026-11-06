"""Projects, slugs, index ownership and pause.

Every function takes the caller's session inside the caller's ``deployment_tx`` and never commits.
A write that changes what the registry snapshot shows bumps ``deployment_counters('projects_revision')``
in the same transaction, so a second process sees the change on its next poll
(``docs/design/v070_rebuild/02a_postgres_schema.md`` section 2).

The two ``*Load`` entry points are the upgrade contract: ``load_project`` writes the rows the
normal create path writes, with the given ids and timestamps, except the default embedding spaces
and without touching OpenSearch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.classes import ClassRegistry
from src.db.models.config import ProjectKeymap, ProjectSettings
from src.db.models.projects import (
    PRE_DELETE_STATUSES,
    DeploymentCounter,
    Project,
    ProjectIndex,
    ProjectSlug,
)


if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from collections.abc import Collection, Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import Session

PROJECTS_REVISION = 'projects_revision'


class ProjectRepoError(Exception):
    """Base of every refusal this repository raises; services map them to the wire codes."""

    def __init__(self, slug: str, message: str) -> None:
        super().__init__(message)
        self.slug = slug


class SlugTakenError(ProjectRepoError):
    def __init__(self, slug: str) -> None:
        super().__init__(slug, f"a project named '{slug}' already exists")


class SlugRetiredError(ProjectRepoError):
    def __init__(self, slug: str) -> None:
        super().__init__(slug, f"'{slug}' was used by a deleted project and its slug is retired")


class ProjectNotFoundError(ProjectRepoError):
    def __init__(self, slug: str) -> None:
        super().__init__(slug, f"no project named '{slug}'")


class InvalidTransitionError(ProjectRepoError):
    """The project's current status is not one the operation may start from."""

    def __init__(self, slug: str, status: str, allowed: Collection[str]) -> None:
        super().__init__(slug, f"'{slug}' is '{status}'; allowed from {sorted(allowed)}")
        self.status = status


class RevisionConflictError(ProjectRepoError):
    """The caller's ``expected_revision`` is not the stored one."""

    def __init__(self, slug: str, current: int) -> None:
        super().__init__(slug, f"'{slug}' is at revision {current}; refresh and retry")
        self.current = current


@dataclass(frozen=True)
class ProjectRow:
    """One project as stored, with its index names by role."""

    id: uuid.UUID
    lock_key: int
    slug: str
    display_name: str
    description: str
    status: str
    pre_delete_status: str | None
    revision: int
    config_epoch: int
    paused_at: dt.datetime | None
    paused_by: str | None
    origin: Any | None
    index_prefix: str
    storage_prefix: str
    model_prefix: str
    mlflow_experiment: str
    created_at: dt.datetime
    updated_at: dt.datetime
    created_by: str
    created_by_user_id: uuid.UUID | None
    indexes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NewProject:
    """What the create path decides; the database assigns the rest."""

    slug: str
    display_name: str
    index_prefix: str
    storage_prefix: str
    model_prefix: str
    mlflow_experiment: str
    indexes: Mapping[str, str]
    created_by: str
    description: str = ''
    origin: Any | None = None
    status: str = 'building'
    created_by_user_id: uuid.UUID | None = None


@dataclass(frozen=True)
class ProjectLoad:
    """A 0.6.0 project, loaded as it was: every column of ``projects`` the create path sets.

    ``created_at``, ``updated_at``, ``revision`` and ``config_epoch`` are preserved. There is no
    archive time: the schema keeps none (an archived project is ``status = 'archived'``).
    """

    slug: str
    display_name: str
    description: str
    status: str
    revision: int
    created_at: dt.datetime
    updated_at: dt.datetime
    index_prefix: str
    storage_prefix: str
    model_prefix: str
    mlflow_experiment: str
    indexes: Mapping[str, str]
    created_by: str
    pre_delete_status: str | None = None
    config_epoch: int = 0
    paused_at: dt.datetime | None = None
    paused_by: str | None = None
    origin: Any | None = None
    created_by_user_id: uuid.UUID | None = None


# ---------------------------------------------------------------- reads


def _row(project: Project, indexes: Mapping[str, str]) -> ProjectRow:
    return ProjectRow(
        id=project.id,
        lock_key=project.lock_key,
        slug=project.slug,
        display_name=project.display_name,
        description=project.description,
        status=project.status,
        pre_delete_status=project.pre_delete_status,
        revision=project.revision,
        config_epoch=project.config_epoch,
        paused_at=project.paused_at,
        paused_by=project.paused_by,
        origin=project.origin,
        index_prefix=project.index_prefix,
        storage_prefix=project.storage_prefix,
        model_prefix=project.model_prefix,
        mlflow_experiment=project.mlflow_experiment,
        created_at=project.created_at,
        updated_at=project.updated_at,
        created_by=project.created_by,
        created_by_user_id=project.created_by_user_id,
        indexes=dict(indexes),
    )


def _index_names_sync(
    session: Session, project_ids: Sequence[uuid.UUID] | None
) -> dict[uuid.UUID, dict[str, str]]:
    stmt = select(ProjectIndex.project_id, ProjectIndex.role, ProjectIndex.index_name)
    if project_ids is not None:
        stmt = stmt.where(ProjectIndex.project_id.in_(project_ids))
    by_project: dict[uuid.UUID, dict[str, str]] = {}
    for project_id, role, name in session.execute(stmt).all():
        by_project.setdefault(project_id, {})[role] = name
    return by_project


def _one_sync(session: Session, where: Any) -> ProjectRow | None:
    project = session.execute(select(Project).where(where)).scalar_one_or_none()
    if project is None:
        return None
    return _row(project, _index_names_sync(session, [project.id]).get(project.id, {}))


def get_by_slug_sync(session: Session, slug: str) -> ProjectRow | None:
    return _one_sync(session, Project.slug == slug)


def list_projects_sync(session: Session) -> list[ProjectRow]:
    """Every existing project (a deleted one has no row), ordered by slug."""
    projects = session.execute(select(Project).order_by(Project.slug)).scalars().all()
    names = _index_names_sync(session, None)
    return [_row(p, names.get(p.id, {})) for p in projects]


def retired_slugs_sync(session: Session) -> list[str]:
    stmt = (
        select(ProjectSlug.slug)
        .where(ProjectSlug.retired_at.is_not(None))
        .order_by(ProjectSlug.slug)
    )
    return list(session.execute(stmt).scalars().all())


def read_projects_revision_sync(session: Session) -> int:
    value = session.scalar(
        select(DeploymentCounter.value).where(DeploymentCounter.name == PROJECTS_REVISION)
    )
    return int(value or 0)


async def get_by_slug(session: AsyncSession, slug: str) -> ProjectRow | None:
    return await session.run_sync(get_by_slug_sync, slug)


async def get_by_id(session: AsyncSession, project_id: uuid.UUID) -> ProjectRow | None:
    return await session.run_sync(lambda sync: _one_sync(sync, Project.id == project_id))


async def bump_projects_revision(session: AsyncSession) -> int:
    stmt = (
        pg_insert(DeploymentCounter)
        .values(name=PROJECTS_REVISION, value=1)
        .on_conflict_do_update(
            index_elements=[DeploymentCounter.name], set_={'value': DeploymentCounter.value + 1}
        )
        .returning(DeploymentCounter.value)
    )
    return int((await session.execute(stmt)).scalar_one())


# ---------------------------------------------------------------- create and load


async def _claim_slug(session: AsyncSession, slug: str, created_at: dt.datetime | None) -> None:
    """Reserve the slug; the primary key makes a second creator wait, then lose."""
    values: dict[str, Any] = {'slug': slug}
    if created_at is not None:
        values['created_at'] = created_at
    stmt = (
        pg_insert(ProjectSlug)
        .values(**values)
        .on_conflict_do_nothing(index_elements=[ProjectSlug.slug])
        .returning(ProjectSlug.slug)
    )
    if (await session.execute(stmt)).scalar_one_or_none() is not None:
        return
    retired_at = (
        await session.execute(select(ProjectSlug.retired_at).where(ProjectSlug.slug == slug))
    ).scalar_one()
    raise SlugRetiredError(slug) if retired_at is not None else SlugTakenError(slug)


async def _insert_project_rows(
    session: AsyncSession, values: dict[str, Any], indexes: Mapping[str, str], actor: str
) -> uuid.UUID:
    """Slug, project, indexes, class registry, settings and keymap rows, then the counter."""
    await _claim_slug(session, values['slug'], values.get('created_at'))
    values = {
        k: v for k, v in values.items() if not (k == 'origin' and v is None)
    }  # SQL NULL, not JSON null
    project_id = (
        await session.execute(pg_insert(Project).values(**values).returning(Project.id))
    ).scalar_one()
    for role, name in indexes.items():
        session.add(ProjectIndex(project_id=project_id, role=role, index_name=name))
    session.add(ClassRegistry(project_id=project_id))
    session.add(ProjectSettings(project_id=project_id, updated_by=actor))
    session.add(ProjectKeymap(project_id=project_id, updated_by=actor))
    await session.flush()
    await bump_projects_revision(session)
    return project_id


async def create_project(session: AsyncSession, new: NewProject) -> ProjectRow:
    """The create transaction: every row of a new project.

    Raises ``SlugTakenError`` or ``SlugRetiredError``; the caller rolls the transaction back.
    """
    values: dict[str, Any] = {
        'slug': new.slug,
        'display_name': new.display_name,
        'description': new.description,
        'status': new.status,
        'origin': new.origin,
        'index_prefix': new.index_prefix,
        'storage_prefix': new.storage_prefix,
        'model_prefix': new.model_prefix,
        'mlflow_experiment': new.mlflow_experiment,
        'created_by': new.created_by,
        'created_by_user_id': new.created_by_user_id,
    }
    project_id = await _insert_project_rows(session, values, new.indexes, new.created_by)
    row = await get_by_id(session, project_id)
    assert row is not None  # just inserted in this transaction
    return row


async def load_project(session: AsyncSession, row: ProjectLoad) -> uuid.UUID:
    """Upgrade entry: the rows ``create_project`` writes, with the given ids and timestamps.

    No default embedding spaces (the upgrade writes its own) and no OpenSearch call.
    """
    values = {
        'slug': row.slug,
        'display_name': row.display_name,
        'description': row.description,
        'status': row.status,
        'pre_delete_status': row.pre_delete_status,
        'revision': row.revision,
        'config_epoch': row.config_epoch,
        'paused_at': row.paused_at,
        'paused_by': row.paused_by,
        'origin': row.origin,
        'index_prefix': row.index_prefix,
        'storage_prefix': row.storage_prefix,
        'model_prefix': row.model_prefix,
        'mlflow_experiment': row.mlflow_experiment,
        'created_at': row.created_at,
        'updated_at': row.updated_at,
        'created_by': row.created_by,
        'created_by_user_id': row.created_by_user_id,
    }
    return await _insert_project_rows(session, values, row.indexes, row.created_by)


# ---------------------------------------------------------------- transitions


async def _failure(
    session: AsyncSession,
    slug: str,
    allowed: Collection[str] | None,
) -> ProjectRepoError:
    """Why a guarded UPDATE touched no row."""
    current = (
        await session.execute(select(Project.status, Project.revision).where(Project.slug == slug))
    ).one_or_none()
    if current is None:
        return ProjectNotFoundError(slug)
    status, revision = current
    if allowed is not None and status not in allowed:
        return InvalidTransitionError(slug, status, allowed)
    return RevisionConflictError(slug, revision)


async def _update(
    session: AsyncSession,
    slug: str,
    values: dict[str, Any],
    *,
    allowed: Collection[str] | None = None,
    expected_revision: int | None = None,
    bump_revision: bool = True,
) -> ProjectRow:
    """One guarded UPDATE: status in ``allowed`` and revision equal to ``expected_revision``."""
    values = {**values, 'updated_at': func.now()}
    if bump_revision:
        values['revision'] = Project.revision + 1
    stmt = update(Project).where(Project.slug == slug)
    if allowed is not None:
        stmt = stmt.where(Project.status.in_(list(allowed)))
    if expected_revision is not None:
        stmt = stmt.where(Project.revision == expected_revision)
    project_id = (
        await session.execute(stmt.values(**values).returning(Project.id))
    ).scalar_one_or_none()
    if project_id is None:
        raise await _failure(session, slug, allowed)
    await bump_projects_revision(session)
    row = await get_by_id(session, project_id)
    assert row is not None
    return row


async def patch_project(
    session: AsyncSession,
    slug: str,
    *,
    expected_revision: int,
    display_name: str | None = None,
    description: str | None = None,
) -> ProjectRow:
    values: dict[str, Any] = {}
    if display_name is not None:
        values['display_name'] = display_name
    if description is not None:
        values['description'] = description
    return await _update(session, slug, values, expected_revision=expected_revision)


async def set_status(
    session: AsyncSession,
    slug: str,
    *,
    to: str,
    from_states: Collection[str],
    expected_revision: int | None = None,
    bump_revision: bool = True,
) -> ProjectRow:
    """Archive, unarchive and the ``building`` to ``active``/``failed`` ends of a create."""
    return await _update(
        session,
        slug,
        {'status': to},
        allowed=from_states,
        expected_revision=expected_revision,
        bump_revision=bump_revision,
    )


async def mark_deleting(
    session: AsyncSession,
    slug: str,
    *,
    from_states: Collection[str],
    pre_delete_status: str | None = None,
    expected_revision: int | None = None,
) -> ProjectRow:
    """``deleting``, remembering the status to roll back to (the current one unless overridden)."""
    if pre_delete_status is not None and pre_delete_status not in PRE_DELETE_STATUSES:
        msg = f'pre_delete_status must be one of {PRE_DELETE_STATUSES}'
        raise ValueError(msg)
    return await _update(
        session,
        slug,
        {
            'status': 'deleting',
            'pre_delete_status': pre_delete_status
            if pre_delete_status is not None
            else Project.status,
        },
        allowed=from_states,
        expected_revision=expected_revision,
        bump_revision=False,
    )


async def claim_delete(session: AsyncSession, slug: str, *, expected_revision: int) -> ProjectRow:
    """Take the finish of a delete: of two finishers reading one revision, one wins."""
    return await _update(
        session, slug, {}, allowed=('deleting',), expected_revision=expected_revision
    )


async def rollback_delete(session: AsyncSession, slug: str) -> ProjectRow:
    """``deleting`` back to the status the delete found (``failed`` if none was recorded)."""
    return await _update(
        session,
        slug,
        {'status': func.coalesce(Project.pre_delete_status, 'failed'), 'pre_delete_status': None},
        allowed=('deleting',),
        bump_revision=False,
    )


async def finish_delete(session: AsyncSession, slug: str) -> None:
    """Retire the slug, delete the project row (the cascade removes every control row), bump."""
    project_id = (
        await session.execute(
            delete(Project)
            .where(Project.slug == slug, Project.status == 'deleting')
            .returning(Project.id)
        )
    ).scalar_one_or_none()
    if project_id is None:
        raise await _failure(session, slug, ('deleting',))
    await session.execute(
        update(ProjectSlug).where(ProjectSlug.slug == slug).values(retired_at=func.now())
    )
    await bump_projects_revision(session)


async def set_paused(
    session: AsyncSession, slug: str, *, paused: bool, by: str | None = None
) -> ProjectRow:
    """Pause or resume the project's workers; idempotent (a repeat keeps the first pause time)."""
    values: dict[str, Any] = (
        {
            'paused_at': func.coalesce(Project.paused_at, func.now()),
            'paused_by': func.coalesce(Project.paused_by, by),
        }
        if paused
        else {'paused_at': None, 'paused_by': None}
    )
    return await _update(session, slug, values, bump_revision=False)
