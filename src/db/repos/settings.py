"""Project settings, keymap, internal state markers and the UMAP view state.

Every function takes a session already inside ``project_tx`` (or ``deployment_tx`` for the
upgrade) and never commits. Settings, keymap and ``project_state`` writes carry the revision
the writer read: ``UPDATE ... WHERE revision = :expected`` blocks on a concurrent writer and
re-checks its predicate after that writer commits, so of two writers naming one revision exactly
one matches (https://www.postgresql.org/docs/current/transaction-iso.html#XACT-READ-COMMITTED).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.config import (
    LOCATION_POLICIES,
    PROJECT_STATE_KEYS,
    ProjectionState,
    ProjectKeymap,
    ProjectSettings,
    ProjectState,
)
from src.db.repos.config import DEFAULT_ACTOR, RevisionConflictError, bump_epoch


if TYPE_CHECKING:
    import datetime as dt
    import uuid
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession

SETTINGS_COLUMNS = (
    'defaults',
    'ingest_policy',
    'vlm_policy',
    'vlm_scope_reps',
    'search',
    'location_policy',
)


@dataclass(frozen=True)
class SettingsRow:
    project_id: uuid.UUID
    defaults: dict[str, Any]
    ingest_policy: dict[str, Any] | None
    vlm_policy: dict[str, Any] | None
    vlm_scope_reps: dict[str, Any] | None
    search: dict[str, Any] | None
    location_policy: str
    revision: int
    updated_at: dt.datetime
    updated_by: str


@dataclass(frozen=True)
class KeymapRow:
    project_id: uuid.UUID
    overrides: dict[str, list[str]]
    revision: int
    updated_at: dt.datetime
    updated_by: str


@dataclass(frozen=True)
class StateRow:
    key: str
    value: dict[str, Any]
    revision: int
    updated_at: dt.datetime


@dataclass(frozen=True)
class ProjectionStateRow:
    project_id: uuid.UUID
    projection_version: int
    scope: str
    cluster_uid: uuid.UUID | None
    n_points: int
    n_components: int
    metric: str
    fitted_at: dt.datetime
    reducer_storage_key: str | None
    job_id: uuid.UUID | None


def _settings_row(row: ProjectSettings) -> SettingsRow:
    return SettingsRow(
        row.project_id,
        row.defaults,
        row.ingest_policy,
        row.vlm_policy,
        row.vlm_scope_reps,
        row.search,
        row.location_policy,
        row.revision,
        row.updated_at,
        row.updated_by,
    )


async def get_settings(
    session: AsyncSession, project_id: uuid.UUID, *, for_update: bool = False
) -> SettingsRow | None:
    """The project's settings row, ``None`` for a project that has none. ``for_update`` locks it
    until the transaction ends (a read-modify-write of one policy column)."""
    stmt = select(ProjectSettings).where(ProjectSettings.project_id == project_id)
    if for_update:
        stmt = stmt.with_for_update()
    row = (await session.execute(stmt)).scalar_one_or_none()
    return None if row is None else _settings_row(row)


async def put_settings(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    expected_revision: int,
    changes: Mapping[str, Any],
    by: str = DEFAULT_ACTOR,
) -> SettingsRow:
    """Replace the named columns when ``expected_revision`` is current; the revision advances by one.

    ``changes`` maps column names (``SETTINGS_COLUMNS``) to the whole new value. A stale
    revision is :class:`RevisionConflictError`. A project that has no settings row yet reads as
    revision ``0``.
    """
    unknown = set(changes) - set(SETTINGS_COLUMNS)
    if unknown:
        msg = f'not settings columns: {sorted(unknown)}'
        raise ValueError(msg)
    if 'location_policy' in changes and changes['location_policy'] not in LOCATION_POLICIES:
        msg = f'location_policy must be one of {LOCATION_POLICIES}'
        raise ValueError(msg)
    if expected_revision == 0:
        created = await session.execute(
            pg_insert(ProjectSettings)
            .values(project_id=project_id, updated_by=by, **changes)
            .on_conflict_do_nothing(index_elements=['project_id'])
            .returning(ProjectSettings.revision)
        )
        won = created.scalar_one_or_none() is not None
    else:
        updated = await session.execute(
            update(ProjectSettings)
            .where(
                ProjectSettings.project_id == project_id,
                ProjectSettings.revision == expected_revision,
            )
            .values(
                **changes,
                revision=ProjectSettings.revision + 1,
                updated_at=func.now(),
                updated_by=by,
            )
            .returning(ProjectSettings.revision)
        )
        won = updated.scalar_one_or_none() is not None
    if not won:
        current = await get_settings(session, project_id)
        raise RevisionConflictError(current.revision if current else 0)
    row = await get_settings(session, project_id)
    if row is None:
        msg = 'settings vanished inside their own transaction'
        raise RuntimeError(msg)
    return row


async def get_keymap(session: AsyncSession, project_id: uuid.UUID) -> KeymapRow | None:
    row = (
        await session.execute(select(ProjectKeymap).where(ProjectKeymap.project_id == project_id))
    ).scalar_one_or_none()
    if row is None:
        return None
    return KeymapRow(row.project_id, row.overrides, row.revision, row.updated_at, row.updated_by)


async def put_keymap(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    overrides: dict[str, list[str]],
    expected_revision: int,
    by: str = DEFAULT_ACTOR,
) -> KeymapRow:
    """Replace the override map when ``expected_revision`` is current (``0``: no row yet)."""
    if expected_revision == 0:
        created = await session.execute(
            pg_insert(ProjectKeymap)
            .values(project_id=project_id, overrides=overrides, updated_by=by)
            .on_conflict_do_nothing(index_elements=['project_id'])
            .returning(ProjectKeymap.revision)
        )
        won = created.scalar_one_or_none() is not None
    else:
        updated = await session.execute(
            update(ProjectKeymap)
            .where(
                ProjectKeymap.project_id == project_id, ProjectKeymap.revision == expected_revision
            )
            .values(
                overrides=overrides,
                revision=ProjectKeymap.revision + 1,
                updated_at=func.now(),
                updated_by=by,
            )
            .returning(ProjectKeymap.revision)
        )
        won = updated.scalar_one_or_none() is not None
    if not won:
        current = await get_keymap(session, project_id)
        raise RevisionConflictError(current.revision if current else 0)
    await bump_epoch(session, project_id)
    row = await get_keymap(session, project_id)
    if row is None:
        msg = 'keymap vanished inside its own transaction'
        raise RuntimeError(msg)
    return row


def _check_state_key(key: str) -> None:
    if key not in PROJECT_STATE_KEYS:
        msg = f'unknown project_state key {key!r}'
        raise ValueError(msg)


async def get_state(session: AsyncSession, project_id: uuid.UUID, key: str) -> StateRow | None:
    _check_state_key(key)
    row = (
        await session.execute(
            select(ProjectState).where(
                ProjectState.project_id == project_id, ProjectState.key == key
            )
        )
    ).scalar_one_or_none()
    return None if row is None else StateRow(row.key, row.value, row.revision, row.updated_at)


async def put_state(
    session: AsyncSession,
    project_id: uuid.UUID,
    key: str,
    value: dict[str, Any],
    *,
    expected_revision: int,
) -> StateRow:
    """Write one internal marker when ``expected_revision`` is current (``0``: never written)."""
    _check_state_key(key)
    if expected_revision == 0:
        created = await session.execute(
            pg_insert(ProjectState)
            .values(project_id=project_id, key=key, value=value)
            .on_conflict_do_nothing(index_elements=['project_id', 'key'])
            .returning(ProjectState.revision)
        )
        won = created.scalar_one_or_none() is not None
    else:
        updated = await session.execute(
            update(ProjectState)
            .where(
                ProjectState.project_id == project_id,
                ProjectState.key == key,
                ProjectState.revision == expected_revision,
            )
            .values(value=value, revision=ProjectState.revision + 1, updated_at=func.now())
            .returning(ProjectState.revision)
        )
        won = updated.scalar_one_or_none() is not None
    if not won:
        current = await get_state(session, project_id, key)
        raise RevisionConflictError(current.revision if current else 0)
    row = await get_state(session, project_id, key)
    if row is None:
        msg = 'state vanished inside its own transaction'
        raise RuntimeError(msg)
    return row


async def get_projection_state(
    session: AsyncSession, project_id: uuid.UUID
) -> ProjectionStateRow | None:
    row = (
        await session.execute(
            select(ProjectionState).where(ProjectionState.project_id == project_id)
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    return ProjectionStateRow(
        row.project_id,
        row.projection_version,
        row.scope,
        row.cluster_uid,
        row.n_points,
        row.n_components,
        row.metric,
        row.fitted_at,
        row.reducer_storage_key,
        row.job_id,
    )


async def put_projection_state(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    projection_version: int,
    scope: str,
    cluster_uid: uuid.UUID | None,
    n_points: int,
    n_components: int,
    metric: str,
    fitted_at: dt.datetime,
    reducer_storage_key: str | None,
    job_id: uuid.UUID | None = None,
) -> None:
    """Replace the project's current view state (one row per project)."""
    values = {
        'projection_version': projection_version,
        'scope': scope,
        'cluster_uid': cluster_uid,
        'n_points': n_points,
        'n_components': n_components,
        'metric': metric,
        'fitted_at': fitted_at,
        'reducer_storage_key': reducer_storage_key,
        'job_id': job_id,
    }
    stmt = pg_insert(ProjectionState).values(project_id=project_id, **values)
    await session.execute(stmt.on_conflict_do_update(index_elements=['project_id'], set_=values))


# --- the 0.6.0 upgrade load ---------------------------------------------------------------------


async def load_setting(
    session: AsyncSession, project_id: uuid.UUID, key: str, body: Any, *, revision: int
) -> None:
    """Write one settings column of an existing project at ``revision`` (the project create wrote
    the row). Raises when ``key`` is not a settings column."""
    if key not in SETTINGS_COLUMNS:
        msg = f'not a settings column: {key!r}'
        raise ValueError(msg)
    result = await session.execute(
        update(ProjectSettings)
        .where(ProjectSettings.project_id == project_id)
        .values({key: body, 'revision': revision})
        .returning(ProjectSettings.project_id)
    )
    if result.scalar_one_or_none() is None:
        await session.execute(
            insert(ProjectSettings).values(
                project_id=project_id, updated_by='upgrade', revision=revision, **{key: body}
            )
        )


async def load_keymap(
    session: AsyncSession, project_id: uuid.UUID, body: dict[str, list[str]], *, revision: int
) -> None:
    """Write the keymap overrides of a project at ``revision``."""
    result = await session.execute(
        update(ProjectKeymap)
        .where(ProjectKeymap.project_id == project_id)
        .values(overrides=body, revision=revision)
        .returning(ProjectKeymap.project_id)
    )
    if result.scalar_one_or_none() is None:
        await session.execute(
            insert(ProjectKeymap).values(
                project_id=project_id, overrides=body, revision=revision, updated_by='upgrade'
            )
        )
