"""Config store repository, part 2: activations, their event log and the activation load.

An explicit deactivation ("off") is not a row in ``config_activations`` (its ``config_id`` is not
nullable). It is an event with no target (``to_config_id`` and ``to_revision`` both NULL): the axis
then reads as ``off`` until the next activation, and "never activated" (no event) stays distinct
from "off".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from src.db.models.config import (
    ConfigActivation,
    ConfigActivationEvent,
    ConfigDocument,
    ConfigRevision,
)
from src.db.models.projects import Project
from src.db.repos.config_base import (
    AXIS_KIND,
    DEFAULT_ACTOR,
    ActivationEvent,
    ActivationEventLoad,
    ActiveConflictError,
    ActiveRef,
    AxisKindError,
    bump_epoch,
)


if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


def _expected_pair(expected: dict[str, Any] | None) -> tuple[str | None, int | None]:
    return (None, None) if expected is None else (expected.get('name'), expected.get('revision'))


async def _active_rows(
    session: AsyncSession, project_id: uuid.UUID, axis: str | None
) -> dict[str, ActiveRef]:
    stmt = (
        select(
            ConfigActivation.axis,
            ConfigActivation.kind,
            ConfigActivation.config_id,
            ConfigDocument.name,
            ConfigActivation.revision,
            ConfigActivation.activated_at,
            ConfigActivation.activated_by,
        )
        .join(ConfigDocument, ConfigDocument.id == ConfigActivation.config_id)
        .where(ConfigActivation.project_id == project_id)
    )
    if axis is not None:
        stmt = stmt.where(ConfigActivation.axis == axis)
    out: dict[str, ActiveRef] = {}
    for a, kind, config_id, name, revision, at, by in (await session.execute(stmt)).all():
        out[a] = ActiveRef(a, 'active', config_id, kind, name, revision, at, by)
    return out


async def _off_axes(
    session: AsyncSession, project_id: uuid.UUID, axis: str | None
) -> dict[str, ActivationEvent]:
    """Axes whose latest event deactivated them (and that hold no activation row)."""
    newest = select(func.max(ConfigActivationEvent.id)).where(
        ConfigActivationEvent.project_id == project_id
    )
    if axis is not None:
        newest = newest.where(ConfigActivationEvent.axis == axis)
    latest = select(ConfigActivationEvent).where(
        ConfigActivationEvent.id.in_(newest.group_by(ConfigActivationEvent.axis))
    )
    rows = (await session.execute(latest)).scalars().all()
    return {r.axis: _event(r) for r in rows if r.to_config_id is None}


def _event(row: ConfigActivationEvent, from_name: str | None = None) -> ActivationEvent:
    return ActivationEvent(
        row.id,
        row.axis,
        row.from_config_id,
        row.from_revision,
        row.to_config_id,
        row.to_revision,
        row.reason,
        row.at,
        row.by,
        from_name,
    )


async def get_activations(session: AsyncSession, project_id: uuid.UUID) -> dict[str, ActiveRef]:
    """Every axis that was ever activated: an ``active`` ref, or ``off`` after a deactivation."""
    active = await _active_rows(session, project_id, None)
    for axis_name, event in (await _off_axes(session, project_id, None)).items():
        if axis_name not in active:
            active[axis_name] = ActiveRef(
                axis_name, 'off', activated_at=event.at, activated_by=event.by
            )
    return active


async def get_activation(
    session: AsyncSession, project_id: uuid.UUID, axis: str
) -> ActiveRef | None:
    return (await get_activations(session, project_id)).get(axis)


def _matches_expected(current: ActiveRef | None, expected: dict[str, Any] | None) -> bool:
    """Never activated and explicitly off are the same state for compare-and-set."""
    exp_name, exp_revision = _expected_pair(expected)
    cur_name = current.name if current is not None and current.state == 'active' else None
    if exp_name is None and cur_name is None:
        return True
    return exp_name == cur_name and exp_revision == (current.revision if current else None)


async def activate(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    axis: str,
    config_id: uuid.UUID | None,
    revision: int | None,
    expected_active: dict[str, Any] | None,
    by: str = DEFAULT_ACTOR,
    reason: str = 'activate',
) -> ActiveRef:
    """Point ``axis`` at ``config_id@revision`` (``None``: deactivate), if ``expected_active`` holds.

    ``expected_active`` is ``{name, revision}`` or ``None`` for "nothing active"; a stale one is
    :class:`ActiveConflictError`. The activation row, its event and the epoch bump are one
    transaction. The vlm axis names a deployment-wide document, so its caller uses a deployment
    transaction (the name lookup must see the endpoint).
    """
    if axis not in AXIS_KIND:
        msg = f'unknown axis {axis!r}'
        raise AxisKindError(msg)
    current = await get_activation(session, project_id, axis)
    if not _matches_expected(current, expected_active):
        raise ActiveConflictError(
            current.as_expected() if current is not None and current.state == 'active' else None
        )
    held = current if current is not None and current.state == 'active' else None
    previous_id = held.config_id if held else None
    previous_revision = held.revision if held else None

    if config_id is None:
        if held is not None:
            await _cas_delete(session, project_id, axis, held)
        target_id, target_revision = None, None
    else:
        if revision is None:
            msg = 'an activation names a revision'
            raise ValueError(msg)
        target_id, target_revision = config_id, revision
        await _cas_write(session, project_id, axis, held, config_id, revision, by)
    await session.execute(
        insert(ConfigActivationEvent).values(
            project_id=project_id,
            axis=axis,
            from_config_id=previous_id,
            from_revision=previous_revision,
            to_config_id=target_id,
            to_revision=target_revision,
            reason=reason,
            by=by,
        )
    )
    await bump_epoch(session, project_id)
    ref = await get_activation(session, project_id, axis)
    if ref is None:
        msg = 'activation vanished inside its own transaction'
        raise RuntimeError(msg)
    return ref


async def _cas_delete(
    session: AsyncSession, project_id: uuid.UUID, axis: str, held: ActiveRef
) -> None:
    result = await session.execute(
        delete(ConfigActivation).where(
            ConfigActivation.project_id == project_id,
            ConfigActivation.axis == axis,
            ConfigActivation.config_id == held.config_id,
            ConfigActivation.revision == held.revision,
        )
    )
    if result.rowcount == 0:  # type: ignore[attr-defined]
        raise ActiveConflictError(None)


async def _cas_write(
    session: AsyncSession,
    project_id: uuid.UUID,
    axis: str,
    held: ActiveRef | None,
    config_id: uuid.UUID,
    revision: int,
    by: str,
) -> None:
    kind = AXIS_KIND[axis]
    try:
        async with session.begin_nested():
            if held is None:
                result = await session.execute(
                    pg_insert(ConfigActivation)
                    .values(
                        project_id=project_id,
                        axis=axis,
                        kind=kind,
                        config_id=config_id,
                        revision=revision,
                        activated_by=by,
                    )
                    .on_conflict_do_nothing(index_elements=['project_id', 'axis'])
                    .returning(ConfigActivation.axis)
                )
                written = result.scalar_one_or_none() is not None
            else:
                result = await session.execute(
                    update(ConfigActivation)
                    .where(
                        ConfigActivation.project_id == project_id,
                        ConfigActivation.axis == axis,
                        ConfigActivation.config_id == held.config_id,
                        ConfigActivation.revision == held.revision,
                    )
                    .values(
                        kind=kind,
                        config_id=config_id,
                        revision=revision,
                        activated_at=func.now(),
                        activated_by=by,
                    )
                    .returning(ConfigActivation.axis)
                )
                written = result.scalar_one_or_none() is not None
    except IntegrityError as exc:
        constraint = getattr(getattr(exc.orig, 'diag', None), 'constraint_name', '') or ''
        if 'axis_kind' in constraint or 'config_id_kind' in constraint:
            msg = f'axis {axis!r} does not take this kind of document'
            raise AxisKindError(msg) from exc
        raise
    if not written:
        raise ActiveConflictError(held.as_expected() if held else None)


async def list_activation_events(
    session: AsyncSession, project_id: uuid.UUID, axis: str, *, limit: int = 50
) -> list[ActivationEvent]:
    """The axis's events, newest first, each with the name of the document it moved away from."""
    rows = await session.execute(
        select(ConfigActivationEvent, ConfigDocument.name)
        .outerjoin(ConfigDocument, ConfigDocument.id == ConfigActivationEvent.from_config_id)
        .where(ConfigActivationEvent.project_id == project_id, ConfigActivationEvent.axis == axis)
        .order_by(ConfigActivationEvent.id.desc())
        .limit(limit)
    )
    return [_event(r, from_name) for r, from_name in rows.all()]


async def load_activation_events(
    session: AsyncSession, project_id: uuid.UUID | None, events: Sequence[ActivationEventLoad]
) -> None:
    """Insert the events in order; the last event of each axis becomes that axis's activation.

    ``project_id`` is required (``None`` is accepted only for an empty list). A last event that
    with no target leaves the axis off: an event and no activation row.
    """
    if not events:
        return
    if project_id is None:
        msg = 'activation events belong to a project'
        raise ValueError(msg)
    last: dict[str, ActivationEventLoad] = {}
    for event in events:
        await session.execute(
            insert(ConfigActivationEvent).values(
                project_id=project_id,
                axis=event.axis,
                from_config_id=event.from_config_id,
                from_revision=event.from_revision,
                to_config_id=event.to_config_id,
                to_revision=event.to_revision,
                reason=event.reason,
                at=event.at,
                by=event.by,
            )
        )
        last[event.axis] = event
    for axis_name, event in last.items():
        if event.to_config_id is None or event.to_revision is None:
            continue
        await session.execute(
            insert(ConfigActivation).values(
                project_id=project_id,
                axis=axis_name,
                kind=AXIS_KIND[axis_name],
                config_id=event.to_config_id,
                revision=event.to_revision,
                activated_at=event.at,
                activated_by=event.by,
            )
        )


@dataclass(frozen=True)
class ActiveBody:
    """One project's active document on an axis, at the pinned revision."""

    slug: str
    name: str
    revision: int
    body: dict[str, Any]


async def list_active_bodies(session: AsyncSession, axis: str) -> list[ActiveBody]:
    """The pinned body each project that is not being deleted has activated on ``axis``."""
    rows = await session.execute(
        select(Project.slug, ConfigDocument.name, ConfigActivation.revision, ConfigRevision.body)
        .select_from(ConfigActivation)
        .join(Project, Project.id == ConfigActivation.project_id)
        .join(ConfigDocument, ConfigDocument.id == ConfigActivation.config_id)
        .join(
            ConfigRevision,
            and_(
                ConfigRevision.config_id == ConfigActivation.config_id,
                ConfigRevision.revision == ConfigActivation.revision,
            ),
        )
        .where(ConfigActivation.axis == axis, Project.status != 'deleting')
        .order_by(Project.slug)
    )
    return [ActiveBody(*r) for r in rows.all()]


async def list_axis_state(session: AsyncSession, axis: str) -> dict[str, ActiveRef | None]:
    """``slug -> activation`` of ``axis`` for every project not being deleted.

    ``None`` means the project never activated the axis; an ``off`` ref means it switched it off.
    """
    slugs = dict(
        (
            await session.execute(
                select(Project.id, Project.slug).where(Project.status != 'deleting')
            )
        ).all()
    )
    out: dict[str, ActiveRef | None] = dict.fromkeys(slugs.values())
    named = await session.execute(
        select(
            ConfigActivation.project_id,
            ConfigDocument.id,
            ConfigDocument.name,
            ConfigActivation.revision,
        )
        .join(ConfigDocument, ConfigDocument.id == ConfigActivation.config_id)
        .where(ConfigActivation.axis == axis)
    )
    held: set[uuid.UUID] = set()
    for project_id, config_id, name, revision in named.all():
        if project_id in slugs:
            held.add(project_id)
            out[slugs[project_id]] = ActiveRef(
                axis, 'active', config_id, AXIS_KIND[axis], name, revision
            )
    ever = await session.execute(
        select(ConfigActivationEvent.project_id)
        .where(ConfigActivationEvent.axis == axis)
        .distinct()
    )
    for (project_id,) in ever.all():
        if project_id in slugs and project_id not in held:
            out[slugs[project_id]] = ActiveRef(axis, 'off')
    return out
