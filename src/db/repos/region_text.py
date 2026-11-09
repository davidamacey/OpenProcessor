"""Region text repository: reader runs and consensus groups.

Every function takes a session already inside ``project_tx`` and never commits. A recompute
replaces groups in the caller's one transaction: every current group that holds any of the new
members is deleted (its members cascade) and the new groups are inserted, so no superseded
history is kept and a box is never in two groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select, tuple_, update

from src.db.models.region_text import (
    RegionTextConsensusGroup,
    RegionTextConsensusMember,
    RegionTextRun,
)


if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

KEY_CHUNK = 2000


@dataclass(frozen=True)
class MemberInput:
    item_id: str
    box_id: str
    weight: float
    agrees: bool


@dataclass(frozen=True)
class GroupInput:
    basis: str
    method: str
    n_frames: int
    n_agree: int
    tier: str
    members: Sequence[MemberInput]
    confidence: float | None = None
    jurisdiction: str | None = None
    jurisdiction_conf: float | None = None


async def open_run(
    session: AsyncSession,
    project_id: uuid.UUID,
    *,
    trigger: str,
    reader: str,
    reader_version: str,
    created_by: str,
    profile: tuple[uuid.UUID, int] | None = None,
    job_id: uuid.UUID | None = None,
    params: Mapping[str, Any] | None = None,
) -> uuid.UUID:
    """Start a run. A second open ``region_pass`` for the same reader configuration raises
    ``IntegrityError`` (``uq_region_text_runs_open_pass``); the caller joins or waits."""
    run = RegionTextRun(
        project_id=project_id,
        job_id=job_id,
        trigger=trigger,
        reader=reader,
        reader_version=reader_version,
        profile_config_id=profile[0] if profile else None,
        profile_revision=profile[1] if profile else None,
        params=dict(params or {}),
        created_by=created_by,
    )
    session.add(run)
    await session.flush()
    return run.id


async def finish_run(
    session: AsyncSession, project_id: uuid.UUID, run_id: uuid.UUID, counts: Mapping[str, Any]
) -> bool:
    """Close an open run with its counts; ``False`` when it was already closed or is not this
    project's."""
    result = await session.execute(
        update(RegionTextRun)
        .where(
            RegionTextRun.id == run_id,
            RegionTextRun.project_id == project_id,
            RegionTextRun.finished_at.is_(None),
        )
        .values(counts=dict(counts), finished_at=func.now())
        .returning(RegionTextRun.id)
    )
    return result.scalar_one_or_none() is not None


async def replace_groups(
    session: AsyncSession,
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    groups: Sequence[GroupInput],
) -> list[uuid.UUID]:
    """Store ``groups`` in place of every current group that holds one of their boxes.

    Returns the new group ids in input order. Two inputs naming one box raise ``IntegrityError``.
    """
    keys = [(m.item_id, m.box_id) for g in groups for m in g.members]
    for start in range(0, len(keys), KEY_CHUNK):
        old = select(RegionTextConsensusMember.group_id).where(
            RegionTextConsensusMember.project_id == project_id,
            tuple_(RegionTextConsensusMember.item_id, RegionTextConsensusMember.box_id).in_(
                keys[start : start + KEY_CHUNK]
            ),
        )
        await session.execute(
            delete(RegionTextConsensusGroup).where(
                RegionTextConsensusGroup.project_id == project_id,
                RegionTextConsensusGroup.id.in_(old),
            )
        )
    ids: list[uuid.UUID] = []
    for group in groups:
        row = RegionTextConsensusGroup(
            project_id=project_id,
            run_id=run_id,
            basis=group.basis,
            method=group.method,
            n_frames=group.n_frames,
            n_agree=group.n_agree,
            confidence=group.confidence,
            tier=group.tier,
            jurisdiction=group.jurisdiction,
            jurisdiction_conf=group.jurisdiction_conf,
        )
        session.add(row)
        await session.flush()
        session.add_all(
            RegionTextConsensusMember(
                group_id=row.id,
                project_id=project_id,
                item_id=m.item_id,
                box_id=m.box_id,
                weight=m.weight,
                agrees=m.agrees,
            )
            for m in group.members
        )
        await session.flush()
        ids.append(row.id)
    return ids


async def groups_for_boxes(
    session: AsyncSession, project_id: uuid.UUID, boxes: Sequence[tuple[str, str]]
) -> dict[tuple[str, str], RegionTextConsensusGroup]:
    """The current group of each ``(item_id, box_id)`` that has one."""
    found: dict[tuple[str, str], RegionTextConsensusGroup] = {}
    for start in range(0, len(boxes), KEY_CHUNK):
        rows = await session.execute(
            select(
                RegionTextConsensusMember.item_id,
                RegionTextConsensusMember.box_id,
                RegionTextConsensusGroup,
            )
            .join(
                RegionTextConsensusGroup,
                (RegionTextConsensusGroup.id == RegionTextConsensusMember.group_id)
                & (RegionTextConsensusGroup.project_id == RegionTextConsensusMember.project_id),
            )
            .where(
                RegionTextConsensusMember.project_id == project_id,
                tuple_(RegionTextConsensusMember.item_id, RegionTextConsensusMember.box_id).in_(
                    list(boxes[start : start + KEY_CHUNK])
                ),
            )
        )
        found.update({(item, box): group for item, box, group in rows})
    return found


async def members_of(
    session: AsyncSession, project_id: uuid.UUID, group_id: uuid.UUID
) -> list[RegionTextConsensusMember]:
    result = await session.execute(
        select(RegionTextConsensusMember)
        .where(
            RegionTextConsensusMember.project_id == project_id,
            RegionTextConsensusMember.group_id == group_id,
        )
        .order_by(RegionTextConsensusMember.item_id, RegionTextConsensusMember.box_id)
    )
    return list(result.scalars())
