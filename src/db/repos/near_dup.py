"""Near-duplicate repository: pending rows, candidate edges, runs, groups and decisions.

Every function takes a session already inside ``project_tx`` (or ``deployment_tx`` for the core
index) and never commits. A pending row is written before the OpenSearch write it describes:
an extra row is harmless because the rescore is idempotent per member, a missing one would be a
silent stale group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, delete, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models.classes import SearchProjectionTask
from src.db.models.near_dup import (
    NearDupDecision,
    NearDupEdge,
    NearDupGroup,
    NearDupMember,
    NearDupPending,
    NearDupRun,
)


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

# 4 bound values per row against the 32767 parameter limit of the wire protocol.
PENDING_CHUNK = 2000


async def enqueue_pending(
    session: AsyncSession,
    project_id: uuid.UUID,
    level: str,
    member_ids: Iterable[str],
    reason: str,
) -> int:
    """Mark members stale; returns how many distinct members were written.

    A member that is already pending keeps its single row: the newest reason wins, the row is
    re-aged and released from any job that claimed it, so a change that lands while a rescore
    runs is never lost when that rescore drains its claim.
    """
    ids = sorted(set(member_ids))
    for start in range(0, len(ids), PENDING_CHUNK):
        chunk = ids[start : start + PENDING_CHUNK]
        insert = pg_insert(NearDupPending).values(
            [
                {'project_id': project_id, 'level': level, 'member_id': member, 'reason': reason}
                for member in chunk
            ]
        )
        await session.execute(
            insert.on_conflict_do_update(
                constraint='uq_near_dup_pending_member',
                set_={
                    'reason': insert.excluded.reason,
                    'enqueued_at': func.now(),
                    'claimed_by_job_id': None,
                },
            )
        )
    return len(ids)


EDGE_CHUNK = 4000
GROUP_CHUNK = 2000
MEMBER_CHUNK = 4000


def _same_project(column: Any, project_id: uuid.UUID | None) -> Any:
    return column.is_(None) if project_id is None else column == project_id


@dataclass(frozen=True)
class ClaimedMember:
    member_id: str
    reason: str


async def claim_pending(
    session: AsyncSession, project_id: uuid.UUID | None, level: str, job_id: uuid.UUID, limit: int
) -> list[ClaimedMember]:
    """Claim the oldest pending rows of one (project, level) for ``job_id`` with ``SKIP LOCKED``.

    Only one ``near_dup`` job per (project, level) is active (the exclusive group), so a row a
    dead predecessor claimed is safe to take over: every pending row of the level is claimable.
    """
    oldest = (
        select(NearDupPending.id)
        .where(_same_project(NearDupPending.project_id, project_id), NearDupPending.level == level)
        .order_by(NearDupPending.enqueued_at, NearDupPending.id)
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    rows = await session.execute(
        update(NearDupPending)
        .where(NearDupPending.id.in_(oldest))
        .values(claimed_by_job_id=job_id)
        .returning(NearDupPending.member_id, NearDupPending.reason)
    )
    return [ClaimedMember(r.member_id, r.reason) for r in rows]


async def drain_claimed(session: AsyncSession, job_id: uuid.UUID, member_ids: Iterable[str]) -> int:
    """Delete the rows ``job_id`` still holds for ``member_ids``; a row re-enqueued meanwhile was
    released from the claim by :func:`enqueue_pending` and stays."""
    ids = list(member_ids)
    done = 0
    for start in range(0, len(ids), EDGE_CHUNK):
        result = await session.execute(
            delete(NearDupPending).where(
                NearDupPending.claimed_by_job_id == job_id,
                NearDupPending.member_id.in_(ids[start : start + EDGE_CHUNK]),
            )
        )
        done += result.rowcount  # type: ignore[attr-defined]
    return done


async def replace_edges(
    session: AsyncSession,
    project_id: uuid.UUID | None,
    level: str,
    space: str,
    member_ids: Sequence[str],
    edges: Iterable[tuple[str, str, float]],
) -> None:
    """Drop every stored edge touching ``member_ids`` and store ``edges`` (``a < b``) instead."""
    for start in range(0, len(member_ids), EDGE_CHUNK):
        ids = member_ids[start : start + EDGE_CHUNK]
        await session.execute(
            delete(NearDupEdge).where(
                _same_project(NearDupEdge.project_id, project_id),
                NearDupEdge.level == level,
                NearDupEdge.space == space,
                or_(NearDupEdge.a_id.in_(ids), NearDupEdge.b_id.in_(ids)),
            )
        )
    pairs = sorted(set(edges))
    for start in range(0, len(pairs), EDGE_CHUNK):
        insert = pg_insert(NearDupEdge).values(
            [
                {
                    'project_id': project_id,
                    'level': level,
                    'space': space,
                    'a_id': a,
                    'b_id': b,
                    'kind': 'embedding',
                    'score': score,
                }
                for a, b, score in pairs[start : start + EDGE_CHUNK]
            ]
        )
        await session.execute(
            insert.on_conflict_do_update(
                constraint='uq_near_dup_edges_pair',
                set_={'score': insert.excluded.score, 'computed_at': func.now()},
            )
        )


async def clear_edges(
    session: AsyncSession, project_id: uuid.UUID | None, level: str, space: str
) -> None:
    await session.execute(
        delete(NearDupEdge).where(
            _same_project(NearDupEdge.project_id, project_id),
            NearDupEdge.level == level,
            NearDupEdge.space == space,
        )
    )


async def load_edges(
    session: AsyncSession,
    project_id: uuid.UUID | None,
    level: str,
    space: str,
    *,
    min_score: float,
    touching: Sequence[str] | None = None,
) -> list[tuple[str, str, float]]:
    """Stored edges at or above ``min_score``; with ``touching``, only those with an endpoint in it."""
    base = and_(
        _same_project(NearDupEdge.project_id, project_id),
        NearDupEdge.level == level,
        NearDupEdge.space == space,
        NearDupEdge.score >= min_score,
    )
    if touching is None:
        rows = await session.execute(
            select(NearDupEdge.a_id, NearDupEdge.b_id, NearDupEdge.score).where(base)
        )
        return [(r.a_id, r.b_id, float(r.score)) for r in rows]
    found: dict[tuple[str, str], float] = {}
    for start in range(0, len(touching), EDGE_CHUNK):
        ids = touching[start : start + EDGE_CHUNK]
        rows = await session.execute(
            select(NearDupEdge.a_id, NearDupEdge.b_id, NearDupEdge.score).where(
                base, or_(NearDupEdge.a_id.in_(ids), NearDupEdge.b_id.in_(ids))
            )
        )
        for r in rows:
            found[(r.a_id, r.b_id)] = float(r.score)
    return [(a, b, s) for (a, b), s in found.items()]


async def load_decisions(
    session: AsyncSession, project_id: uuid.UUID, level: str
) -> dict[tuple[str, str], str]:
    rows = await session.execute(
        select(NearDupDecision.a_id, NearDupDecision.b_id, NearDupDecision.decision).where(
            NearDupDecision.project_id == project_id, NearDupDecision.level == level
        )
    )
    return {(r.a_id, r.b_id): r.decision for r in rows}


@dataclass(frozen=True)
class RunRow:
    id: uuid.UUID
    project_id: uuid.UUID | None
    level: str
    mode: str
    status: str
    space: str
    profile_config_id: uuid.UUID | None
    profile_revision: int | None
    params: dict[str, Any]
    members_scored: int
    dedup_groups: int
    leak_groups: int
    grouped_members: int
    largest_group: int
    oversized_split: int
    holdout_absorbed: int
    algorithm_version: str


def _run_row(r: NearDupRun) -> RunRow:
    return RunRow(
        r.id, r.project_id, r.level, r.mode, r.status, r.space, r.profile_config_id,
        r.profile_revision, dict(r.params), r.members_scored, r.dedup_groups, r.leak_groups,
        r.grouped_members, r.largest_group, r.oversized_split, r.holdout_absorbed,
        r.algorithm_version,
    )  # fmt: skip


async def current_run(
    session: AsyncSession, project_id: uuid.UUID | None, level: str
) -> RunRow | None:
    row = await session.scalar(
        select(NearDupRun).where(
            _same_project(NearDupRun.project_id, project_id),
            NearDupRun.level == level,
            NearDupRun.status == 'current',
        )
    )
    return None if row is None else _run_row(row)


async def get_run(session: AsyncSession, run_id: uuid.UUID) -> RunRow | None:
    row = await session.get(NearDupRun, run_id)
    return None if row is None else _run_row(row)


async def start_run(
    session: AsyncSession,
    *,
    project_id: uuid.UUID | None,
    level: str,
    mode: str,
    job_id: uuid.UUID | None,
    parent_run_id: uuid.UUID | None,
    profile: tuple[uuid.UUID, int] | None,
    params: Mapping[str, Any],
    space: str,
    algorithm_version: str,
) -> uuid.UUID:
    """Insert the one ``building`` run of (project, level); a second one is a unique violation."""
    run = NearDupRun(
        project_id=project_id,
        level=level,
        mode=mode,
        status='building',
        job_id=job_id,
        parent_run_id=parent_run_id,
        profile_config_id=None if profile is None else profile[0],
        profile_revision=None if profile is None else profile[1],
        params=dict(params),
        space=space,
        algorithm_version=algorithm_version,
    )
    session.add(run)
    await session.flush()
    return run.id


async def discard_building(session: AsyncSession, project_id: uuid.UUID | None, level: str) -> int:
    """Delete the unfinished run a dead attempt left; the unique index allows one per level."""
    result = await session.execute(
        delete(NearDupRun).where(
            _same_project(NearDupRun.project_id, project_id),
            NearDupRun.level == level,
            NearDupRun.status == 'building',
        )
    )
    return result.rowcount  # type: ignore[attr-defined,no-any-return]


async def new_ids(session: AsyncSession, count: int) -> list[uuid.UUID]:
    rows = await session.execute(text('SELECT uuidv7() FROM generate_series(1, :n)'), {'n': count})
    return [r[0] for r in rows]


@dataclass(frozen=True)
class GroupToWrite:
    lineage_id: uuid.UUID
    representative_id: str
    members: Sequence[str]
    cosine_to_rep: Mapping[str, float]
    min_edge_cosine: float
    min_cosine_to_rep: float


async def write_groups(
    session: AsyncSession,
    run_id: uuid.UUID,
    purpose: str,
    groups: Sequence[GroupToWrite],
) -> None:
    """Insert groups and their members. ``project_id`` is copied from the run by the trigger."""
    ids = await new_ids(session, len(groups))
    for start in range(0, len(groups), GROUP_CHUNK):
        chunk = list(
            zip(ids[start : start + GROUP_CHUNK], groups[start : start + GROUP_CHUNK], strict=True)
        )
        await session.execute(
            pg_insert(NearDupGroup).values(
                [
                    {
                        'id': gid,
                        'run_id': run_id,
                        'purpose': purpose,
                        'lineage_id': g.lineage_id,
                        'representative_id': g.representative_id,
                        'member_count': len(g.members),
                        'min_edge_cosine': g.min_edge_cosine,
                        'min_cosine_to_rep': g.min_cosine_to_rep,
                    }
                    for gid, g in chunk
                ]
            )
        )
    rows = [
        {
            'run_id': run_id,
            'purpose': purpose,
            'member_id': member,
            'group_id': gid,
            'is_representative': member == g.representative_id,
            'cosine_to_rep': g.cosine_to_rep[member],
        }
        for gid, g in zip(ids, groups, strict=True)
        for member in g.members
    ]
    for start in range(0, len(rows), MEMBER_CHUNK):
        await session.execute(pg_insert(NearDupMember).values(rows[start : start + MEMBER_CHUNK]))


@dataclass(frozen=True)
class StoredGroup:
    lineage_id: uuid.UUID
    representative_id: str
    members: tuple[str, ...]
    cosine_to_rep: Mapping[str, float]
    min_edge_cosine: float
    min_cosine_to_rep: float


async def load_groups(session: AsyncSession, run_id: uuid.UUID, purpose: str) -> list[StoredGroup]:
    groups = (
        await session.execute(
            select(NearDupGroup).where(
                NearDupGroup.run_id == run_id, NearDupGroup.purpose == purpose
            )
        )
    ).scalars()
    by_id = {g.id: g for g in groups}
    members: dict[uuid.UUID, dict[str, float]] = {gid: {} for gid in by_id}
    rows = await session.execute(
        select(NearDupMember.group_id, NearDupMember.member_id, NearDupMember.cosine_to_rep).where(
            NearDupMember.run_id == run_id, NearDupMember.purpose == purpose
        )
    )
    for r in rows:
        members[r.group_id][r.member_id] = float(r.cosine_to_rep)
    return [
        StoredGroup(
            g.lineage_id, g.representative_id, tuple(sorted(members[gid])), members[gid],
            float(g.min_edge_cosine), float(g.min_cosine_to_rep),
        )
        for gid, g in by_id.items()
    ]  # fmt: skip


async def finish_run(
    session: AsyncSession, run_id: uuid.UUID, counters: Mapping[str, int], *, status: str
) -> None:
    await session.execute(
        update(NearDupRun)
        .where(NearDupRun.id == run_id)
        .values(status=status, finished_at=func.now(), **dict(counters))
    )


async def flip_current(session: AsyncSession, run: RunRow, counters: Mapping[str, int]) -> None:
    """Make ``run`` the one ``current`` run of its (project, level) and queue its projection.

    The partial unique index makes two current runs impossible; the old one is superseded first
    inside the same transaction. Core runs have no project and no projection task.
    """
    await session.execute(
        update(NearDupRun)
        .where(
            _same_project(NearDupRun.project_id, run.project_id),
            NearDupRun.level == run.level,
            NearDupRun.status == 'current',
        )
        .values(status='superseded')
    )
    await finish_run(session, run.id, counters, status='current')
    if run.project_id is not None:
        session.add(
            SearchProjectionTask(
                project_id=run.project_id,
                kind='near_dup_run',
                payload={'run_id': str(run.id), 'level': run.level},
            )
        )
    await session.flush()


async def prune_runs(
    session: AsyncSession, project_id: uuid.UUID | None, level: str, keep_superseded: int
) -> int:
    """Delete superseded runs beyond the newest ``keep_superseded`` (cascade: groups, members)."""
    old = (
        select(NearDupRun.id)
        .where(
            _same_project(NearDupRun.project_id, project_id),
            NearDupRun.level == level,
            NearDupRun.status.in_(('superseded', 'failed')),
        )
        .order_by(NearDupRun.started_at.desc())
        .offset(keep_superseded)
    )
    result = await session.execute(delete(NearDupRun).where(NearDupRun.id.in_(old)))
    return result.rowcount  # type: ignore[attr-defined,no-any-return]


@dataclass(frozen=True)
class MemberState:
    """What a member shows on the index: its group lineage, role and group size per purpose."""

    lineage_id: uuid.UUID
    is_representative: bool
    group_size: int


async def member_states(
    session: AsyncSession, run_id: uuid.UUID
) -> dict[tuple[str, str], MemberState]:
    """``(purpose, member_id) -> state`` for every grouped member of one run."""
    rows = await session.execute(
        select(
            NearDupMember.purpose,
            NearDupMember.member_id,
            NearDupGroup.lineage_id,
            NearDupMember.is_representative,
            NearDupGroup.member_count,
        )
        .join(NearDupGroup, NearDupGroup.id == NearDupMember.group_id)
        .where(NearDupMember.run_id == run_id)
    )
    return {
        (r.purpose, r.member_id): MemberState(r.lineage_id, r.is_representative, r.member_count)
        for r in rows
    }


async def mark_projection_done(session: AsyncSession, task_id: uuid.UUID) -> None:
    await session.execute(
        update(SearchProjectionTask)
        .where(SearchProjectionTask.id == task_id, SearchProjectionTask.done_at.is_(None))
        .values(done_at=func.now())
    )


async def parent_member_states(
    session: AsyncSession, run_id: uuid.UUID, mode: str
) -> dict[tuple[str, str], MemberState] | None:
    """The parent run's member states for an incremental run; ``None`` for a full run, whose
    projection rewrites everything and removes what is stale."""
    if mode != 'incremental':
        return None
    parent_id = await session.scalar(
        select(NearDupRun.parent_run_id).where(NearDupRun.id == run_id)
    )
    return None if parent_id is None else await member_states(session, parent_id)
