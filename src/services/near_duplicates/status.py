"""Reads and small writes behind the near-duplicate routes (status, runs, groups, decisions).

Everything runs in the caller's project scope; a run, a group or a decision of another project
is simply not found. The status tells Cropwright what to do next: ``no_run`` (start a full
build), ``pending`` (a rescore is due) or ``rebuild_required`` (the active profile or the primary
embedding space is not the one the current run grouped).
"""

from __future__ import annotations

import base64
import binascii
import uuid
from typing import TYPE_CHECKING

from src.db.repos import near_dup as repo
from src.db.session import project_tx
from src.domain.embedding_spaces import Target
from src.schemas.curation import near_duplicates as wire
from src.services.embedding_spaces.resolve import embedding_space
from src.services.jobs.service import rejection, start_job
from src.services.near_duplicates import pending
from src.services.near_duplicates.profiles import NoActiveProfileError, active_profile


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

LEVELS: tuple[wire.Level, ...] = ('image', 'item')
_TARGETS = {'image': Target.FRAME, 'item': Target.ITEM}
DEFAULT_PAGE = 100
MAX_PAGE = 500
KIND_FOR_MODE = {'full': 'near_dup_score', 'incremental': 'near_dup_rescore'}


def run_view(run: repo.RunRow) -> wire.NearDupRunView:
    return wire.NearDupRunView(
        id=str(run.id), level=run.level, mode=run.mode, status=run.status, space=run.space,
        algorithm_version=run.algorithm_version, profile_revision=run.profile_revision,
        params=run.params, members_scored=run.members_scored, dedup_groups=run.dedup_groups,
        leak_groups=run.leak_groups, grouped_members=run.grouped_members,
        largest_group=run.largest_group, oversized_split=run.oversized_split,
        holdout_absorbed=run.holdout_absorbed, started_at=run.started_at.isoformat(),
        finished_at=None if run.finished_at is None else run.finished_at.isoformat(),
    )  # fmt: skip


async def _level_status(
    session: AsyncSession, project: uuid.UUID, level: wire.Level, profile_revision: int | None
) -> wire.NearDupLevelStatus:
    current = await repo.current_run(session, project, level)
    count = await repo.pending_count(session, project, level)
    reasons: list[wire.StaleReason] = []
    if current is None:
        reasons.append('no_run')
    else:
        space = await embedding_space(_TARGETS[level])
        if current.profile_revision != profile_revision or current.space != space.space_key:
            reasons.append('rebuild_required')
    if count:
        reasons.append('pending')
    return wire.NearDupLevelStatus(
        level=level,
        current_run=None if current is None else run_view(current),
        pending_count=count,
        stale_reasons=reasons,
    )


async def status(session: AsyncSession, project: uuid.UUID) -> wire.NearDupStatus:
    async with project_tx(session, project):
        try:
            profile = await active_profile(session, project)
            revision, active = (
                profile.revision,
                wire.NearDupActiveProfile(name='default', revision=profile.revision),
            )
        except NoActiveProfileError:
            revision, active = None, None
        levels = [await _level_status(session, project, level, revision) for level in LEVELS]
    return wire.NearDupStatus(profile=active, levels=levels)


async def start_run(
    session: AsyncSession, project: uuid.UUID, body: wire.NearDupRunRequest, requested_by: str
) -> wire.NearDupRunStarted:
    kind = KIND_FOR_MODE[body.mode]
    result = await start_job(
        session, kind, project, params={'level': body.level}, requested_by=requested_by
    )
    return wire.NearDupRunStarted(job_id=str(result.job.id), kind=kind)


async def runs(
    session: AsyncSession, project: uuid.UUID, level: str | None, limit: int
) -> wire.NearDupRunPage:
    async with project_tx(session, project):
        rows = await repo.list_runs(session, project, level, limit)
        return wire.NearDupRunPage(items=[run_view(r) for r in rows])


async def run(session: AsyncSession, project: uuid.UUID, run_id: uuid.UUID) -> wire.NearDupRunView:
    async with project_tx(session, project):
        found = await repo.get_run(session, run_id)
        if found is None or found.project_id != project:
            raise rejection(404, 'run_not_found', f'no near-duplicate run {run_id}')  # type: ignore[arg-type]
        return run_view(found)


def _summary_view(row: repo.GroupSummaryRow) -> wire.NearDupGroupSummary:
    return wire.NearDupGroupSummary(
        group_id=str(row.lineage_id), purpose=row.purpose,
        size=row.member_count, representative_id=row.representative_id,
        min_edge_cosine=row.min_edge_cosine, min_cosine_to_rep=row.min_cosine_to_rep,
        has_holdout=row.has_holdout,
    )  # fmt: skip


def _decode_cursor(cursor: str | None) -> tuple[int, uuid.UUID] | None:
    if cursor is None:
        return None
    try:
        size, lineage = base64.urlsafe_b64decode(cursor.encode()).decode().split(':')
        return int(size), uuid.UUID(lineage)
    except (ValueError, binascii.Error):
        raise rejection(422, 'cursor_invalid', 'the cursor is not one this route issued') from None  # type: ignore[arg-type]


def _encode_cursor(row: repo.GroupSummaryRow) -> str:
    return base64.urlsafe_b64encode(f'{row.member_count}:{row.lineage_id}'.encode()).decode()


async def groups(
    session: AsyncSession,
    project: uuid.UUID,
    *,
    level: wire.Level,
    purpose: wire.Purpose,
    min_size: int,
    has_holdout: bool | None,
    cursor: str | None,
    limit: int,
) -> wire.NearDupGroupPage:
    after = _decode_cursor(cursor)
    async with project_tx(session, project):
        current = await repo.current_run(session, project, level)
        if current is None:
            return wire.NearDupGroupPage(items=[], next_cursor=None)
        rows = await repo.page_groups(
            session, current.id, purpose=purpose, min_size=min_size, has_holdout=has_holdout,
            after=after, limit=limit + 1,
        )  # fmt: skip
    page = rows[:limit]
    more = len(rows) > limit
    return wire.NearDupGroupPage(
        items=[_summary_view(r) for r in page],
        next_cursor=_encode_cursor(page[-1]) if more else None,
    )


async def group(
    session: AsyncSession, project: uuid.UUID, level: wire.Level, group_id: uuid.UUID
) -> wire.NearDupGroupDetail:
    async with project_tx(session, project):
        current = await repo.current_run(session, project, level)
        found = None
        for purpose in ('dedup', 'leak'):
            if current is not None and found is None:
                found = await repo.group_with_members(session, current.id, group_id, purpose)
    if found is None:
        raise rejection(404, 'group_not_found', f'no current {level} group {group_id}')  # type: ignore[arg-type]
    summary, members = found
    return wire.NearDupGroupDetail(
        group=_summary_view(summary),
        members=[
            wire.NearDupMemberView(member_id=m, is_representative=rep, cosine_to_rep=c)
            for m, rep, c in members
        ],
    )


def _decision_view(row: repo.DecisionRow) -> wire.NearDupDecisionView:
    return wire.NearDupDecisionView(
        id=str(row.id), level=row.level, a_id=row.a_id, b_id=row.b_id,
        decision=row.decision, decided_by=row.decided_by, decided_at=row.decided_at.isoformat(),
    )  # fmt: skip


async def decide(
    session: AsyncSession, project: uuid.UUID, body: wire.NearDupDecisionIn, decided_by: str
) -> wire.NearDupDecisionView:
    """Store the decision and mark both members pending, so the next rescore applies it."""
    async with project_tx(session, project):
        row = await repo.upsert_decision(
            session, project, level=body.level, a_id=body.a_id, b_id=body.b_id,
            decision=body.decision, decided_by=decided_by,
        )  # fmt: skip
        await pending.enqueue_pending(
            session, project, body.level, [body.a_id, body.b_id], 'vector_changed'
        )
    return _decision_view(row)


async def undecide(session: AsyncSession, project: uuid.UUID, decision_id: uuid.UUID) -> None:
    async with project_tx(session, project):
        row = await repo.delete_decision(session, project, decision_id)
        if row is None:
            raise rejection(404, 'decision_not_found', f'no decision {decision_id}')  # type: ignore[arg-type]
        await pending.enqueue_pending(
            session, project, row.level, [row.a_id, row.b_id], 'vector_changed'
        )
