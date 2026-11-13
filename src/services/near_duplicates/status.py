"""Reads and small writes behind the near-duplicate routes (status, runs, groups, decisions).

Everything runs in the caller's project scope; a run, a group or a decision of another project
is simply not found. The status tells Cropwright what to do next: ``no_run`` (start a full
build), ``pending`` (a rescore is due) or ``rebuild_required`` (the active profile or the primary
embedding space is not the one the current run grouped).
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import uuid
from typing import TYPE_CHECKING, Any

from src.config.project_context import current_project
from src.db.repos import near_dup as repo
from src.db.session import project_tx
from src.domain.embedding_spaces import Target
from src.schemas.curation import near_duplicates as wire
from src.services.embedding_spaces.resolve import embedding_space
from src.services.jobs.service import rejection, start_job
from src.services.near_duplicates import member_views, pending
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
    if body.level == 'item':
        async with project_tx(session, project):
            has_image_run = await repo.current_run(session, project, 'image') is not None
        if not has_image_run:
            raise rejection(
                409,
                'no_image_run',
                "the item level groups inside the image level's leak groups: "
                'run the image level first',
            )
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


def _summary_view(level: wire.Level, row: repo.GroupSummaryRow) -> wire.NearDupGroupSummary:
    return wire.NearDupGroupSummary(
        representative_thumbnail_url=member_views.thumbnail_url(level, row.representative_id),
        group_id=str(row.lineage_id), purpose=row.purpose,
        size=row.member_count, representative_id=row.representative_id,
        min_edge_cosine=row.min_edge_cosine, min_cosine_to_rep=row.min_cosine_to_rep,
        has_holdout=row.has_holdout,
    )  # fmt: skip


def _parse_cursor(cursor: str) -> list[str]:
    try:
        return base64.urlsafe_b64decode(cursor.encode()).decode().split('|')
    except (ValueError, binascii.Error):
        raise rejection(422, 'cursor_invalid', 'the cursor is not one this route issued') from None  # type: ignore[arg-type]


def _make_cursor(*parts: object) -> str:
    return base64.urlsafe_b64encode('|'.join(map(str, parts)).encode()).decode()


def _decode_cursor(cursor: str | None) -> tuple[int, uuid.UUID] | None:
    if cursor is None:
        return None
    try:
        size, lineage = _parse_cursor(cursor)
        return int(size), uuid.UUID(lineage)
    except ValueError:
        raise rejection(422, 'cursor_invalid', 'the cursor is not one this route issued') from None  # type: ignore[arg-type]


def _encode_cursor(row: repo.GroupSummaryRow) -> str:
    return _make_cursor(row.member_count, row.lineage_id)


async def groups(
    session: AsyncSession,
    project: uuid.UUID,
    opensearch: Any,
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
            return wire.NearDupGroupPage(
                items=[],
                next_cursor=None,
                empty_reason=f'no {level}-level run yet: build it to find near-duplicates',
            )
        rows = await repo.page_groups(
            session, current.id, purpose=purpose, min_size=min_size, has_holdout=has_holdout,
            after=after, limit=limit + 1,
        )  # fmt: skip
    page = rows[:limit]
    more = len(rows) > limit
    return wire.NearDupGroupPage(
        items=[_summary_view(level, r) for r in page],
        next_cursor=_encode_cursor(page[-1]) if more else None,
        empty_reason=None
        if rows or cursor
        else empty_reason_text(
            current, level, purpose, await _without_vector(opensearch, session, project, level)
        ),
    )


async def _without_vector(
    opensearch: Any, session: AsyncSession, project: uuid.UUID, level: wire.Level
) -> int:
    """How many documents of the level's index hold no vector of the level's space: a run can
    only compare the others."""
    async with project_tx(session, project):
        space = await embedding_space(_TARGETS[level])
    index = current_project().record.resources.indexes[space.index_role]  # type: ignore[index]
    body = {'query': {'bool': {'must_not': [{'exists': {'field': space.vector_field}}]}}}
    return int((await opensearch.count(index=index, body=body))['count'])


def empty_reason_text(
    run: repo.RunRow, level: wire.Level, purpose: wire.Purpose, without_vector: int
) -> str:
    """Why a run's page is empty: the cut and how many members were compared, plus the members
    that could not be (no vector), which is the usual reason a run compared fewer than exist."""
    cut = run.params.get('dedup_cosine' if purpose == 'dedup' else 'leak_cosine')
    noun = 'images' if level == 'image' else 'items'
    text = f'0 {purpose} groups at cosine {cut} among {run.members_scored} {noun}'
    if without_vector:
        text += (
            f'; {without_vector} {noun} have no vector in this space and were not compared '
            '(Reprocess with scope embed fills them)'
        )
    return text


async def group(
    session: AsyncSession,
    project: uuid.UUID,
    opensearch: Any,
    level: wire.Level,
    group_id: uuid.UUID,
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
        decisions = await repo.decisions_with(session, project, level, summary.representative_id)
    facts = await member_views.item_facts(opensearch, level, [m for m, _rep, _c in members])
    pins = member_views.pins_to(summary.representative_id, decisions)
    return wire.NearDupGroupDetail(
        group=_summary_view(level, summary),
        members=[member_views.member_view(level, m, facts, pins) for m in members],
    )


def _decision_view(row: repo.DecisionRow) -> wire.NearDupDecisionView:
    return wire.NearDupDecisionView(
        id=str(row.id), level=row.level, a_id=row.a_id, b_id=row.b_id,
        decision=row.decision, decided_by=row.decided_by, decided_at=row.decided_at.isoformat(),
        a_thumbnail_url=member_views.thumbnail_url(row.level, row.a_id),  # type: ignore[arg-type]
        b_thumbnail_url=member_views.thumbnail_url(row.level, row.b_id),  # type: ignore[arg-type]
    )  # fmt: skip


def _decode_decision_cursor(cursor: str | None) -> tuple[dt.datetime, uuid.UUID] | None:
    if cursor is None:
        return None
    try:
        when, decision_id = _parse_cursor(cursor)
        return dt.datetime.fromisoformat(when), uuid.UUID(decision_id)
    except ValueError:
        raise rejection(422, 'cursor_invalid', 'the cursor is not one this route issued') from None  # type: ignore[arg-type]


async def decisions(
    session: AsyncSession,
    project: uuid.UUID,
    *,
    level: wire.Level | None,
    member_id: str | None,
    cursor: str | None,
    limit: int,
) -> wire.NearDupDecisionPage:
    """The pinned pairs, newest first."""
    after = _decode_decision_cursor(cursor)
    async with project_tx(session, project):
        rows = await repo.page_decisions(
            session, project, level=level, member_id=member_id, after=after, limit=limit + 1
        )
    page = rows[:limit]
    last = page[-1] if len(rows) > limit else None
    return wire.NearDupDecisionPage(
        items=[_decision_view(r) for r in page],
        next_cursor=None if last is None else _make_cursor(last.decided_at.isoformat(), last.id),
    )


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
