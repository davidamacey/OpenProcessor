"""What the loop routes serve: the run's tables as the wire models, the nodes of its tree and
the member ids of one node."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from src.db.repos import assist_loop as budget_repo, assist_loop_record as rows
from src.domain.assist_loop_control import control_flags
from src.domain.loop_tree import parse_path, path_level
from src.schemas.curation.assist_loop import (
    AssistLoopControl,
    AssistLoopRequest,
    AssistLoopState,
    LoopAuditView,
    LoopBudgetView,
    LoopLevel,
    LoopMembersPage,
    LoopNewClassVote,
    LoopNode,
    LoopNodePage,
    LoopPartition,
    LoopPinned,
    LoopPlan,
    LoopPoolBudget,
    LoopTotals,
    LoopUnitBudget,
    LoopVote,
)
from src.services.curation.assist_loop.snapshots import SnapshotStore
from src.services.jobs.blockers import VLM_UNAVAILABLE, project_is_paused, vlm_available
from src.services.rejections import rejection


if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.orm import Session

    from src.domain.assist_loop_control import ResumeBlocker
    from src.schemas.jobs import JobView

REASON_MESSAGES = {
    'budget_crops': 'the crop budget could not cover it',
    'budget_requests': 'the request budget could not cover it',
    'budget_tokens': 'the token budget could not cover it',
    'budget_time': 'the time budget ran out',
    'no_eligible_members': 'it has no member the loop may label',
    'no_split': 'it could not be split into parts the loop can label',
    'class_not_in_registry': 'the class it agreed on is no longer in the class list',
    'drift': 'its members changed after the loop pinned them',
}
"""What a node's ``reason`` code means, served so the client renders it instead of keeping
its own wording."""


def parse_loop_id(loop_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(loop_id)
    except ValueError:
        raise rejection(404, 'not_found', f'no auto-label loop {loop_id}') from None


def _unit(by_unit: dict[str, budget_repo.BudgetRow], unit: str) -> LoopUnitBudget | None:
    row = by_unit.get(unit)
    if row is None:
        return None
    return LoopUnitBudget(limit=row.limit_value, reserved=row.reserved, spent=row.spent)


def _pool(rows_: list[budget_repo.BudgetRow], pool: str) -> LoopPoolBudget | None:
    by_unit = {r.unit: r for r in rows_ if r.pool == pool}
    crops, requests = by_unit.get('crops'), by_unit.get('requests')
    if crops is None or requests is None:
        return None
    return LoopPoolBudget(
        max_crops=crops.limit_value or 0,
        max_requests=requests.limit_value or 0,
        crops_spent=crops.spent,
        crops_reserved=crops.reserved,
        requests_spent=requests.spent,
        requests_reserved=requests.reserved,
        prompt_tokens=_unit(by_unit, 'prompt_tokens'),
        completion_tokens=_unit(by_unit, 'completion_tokens'),
    )


def _budget_view(rows_: list[budget_repo.BudgetRow], usage_missing: int) -> LoopBudgetView | None:
    vlm = _pool(rows_, 'vlm')
    if vlm is None:
        return None
    wall = next((r for r in rows_ if (r.pool, r.unit) == ('vlm', 'wall_s')), None)
    return LoopBudgetView(
        vlm=vlm,
        audit=_pool(rows_, 'audit'),
        wall_s=float(wall.spent) if wall else 0.0,
        max_wall_s=None if wall is None or wall.limit_value is None else float(wall.limit_value),
        usage_missing=usage_missing,
    )


def _audit_view(
    counts: dict[str, Any], pinned: dict[str, Any], audit_pool: LoopPoolBudget | None
) -> LoopAuditView:
    if not pinned.get('audit_vlm'):
        return LoopAuditView(status='off')
    stored = counts.get('audit') or {}
    sampled, agreed = int(stored.get('total_sampled', 0)), int(stored.get('total_agreed', 0))
    return LoopAuditView(
        status=stored.get('status', 'pending'),
        sampled=sampled,
        agreed=agreed,
        agreement=round(agreed / sampled, 4) if sampled else None,
        lower_bound=stored.get('lower_bound'),
        crops_spent=0 if audit_pool is None else audit_pool.crops_spent + audit_pool.crops_reserved,
    )


def loop_state(session: Session, run: rows.RunRow, job: JobView) -> AssistLoopState:
    """The wire state of one run, from its rows and its job."""
    counts = rows.iteration_counts(session, run.id)
    budget_rows = budget_repo.budgets(session, run.id)
    budget = _budget_view(budget_rows, int(counts.get('usage_missing', 0)))
    blocker: ResumeBlocker | None = None
    if job.status in ('paused', 'interrupted'):
        if project_is_paused(session, run.project_id):
            blocker = 'project_paused'
        elif job.pause_reason == VLM_UNAVAILABLE and not vlm_available(session):
            blocker = 'vlm_unavailable'
    flags = control_flags(job.status, pause_requested=job.pause_requested, blocker=blocker)
    partition = (run.pinned.get('partition') or None) and LoopPartition.model_validate(
        run.pinned['partition']
    )
    return AssistLoopState(
        loop_id=str(run.id),
        status=job.status,
        stop_reason=run.stop_reason,
        pause_reason=job.pause_reason,
        control=AssistLoopControl(**flags.__dict__),
        slices=rows.slices(session, run.job_id),
        created_at=run.created_at.isoformat(),
        started_at=None if job.started_at is None else job.started_at.isoformat(),
        finished_at=None if job.finished_at is None else job.finished_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        error=job.error_message,
        request=AssistLoopRequest.model_validate(run.request),
        pinned=LoopPinned.model_validate(run.pinned),
        plan=LoopPlan.model_validate(run.pinned['plan']),
        partition=partition,
        budget=budget,
        levels=[LoopLevel.model_validate(x) for x in counts.get('levels') or []],
        totals=LoopTotals.model_validate(counts['totals']) if counts.get('totals') else None,
        audit=_audit_view(counts, run.pinned, budget.audit if budget else None),
    )


def _order(path: str) -> tuple[int, int, tuple[int, ...]]:
    cluster, indices = parse_path(path)
    return (path_level(path), cluster, indices)


def _node(row: rows.NodeRow) -> LoopNode:
    stored = row.distribution or {}
    detail = stored.get('detail') or {}
    return LoopNode(
        path=row.path,
        level=row.level,
        parent=row.parent_path,
        n=row.eligible_count,
        sampled=int(detail.get('sampled', row.answered or 0)),
        answered=int(row.answered or 0),
        decision=row.decision,
        reason=detail.get('reason'),
        reason_message=REASON_MESSAGES.get(str(detail.get('reason'))),
        class_name=row.class_name,
        agree=row.agree,
        bound=row.bound,
        distribution=[LoopVote(class_name=c, votes=v) for c, v in stored.get('votes') or []],
        new_class_proposals=[
            LoopNewClassVote(name=c, votes=v) for c, v in stored.get('new_class_proposals') or []
        ],
        items_propagated=row.items_propagated,
    )


def _page[T](
    items: list[T], key: Callable[[T], Any], after: Any, limit: int
) -> tuple[list[T], T | None]:
    """``items`` (in ``key`` order) after the cursor, at most ``limit``, and the last one
    returned when more remain."""
    rest = [x for x in items if after is None or key(x) > after]
    page = rest[:limit]
    return page, (page[-1] if len(rest) > limit else None)


def node_page(
    session: Session,
    run_id: uuid.UUID,
    *,
    level: int | None,
    decision: str | None,
    after: str | None,
    limit: int,
) -> LoopNodePage:
    found = [
        _node(r)
        for r in rows.nodes(session, run_id)
        if (level is None or r.level == level) and (decision is None or r.decision == decision)
    ]
    found.sort(key=lambda n: _order(n.path))
    try:
        cursor = None if after is None else _order(after)
    except ValueError:
        raise rejection(422, 'invalid_request', f'{after!r} is not a node path') from None
    page, last = _page(found, lambda n: _order(n.path), cursor, limit)
    return LoopNodePage(nodes=page, next_after=last.path if last else None)


async def member_page(
    node: rows.NodeRow | None,
    run_id: uuid.UUID,
    slug: str,
    path: str,
    *,
    after: str | None,
    limit: int,
) -> LoopMembersPage:
    """The crop ids pinned in ``node`` (read by the caller), from its blob."""
    if node is None:
        raise rejection(404, 'not_found', f'loop {run_id} has no node {path}')
    store = SnapshotStore(slug, run_id)
    try:
        ids = (
            (await store.read_root(node.snapshot_storage_key)).member_ids
            if node.level == 0
            else await store.read_ids(node.snapshot_storage_key)
        )
    except ValueError as exc:
        raise rejection(404, 'not_found', str(exc)) from exc
    page, last = _page(sorted(ids), lambda i: i, after, limit)
    return LoopMembersPage(crop_ids=page, next_after=last)


__all__ = ['loop_state', 'member_page', 'node_page', 'parse_loop_id']
