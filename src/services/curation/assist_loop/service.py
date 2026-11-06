"""Start, read, control and undo the assist loops of the bound project.

A loop is a ``jobs`` row of kind ``assist_loop`` (exclusive per project) plus its run in the
``assist_loop_*`` tables; the job carries status, pause and cancel, the tables carry the loop.
The loop id on the wire is the run id. Cancel, pause and resume are the job service's:
cancel is terminal, pause is cooperative, and a resume answers 409 ``resume_blocked`` with the
reason while the project is paused or the VLM is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import TYPE_CHECKING, Any

from src.config.project_context import current_project, current_project_id
from src.core.exceptions import RequestRejectedError
from src.db import session as db
from src.db.repos import assist_loop as budget_repo, assist_loop_record as rows
from src.schemas.curation.assist_loop import (
    AssistLoopBudgetEdit,
    AssistLoopList,
    AssistLoopState,
    LoopMembersPage,
    LoopNodePage,
    LoopPackPin,
    LoopPinned,
    LoopPlan,
    LoopRollbackRequest,
    LoopRollbackResponse,
    LoopVlmPin,
)
from src.services.curation.assist_loop import views
from src.services.curation.assist_loop.budget import BudgetLimits
from src.services.curation.assist_loop.plan import build_plan
from src.services.curation.assist_loop.undo import undo_loop
from src.services.jobs import service as jobs, status as job_status
from src.services.rejections import rejection


if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import Session

    from src.services.curation.assist_loop.prepare import PreparedLoop
    from src.services.curation.autolabel.run_vlm import RunVlm

KIND = 'assist_loop'


def _pin(vlm: RunVlm | None) -> LoopVlmPin | None:
    if vlm is None or vlm.endpoint is None:
        return None
    return LoopVlmPin(name=vlm.name, revision=vlm.revision, model=vlm.endpoint.model_id)


def _pinned(prepared: PreparedLoop) -> LoopPinned:
    primary = _pin(prepared.vlm)
    assert primary is not None  # require_run_vlm refused a project with its VLM off
    return LoopPinned(
        vlm=primary,
        prompt_pack=LoopPackPin(name=prepared.pack.name, revision=prepared.pack_revision),
        audit_vlm=_pin(prepared.audit_vlm),
    )


def plan_of(prepared: PreparedLoop) -> LoopPlan:
    return build_plan(prepared.request, prepared.roots, images_per_call=prepared.images_per_call)


def dry_run_state(prepared: PreparedLoop) -> AssistLoopState:
    return AssistLoopState(
        status='dry_run',
        request=prepared.request,
        pinned=_pinned(prepared),
        plan=plan_of(prepared),
    )


def limits_of(prepared: PreparedLoop) -> BudgetLimits:
    caps = prepared.request.budget
    return BudgetLimits.from_request(
        max_vlm_crops=caps.max_vlm_crops if caps else None,
        max_vlm_requests=caps.max_vlm_requests if caps else None,
        max_wall_s=caps.max_wall_s if caps else None,
        images_per_call=prepared.images_per_call,
        vlm_concurrency=prepared.request.vlm_concurrency,
        audit_max_crops=prepared.request.audit.max_crops,
    )


async def _read[T](session: AsyncSession, fn: Callable[[Session], T]) -> T:
    async with db.project_tx(session, current_project_id()):
        return await session.run_sync(fn)


async def run_of(session: AsyncSession, loop_id: str) -> rows.RunRow:
    """The run named by ``loop_id`` in the bound project (404 otherwise)."""
    run_id = views.parse_loop_id(loop_id)
    run = await _read(session, lambda s: rows.run_by_id(s, run_id))
    if run is None or run.project_id != current_project_id():
        raise rejection(404, 'not_found', f'no auto-label loop {loop_id}')
    return run


async def state_of(session: AsyncSession, run: rows.RunRow) -> AssistLoopState:
    job = await job_status.get_job(session, run.job_id, project=current_project_id())
    return await _read(session, lambda s: views.loop_state(s, run, job))


async def get_state(session: AsyncSession, loop_id: str) -> AssistLoopState:
    return await state_of(session, await run_of(session, loop_id))


async def list_states(session: AsyncSession, *, active_only: bool, limit: int) -> AssistLoopList:
    found = await _read(session, lambda s: rows.list_runs(s, current_project_id(), limit=limit))
    states = [await state_of(session, run) for run in found]
    if active_only:
        states = [s for s in states if s.control is not None and s.control.state in _ACTIVE_STATES]
    return AssistLoopList(loops=states)


_ACTIVE_STATES = frozenset({'queued', 'running', 'pausing'})


async def start_loop(
    session: AsyncSession, prepared: PreparedLoop, *, idempotency_key: str | None = None
) -> AssistLoopState:
    """Queue the loop (409 ``assist_loop_busy`` while another is live); its state."""
    project = current_project_id()
    request = prepared.request.model_dump(mode='json')
    pinned = {
        **_pinned(prepared).model_dump(mode='json'),
        'plan': plan_of(prepared).model_dump(mode='json'),
    }
    seed_limits = limits_of(prepared).seed()
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()

    async def create_rows(sess: AsyncSession, job: Any) -> None:
        await sess.run_sync(
            lambda s: rows.create_run(
                s,
                project_id=job.project_id,
                job_id=job.id,
                request=request,
                pinned=pinned,
                seed=int(request['seed']),
                created_by='api',
                limits=seed_limits,
            )
        )

    try:
        result = await jobs.start_job(
            session,
            KIND,
            project,
            params={'request_sha256': digest},
            idempotency_key=idempotency_key,
            requested_by='api',
            with_rows=create_rows,
        )
    except RequestRejectedError as refusal:
        detail = refusal.detail if isinstance(refusal.detail, dict) else {}
        if detail.get('error') != 'job_busy':
            raise
        active = detail.get('active_job_id')
        run = (
            await _read(session, lambda s: rows.run_by_job(s, uuid.UUID(active)))
            if active
            else None
        )
        raise rejection(
            409,
            'assist_loop_busy',
            'an auto-label loop is already running for this project; cancel it or wait',
            loop_id=None if run is None else str(run.id),
        ) from None
    run = await _read(session, lambda s: rows.run_by_job(s, result.job.id))
    assert run is not None
    return await state_of(session, run)


async def cancel(session: AsyncSession, loop_id: str) -> AssistLoopState:
    run = await run_of(session, loop_id)
    await jobs.cancel_job(session, run.job_id, current_project_id(), requested_by='api')
    return await state_of(session, run)


async def pause(session: AsyncSession, loop_id: str) -> AssistLoopState:
    run = await run_of(session, loop_id)
    await jobs.pause_job(session, run.job_id, current_project_id(), reason='user')
    return await state_of(session, run)


async def resume(session: AsyncSession, loop_id: str) -> AssistLoopState:
    run = await run_of(session, loop_id)
    await jobs.resume_job(session, run.job_id, current_project_id())
    return await state_of(session, run)


async def edit_budget(
    session: AsyncSession, loop_id: str, edit: AssistLoopBudgetEdit
) -> AssistLoopState:
    """Set limits; allowed while the loop is queued, running or paused. A limit under what is
    spent plus reserved is 409 ``budget_below_spent``. Never resumes the loop."""
    run = await run_of(session, loop_id)
    state = await state_of(session, run)
    if state.control is None or not state.control.can_edit_budget:
        raise rejection(409, 'budget_not_editable', f'a {state.status} loop has no budget to edit')

    def write(s: Any) -> None:
        for item in edit.limits:
            try:
                budget_repo.edit_limit(
                    s, run_id=run.id, pool=item.pool, unit=item.unit, limit_value=item.limit_value
                )
            except budget_repo.BudgetBelowSpentError as exc:
                raise rejection(
                    409,
                    'budget_below_spent',
                    str(exc),
                    pool=exc.pool,
                    unit=exc.unit,
                    floor=exc.floor,
                ) from None
            except ValueError as exc:
                raise rejection(422, 'invalid_request', str(exc)) from None

    await _read(session, write)
    return await state_of(session, run)


async def node_page(
    session: AsyncSession,
    loop_id: str,
    *,
    level: int | None,
    decision: str | None,
    after: str | None,
    limit: int,
) -> LoopNodePage:
    run = await run_of(session, loop_id)
    return await _read(
        session,
        lambda s: views.node_page(
            s, run.id, level=level, decision=decision, after=after, limit=limit
        ),
    )


async def member_page(
    session: AsyncSession, loop_id: str, path: str, *, after: str | None, limit: int
) -> LoopMembersPage:
    run = await run_of(session, loop_id)
    found = await _read(
        session, lambda s: next((n for n in rows.nodes(s, run.id) if n.path == path), None)
    )
    return await views.member_page(
        found, run.id, current_project().record.slug, path, after=after, limit=limit
    )


async def rollback(
    session: AsyncSession, opensearch: Any, loop_id: str, request: LoopRollbackRequest
) -> LoopRollbackResponse:
    """The label-operation undo over the loop's propagations (409 ``assist_loop_running``
    while the loop is live: cancel or pause it first)."""
    run = await run_of(session, loop_id)
    state = await state_of(session, run)
    if state.control is None or not state.control.can_rollback:
        raise rejection(
            409,
            'assist_loop_running',
            f'auto-label loop {loop_id} is {state.status}; stop it before rolling it back',
            loop_id=loop_id,
        )
    links = await _read(session, lambda s: rows.node_ops(s, run.id))
    result = await undo_loop(
        opensearch,
        loop_id=loop_id,
        links=links,
        node_paths=request.node_paths,
        dry_run=request.dry_run,
    )
    return LoopRollbackResponse(
        dry_run=result.dry_run,
        matched=result.matched,
        restored=result.restored,
        kept_changed=result.kept_changed,
        conflicts=result.conflicts,
    )


async def require_no_live_loop(session: AsyncSession) -> None:
    """The refresh hold: nothing may retrain the clusters under a live loop (409
    ``assist_loop_running``)."""
    live = await _live_loop_id(session)
    if live is not None:
        raise rejection(
            409,
            'assist_loop_running',
            f'auto-label loop {live} is running; cancel it before retraining the clusters',
            loop_id=live,
        )


async def _live_loop_id(session: AsyncSession) -> str | None:
    """The id of the bound project's loop that is queued, running or paused, or ``None``."""
    page = await job_status.list_jobs(
        session,
        project=current_project_id(),
        kinds=(KIND,),
        statuses=('queued', 'running', 'paused', 'cancelling'),
        limit=1,
    )
    if not page.items:
        return None
    run = await _read(session, lambda s: rows.run_by_job(s, uuid.UUID(page.items[0].id)))
    return None if run is None else str(run.id)


__all__ = [
    'KIND',
    'cancel',
    'dry_run_state',
    'edit_budget',
    'get_state',
    'limits_of',
    'list_states',
    'member_page',
    'node_page',
    'pause',
    'plan_of',
    'require_no_live_loop',
    'resume',
    'rollback',
    'run_of',
    'start_loop',
    'state_of',
]
