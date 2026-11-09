"""Curation router sub-module: auto-label loops.

``POST /assist/loops`` sizes a loop (a dry run, the default) or queues it as a job, which
samples each cluster, asks the VLM, propagates a class to a cluster that passes its
statistical gate and splits one that does not. The routes only adapt HTTP; the rules (the
plan, the pins, one loop per project, the lock rule, rollback) live in
:mod:`src.services.curation.assist_loop`. Nothing here writes a label except ``rollback``.
"""

from typing import Annotated

from fastapi import Header, Query

from src.core.db_session import DbSessionDep
from src.routers.curation._common import OpenSearchDep, router
from src.schemas.curation.assist_loop import (
    AssistLoopBudgetEdit,
    AssistLoopList,
    AssistLoopRequest,
    AssistLoopState,
    LoopDecision,
    LoopMembersPage,
    LoopNodePage,
    LoopRollbackRequest,
    LoopRollbackResponse,
)
from src.services.curation.assist_loop import service
from src.services.curation.assist_loop.prepare import prepare_loop


@router.post('/assist/loops', response_model=AssistLoopState)
async def create_assist_loop(
    body: AssistLoopRequest,
    opensearch: OpenSearchDep,
    session: DbSessionDep,
    idempotency_key: Annotated[str | None, Header()] = None,
) -> AssistLoopState:
    """Size an auto-label loop, and queue it unless ``dry_run`` (the default).

    The response is the plan: the clusters and eligible items in scope, the sample each
    needs, the VLM crops the first draws cost and whether that fits ``budget``, plus the
    endpoints and prompt pack pinned. A start needs ``budget.max_vlm_crops``. Refusals:
    409 ``assist_loop_busy``, ``vlm_not_configured``, ``clustering_running`` (a clustering
    run is live); 422 ``invalid_request``, ``no_class_names``, ``no_roots``,
    ``vlm_external_not_acknowledged`` and the other external-endpoint refusals (the main or
    the audit endpoint). A loop never overwrites a human,
    validated, imported-validated, excluded or holdout item; what it propagates is a machine
    label (``assist_propagated``) that rollback can undo."""
    prepared = await prepare_loop(opensearch, session, body)
    if body.dry_run:
        return service.dry_run_state(prepared)
    return await service.start_loop(session, prepared, idempotency_key=idempotency_key)


@router.get('/assist/loops', response_model=AssistLoopList)
async def list_assist_loops(
    session: DbSessionDep,
    active: Annotated[bool, Query(description='Only queued, running and pausing loops.')] = False,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> AssistLoopList:
    """The newest loops, newest first."""
    return await service.list_states(session, active_only=active, limit=limit)


@router.get('/assist/loops/{loop_id}', response_model=AssistLoopState)
async def get_assist_loop(loop_id: str, session: DbSessionDep) -> AssistLoopState:
    """One loop: its status, budget, per-level record, totals, audit and what may be done
    with it now (`control`). A loop whose worker died reads `interrupted` until resumed."""
    return await service.get_state(session, loop_id)


@router.get('/assist/loops/{loop_id}/nodes', response_model=LoopNodePage)
async def list_assist_loop_nodes(
    loop_id: str,
    session: DbSessionDep,
    level: Annotated[int | None, Query(ge=0)] = None,
    decision: Annotated[LoopDecision | None, Query()] = None,
    after: Annotated[str | None, Query(description='`next_after` of the previous page.')] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> LoopNodePage:
    """The nodes of the loop's tree by level then path, with each node's vote distribution
    and decision (`review_*`, `stale` and `unvisited` nodes are the ones left for review)."""
    return await service.node_page(
        session, loop_id, level=level, decision=decision, after=after, limit=limit
    )


@router.get('/assist/loops/{loop_id}/nodes/{node_path}/members', response_model=LoopMembersPage)
async def list_assist_loop_node_members(
    loop_id: str,
    node_path: str,
    session: DbSessionDep,
    after: Annotated[str | None, Query(description='`next_after` of the previous page.')] = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> LoopMembersPage:
    """The crop ids pinned in a node at the snapshot, in id order: every member, including
    ones the lock rule keeps out of the loop."""
    return await service.member_page(session, loop_id, node_path, after=after, limit=limit)


@router.post('/assist/loops/{loop_id}/cancel', response_model=AssistLoopState)
async def cancel_assist_loop(loop_id: str, session: DbSessionDep) -> AssistLoopState:
    """Stop a live loop after the requests in flight; what it wrote stays (roll it back to
    undo). Cancel is final: a stopped loop that should continue is paused, not cancelled. A
    loop still waiting for a worker is cancelled at once. 409 `job_finished` for a loop that
    already ended."""
    return await service.cancel(session, loop_id)


@router.post('/assist/loops/{loop_id}/pause', response_model=AssistLoopState)
async def pause_assist_loop(loop_id: str, session: DbSessionDep) -> AssistLoopState:
    """Ask the loop to pause at its next node boundary (`control.state` is `pausing` until
    it does); requests in flight finish. A queued loop pauses at once; a paused one stays
    paused. 409 `job_cancelling` or `job_finished` otherwise."""
    return await service.pause(session, loop_id)


@router.post('/assist/loops/{loop_id}/resume', response_model=AssistLoopState)
async def resume_assist_loop(loop_id: str, session: DbSessionDep) -> AssistLoopState:
    """Queue a `paused` or `interrupted` loop again; it continues from its tables. 409
    `resume_blocked` names the reason (`project_paused`, `vlm_unavailable`), `job_busy` while
    another loop is live, `job_finished` for a loop that ended."""
    return await service.resume(session, loop_id)


@router.patch('/assist/loops/{loop_id}/budget', response_model=AssistLoopState)
async def edit_assist_loop_budget(
    loop_id: str, body: AssistLoopBudgetEdit, session: DbSessionDep
) -> AssistLoopState:
    """Set limits of a queued, running or paused loop. A limit under what is already spent
    plus reserved is 409 `budget_below_spent`; the loop is never resumed by an edit."""
    return await service.edit_budget(session, loop_id, body)


@router.post('/assist/loops/{loop_id}/rollback', response_model=LoopRollbackResponse)
async def rollback_assist_loop(
    loop_id: str,
    opensearch: OpenSearchDep,
    session: DbSessionDep,
    body: LoopRollbackRequest | None = None,
) -> LoopRollbackResponse:
    """Undo the loop's propagations (only those of `node_paths` when given): each is a label
    operation, restored only while the item is as the loop left it. A dry run (the default)
    counts. An item changed by anyone since, or locked, validated, excluded or in the holdout,
    is kept and counted in `kept_changed`. 409 `assist_loop_running` while the loop is live:
    cancel or pause it first."""
    return await service.rollback(session, opensearch, loop_id, body or LoopRollbackRequest())
