"""Curation router sub-module: auto-label loops.

``POST /assist/loops`` sizes a loop (a dry run, the default) or queues it for the loop
worker, which samples each cluster, asks the VLM, propagates a class to a cluster that
passes its statistical gate and splits one that does not. The routes only adapt HTTP; the
rules (the plan, the pins, one loop per project, the lock rule, rollback) live in
:mod:`src.services.curation.assist_loop`. Nothing here writes a label except ``rollback``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Query

from src.routers.curation._common import OpenSearchDep, router
from src.schemas.curation.assist_loop import (
    AssistLoopList,
    AssistLoopRequest,
    AssistLoopState,
    LoopDecision,
    LoopMembersPage,
    LoopNodePage,
    LoopRollbackRequest,
    LoopRollbackResponse,
)
from src.services.curation.assist_loop import job, views
from src.services.curation.assist_loop.prepare import prepare_loop
from src.services.curation.assist_loop.rollback import rollback_loop
from src.services.curation.assist_loop.start import dry_run_state, start_loop


def _state(loop_id: str) -> AssistLoopState:
    state = job.read_state(loop_id)
    if state is None:
        views.open_store(loop_id)  # the 404
    assert state is not None
    return views.loop_state(state)


@router.post('/assist/loops', response_model=AssistLoopState)
async def create_assist_loop(body: AssistLoopRequest, opensearch: OpenSearchDep) -> AssistLoopState:
    """Size an auto-label loop, and queue it unless ``dry_run`` (the default).

    The response is the plan: the clusters and eligible items in scope, the sample each
    needs, the VLM crops the first draws cost and whether that fits ``budget``, plus the
    endpoints and prompt pack pinned. A start needs ``budget.max_vlm_crops``. Refusals:
    409 ``assist_loop_busy``, ``vlm_not_configured``, ``clustering_running`` (an auto-label
    job is live); 422 ``invalid_request``, ``no_class_names``, ``no_roots``,
    ``vlm_external_not_acknowledged`` and the other external-endpoint refusals (the main or
    the audit endpoint). A loop never overwrites a human,
    validated, imported-validated, excluded or holdout item; what it propagates is a machine
    label (``assist_propagated``) that rollback can undo."""
    prepared = await prepare_loop(opensearch, body)
    if body.dry_run:
        return dry_run_state(prepared)
    return views.loop_state(start_loop(prepared))


@router.get('/assist/loops', response_model=AssistLoopList)
async def list_assist_loops(
    active: Annotated[bool, Query(description='Only queued and running loops.')] = False,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> AssistLoopList:
    """The newest loops, newest first."""
    states = job.list_states(active_only=active, limit=limit)
    return AssistLoopList(loops=[views.loop_state(s) for s in states])


@router.get('/assist/loops/{loop_id}', response_model=AssistLoopState)
async def get_assist_loop(loop_id: str) -> AssistLoopState:
    """One loop: its status, budget, per-level record, totals and audit. A loop whose worker
    died reads ``interrupted`` until resumed."""
    return _state(loop_id)


@router.get('/assist/loops/{loop_id}/nodes', response_model=LoopNodePage)
async def list_assist_loop_nodes(
    loop_id: str,
    level: Annotated[int | None, Query(ge=0)] = None,
    decision: Annotated[LoopDecision | None, Query()] = None,
    after: Annotated[str | None, Query(description='`next_after` of the previous page.')] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> LoopNodePage:
    """The nodes of the loop's tree by level then path, with each node's vote distribution
    and decision (`review_*`, `stale` and `unvisited` nodes are the ones left for review)."""
    return views.node_page(
        views.open_store(loop_id), level=level, decision=decision, after=after, limit=limit
    )


@router.get('/assist/loops/{loop_id}/nodes/{node_path}/members', response_model=LoopMembersPage)
async def list_assist_loop_node_members(
    loop_id: str,
    node_path: str,
    after: Annotated[str | None, Query(description='`next_after` of the previous page.')] = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> LoopMembersPage:
    """The crop ids pinned in a node at the snapshot, in id order: every member, including
    ones the lock rule keeps out of the loop."""
    return views.member_page(views.open_store(loop_id), node_path, after=after, limit=limit)


@router.post('/assist/loops/{loop_id}/cancel', response_model=AssistLoopState)
async def cancel_assist_loop(loop_id: str) -> AssistLoopState:
    """Stop a live loop after the requests in flight; what it wrote stays (roll it back to
    undo). A loop still waiting for the worker is cancelled at once."""
    _state(loop_id)
    job.cancel_loop(loop_id)
    return _state(loop_id)


@router.post('/assist/loops/{loop_id}/resume', response_model=AssistLoopState)
async def resume_assist_loop(loop_id: str) -> AssistLoopState:
    """Queue an `interrupted`, `paused` or `cancelled` loop again; it continues from its
    event log. 409 `assist_loop_not_resumable` for any other status, `assist_loop_busy` while
    another loop is live."""
    job.resume_loop(loop_id)
    return _state(loop_id)


@router.post('/assist/loops/{loop_id}/rollback', response_model=LoopRollbackResponse)
async def rollback_assist_loop(
    loop_id: str, opensearch: OpenSearchDep, body: LoopRollbackRequest | None = None
) -> LoopRollbackResponse:
    """Restore the items the loop propagated (only those of `node_paths` when given) to the
    class state before it wrote them. A dry run (the default) counts. An item changed by
    anyone since the loop wrote it, or locked, validated, excluded or in the holdout, is kept
    and counted in `kept_changed`. 409 `assist_loop_running` while the loop is live: cancel it
    first."""
    request = body or LoopRollbackRequest()
    store = views.open_store(loop_id)
    job.require_stopped(loop_id)
    result = await rollback_loop(
        opensearch, store, node_paths=request.node_paths, dry_run=request.dry_run
    )
    return LoopRollbackResponse(
        dry_run=result.dry_run,
        matched=result.matched,
        restored=result.restored,
        kept_changed=result.kept_changed,
        conflicts=result.conflicts,
    )
