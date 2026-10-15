"""Curation router sub-module: assist runs.

``POST /assist/runs`` takes one request shape for every scope (a crop, a
selection, a cluster, a sub-cluster, a filter): a dry run by default that counts
what the run would send to the VLM, then a background run that writes the VLM's
class suggestions and pack-declared attributes. The run's result carries a
proposal for the (sub-)cluster; ``apply`` is the only route that turns it into
labels. The routes only adapt HTTP; the rules (scope, sample, lock rule, one run per
project) live in :mod:`src.services.curation.assist`.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Query

from src.routers.curation._api_error import api_error
from src.routers.curation._common import OpenSearchDep, router
from src.services.curation.assist.apply import apply_proposal
from src.services.curation.assist.job import cancel_run, list_runs, read_state
from src.services.curation.assist.models import (
    AssistProgress,
    AssistRunList,
    AssistRunRequest,
    AssistRunState,
)
from src.services.curation.assist.prepare import prepare_run
from src.services.curation.assist.proposal_models import AssistApplyRequest, AssistApplyResponse
from src.services.curation.assist.runner import start_run
from src.services.curation.assist.scope_now import current_members, scope_now


def _state(run_id: str) -> AssistRunState:
    state = read_state(run_id)
    if state is None:
        raise api_error(404, 'not_found', f'no assist run {run_id}')
    return state


async def _with_scope(
    opensearch: Any,
    state: AssistRunState,
    *,
    members: list[tuple[str, dict[str, Any]]] | None = None,
) -> AssistRunState:
    """``state`` as read: its scope against the members it resolved, and the proposal's
    class checked against the registry as it is now (the registry can change)."""
    assert state.run_id is not None
    state.scope_now = await scope_now(opensearch, state.run_id, state.request, members=members)
    result = state.result
    if result is not None and result.proposal is not None and result.proposal.class_name:
        from src.clients.curation_opensearch.registry import get_class_registry

        names = {c.class_name for c in get_class_registry().load().classes}
        fresh = result.proposal.model_copy(
            update={'in_registry': result.proposal.class_name in names}
        )
        state.result = result.model_copy(update={'proposal': fresh})
    return state


@router.post('/assist/runs', response_model=AssistRunState)
async def create_assist_run(body: AssistRunRequest, opensearch: OpenSearchDep) -> AssistRunState:
    """Plan an assist run, and start it unless ``dry_run`` (the default).

    The response says what the target resolved to (members, the sample, what is
    skipped, the endpoint and pack pinned, the VLM calls to expect). Refusals:
    409 ``vlm_not_configured`` (the project's VLM is off and ``vlm`` names none),
    409 ``assist_busy``, 404 ``not_found`` (an empty cluster), 422
    ``invalid_request``, ``assist_too_large``, ``pack_declares_no_attributes``,
    ``no_class_names``. Class suggestions never overwrite a locked, validated or
    holdout item; attributes are stored apart from the labels."""
    prepared = await prepare_run(opensearch, body)
    if body.dry_run:
        return AssistRunState(
            status='dry_run',
            request=body,
            resolved=prepared.resolved(),
            progress=AssistProgress(total=len(prepared.scope.sampled_ids)),
        )
    if not prepared.scope.sampled_ids:
        raise api_error(422, 'invalid_request', 'the target selects no items the VLM may see')
    return await start_run(opensearch, prepared)


@router.get('/assist/runs', response_model=AssistRunList)
async def list_assist_runs(
    opensearch: OpenSearchDep,
    cluster_id: Annotated[int | None, Query()] = None,
    cluster_subid: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> AssistRunList:
    """The newest runs, newest first, whose target or context names that cluster (and
    sub-cluster); all runs without ``cluster_id``. Each carries ``scope_now``, so a
    client can mark a proposal stale after a re-refine. The newest 100 runs are kept."""
    runs = list_runs(cluster_id=cluster_id, cluster_subid=cluster_subid, limit=limit)
    shared: dict[tuple[int | None, str | None], Any] = {}
    out = []
    for run in runs:
        flt = getattr(run.request.target, 'context', None) or run.request.target
        key = (getattr(flt, 'cluster_id', None), getattr(flt, 'cluster_subid', None))
        if key not in shared:
            shared[key] = await current_members(opensearch, run.request)
        out.append(await _with_scope(opensearch, run, members=shared[key]))
    return AssistRunList(runs=out)


@router.get('/assist/runs/{run_id}', response_model=AssistRunState)
async def get_assist_run(run_id: str, opensearch: OpenSearchDep) -> AssistRunState:
    """The state of one run: progress while it runs, ``result`` (with the cluster
    proposal) when it ends, and ``scope_now`` for a run with a cluster scope."""
    return await _with_scope(opensearch, _state(run_id))


@router.post('/assist/runs/{run_id}/cancel', response_model=AssistRunState)
async def cancel_assist_run(run_id: str, opensearch: OpenSearchDep) -> AssistRunState:
    """Ask a live run to stop after its current chunk; what it wrote stays."""
    _state(run_id)
    cancel_run(run_id)
    return await _with_scope(opensearch, _state(run_id))


@router.post('/assist/runs/{run_id}/apply', response_model=AssistApplyResponse)
async def apply_assist_run(
    run_id: str, body: AssistApplyRequest, opensearch: OpenSearchDep
) -> AssistApplyResponse:
    """Label the scope's current members with ``class_name`` as a human would
    (validated, trainable), through the same writer as ``PUT /crops/batch_label``.

    A dry run (the default) counts. Members with a locked or validated class, and
    holdout or ignored ones, are left alone (`skipped_locked`). 409 ``scope_changed``
    when under 90% of the scope is what the run saw (``allow_changed_scope`` overrides),
    ``class_not_in_registry`` (matched by name) and ``run_has_no_scope`` (a bare
    selection or a filter run)."""
    return await apply_proposal(opensearch, run_id, _state(run_id), body)
