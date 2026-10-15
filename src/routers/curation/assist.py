"""Curation router sub-module: assist runs.

``POST /assist/runs`` takes one request shape for every scope (a crop, a
selection, a cluster, a sub-cluster, a filter): a dry run by default that counts
what the run would send to the VLM, then a background run that writes the VLM's
class suggestions and pack-declared attributes. The route only adapts HTTP; the
rules (scope, sample, lock rule, one run per project) live in
:mod:`src.services.curation.assist`.
"""

from __future__ import annotations

from src.routers.curation._api_error import api_error
from src.routers.curation._common import OpenSearchDep, router
from src.services.curation.assist.job import cancel_run, read_state
from src.services.curation.assist.models import AssistProgress, AssistRunRequest, AssistRunState
from src.services.curation.assist.prepare import prepare_run
from src.services.curation.assist.runner import start_run


def _state(run_id: str) -> AssistRunState:
    state = read_state(run_id)
    if state is None:
        raise api_error(404, 'not_found', f'no assist run {run_id}')
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
    return start_run(opensearch, prepared)


@router.get('/assist/runs/{run_id}', response_model=AssistRunState)
async def get_assist_run(run_id: str) -> AssistRunState:
    """The state of one run: progress while it runs, ``result`` when it ends."""
    return _state(run_id)


@router.post('/assist/runs/{run_id}/cancel', response_model=AssistRunState)
async def cancel_assist_run(run_id: str) -> AssistRunState:
    """Ask a live run to stop after its current chunk; what it wrote stays."""
    _state(run_id)
    cancel_run(run_id)
    return _state(run_id)
