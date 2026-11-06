"""``POST /train/promote/{job_id}`` (a `promote` job), its status poll and ``/reload_promoted``."""

from __future__ import annotations

import uuid  # noqa: TC003 - FastAPI resolves the path parameter type
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path as PathParam, Query

from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.schemas.train_promote import PromoteJobStatus, PromoteRequest, ReloadPromotedResponse
from src.services.jobs import status as job_status
from src.services.training import promote_job
from src.services.training.promote_gate import PromoteGateFailedResponse
from src.services.training.promote_run import prepare_promote


router = APIRouter()
REQUESTED_BY = 'api'


@router.post(
    '/promote/{job_id}',
    response_model=PromoteJobStatus,
    status_code=202,
    responses={
        409: {'description': 'a promote of this Triton model name is already active (`job_busy`)'},
        422: {'model': PromoteGateFailedResponse},
    },
)
async def promote_run_job(
    payload: PromoteRequest,
    job_id: Annotated[str, PathParam(description='Training job_id from {api_prefix}/train/runs')],
    session: DbSessionDep,
    force: Annotated[
        bool,
        Query(
            description=(
                'Same as the request-body ``force`` (either one bypasses the promote gate); '
                'accepted on the query string so it reads like train/start.'
            )
        ),
    ] = False,
) -> PromoteJobStatus:
    """Promote a finished training run into the Triton model repo, as a job.

    Every 4xx a caller can fix is answered here, before anything is queued; the
    build, load and warm-up (2-3 minutes) run as a ``promote`` job. Poll
    ``GET {api_prefix}/train/promote/{job_id}/jobs/{promote_id}`` with the
    returned ``promote_id``: ``queued`` -> ``exporting`` -> ``loading`` ->
    ``building`` -> ``warming`` -> ``done`` | ``failed``.

    The job reads the run's ``status.json`` for the ONNX export the trainer
    produced during the ``exporting`` state, copies it into
    ``models/<triton_name>/<version>/model.onnx``, stores the ``models`` and
    ``promotions`` rows (the previous live promotion of the name is superseded),
    writes ``config.pbtxt`` and ``labels.txt`` from the stored labels and
    POSTs Triton's load endpoint.

    Errors:
        404: job not found, or its ONNX export hasn't been written
        409: ``job_busy``: a promote of this name is already queued or running;
              or a Triton model with this name already exists (use
              ``overwrite=true`` to clobber)
        422: status.json shows the run isn't in a promote-ready state, or
              the promote gate failed. The ``detail`` body always matches
              ``PromoteGateFailedResponse``: ``failures`` is a list of
              ``{code, message}`` (plus ``class_name`` where relevant) so
              a caller never has to string-parse a message, and
              ``force_allowed`` says whether resubmitting with
              ``force=true`` can get past this specific failure.
    """
    if force and not payload.force:
        payload = payload.model_copy(update={'force': True})

    plan = await prepare_promote(job_id, payload)
    started = await promote_job.start_promote(
        session, current_project_id(), plan, payload, requested_by=REQUESTED_BY
    )
    view = await job_status.get_job(session, started.job.id, project=current_project_id())
    return promote_job.wire(view)


@router.get('/promote/{job_id}/jobs/{promote_id}', response_model=PromoteJobStatus)
async def promote_job_status(
    job_id: Annotated[str, PathParam(description='Training job_id the promote belongs to')],
    promote_id: Annotated[uuid.UUID, PathParam(description='promote_id from the 202 response')],
    session: DbSessionDep,
) -> PromoteJobStatus:
    """Phase of a promote job: ``queued`` -> ``exporting`` -> ``loading``
    -> ``building`` -> ``warming`` -> ``done`` | ``failed``. ``result`` (the
    :class:`PromoteResponse` shape) is set on ``done``; ``error`` on ``failed``.
    404 for an unknown id or one that belongs to a different run."""
    state = await promote_job.read_job(session, current_project_id(), promote_id)
    if state is None or state.job_id != job_id:
        raise HTTPException(status_code=404, detail=f'promote job {str(promote_id)!r} not found')
    return state


# =============================================================================
# /reload_promoted
# =============================================================================


@router.post('/reload_promoted', response_model=ReloadPromotedResponse)
async def reload_promoted() -> ReloadPromotedResponse:
    """Re-``/load`` every promoted model Triton doesn't report READY.

    Triton in explicit-control mode only loads its ``--load-model`` list
    at startup, so a bare Triton restart (``make restart-triton``, or any
    ``docker compose restart``/recreate of the Triton service) silently
    strands every previously-promoted model at UNAVAILABLE until someone
    POSTs ``/load`` again. The API already runs this once at its own
    startup and on its periodic reconcile tick (see ``src/app/lifespan.py``); this
    route lets an operator trigger it on demand right after bouncing
    Triton, without needing a full API restart. ``make reload-promoted``
    calls this.

    A model that's been through ``DELETE {api_prefix}/models/{name}`` is
    never resurrected here -- that route removes the whole model
    directory, which ends its live promotion, and the scan only reloads
    live promotions.
    """
    from src.services.training.triton_reload import reload_promoted_models

    result = await reload_promoted_models(honor_unloaded=False)
    return ReloadPromotedResponse(
        status=result.get('status', 'ok'),
        reloaded=result.get('reloaded', []),
        failed=result.get('failed', []),
    )
