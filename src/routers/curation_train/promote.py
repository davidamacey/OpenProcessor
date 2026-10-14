"""``POST /train/promote/{job_id}``, its job-status poll and ``/reload_promoted``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, HTTPException, Path as PathParam, Query, Response

from src.schemas.train_promote import (
    PromoteJobStatus,
    PromoteRequest,
    PromoteResponse,
    ReloadPromotedResponse,
)
from src.services.training import promote_job
from src.services.training.promote_gate import PromoteGateFailedResponse
from src.services.training.promote_run import execute_promote, prepare_promote


if TYPE_CHECKING:
    from collections.abc import Callable

router = APIRouter()


@router.post(
    '/promote/{job_id}',
    response_model=PromoteResponse | PromoteJobStatus,
    status_code=202,
    responses={
        200: {'model': PromoteResponse, 'description': 'wait=true, or an already-active job'},
        422: {'model': PromoteGateFailedResponse},
    },
)
async def promote_run(
    payload: PromoteRequest,
    job_id: Annotated[str, PathParam(description='Training job_id from {api_prefix}/train/runs')],
    response: Response,
    wait: Annotated[
        bool,
        Query(
            description=(
                'true: block until the model is loaded and warmed (2-3 minutes; '
                'raise client/proxy timeouts to >= 300 s) and return the full '
                'PromoteResponse with 200. false (default): return 202 with a '
                'promote_id to poll.'
            )
        ),
    ] = False,
    force: Annotated[
        bool,
        Query(
            description=(
                'Same as the request-body ``force`` (either one bypasses the promote gate); '
                'accepted on the query string so it reads like train/start.'
            )
        ),
    ] = False,
) -> PromoteResponse | PromoteJobStatus:
    """Promote a finished training run into the Triton model repo.

    Reads the run's ``status.json`` for the ONNX export the trainer
    produced during the ``exporting`` state, copies it into
    ``models/<triton_name>/1/model.onnx``, writes ``config.pbtxt``
    using the YOLO26 single-output template (no NMS plugin — NMS is
    internal to the YOLO26 forward pass), writes
    ``labels.txt`` honoring any subset-training class_remap, and POSTs
    Triton's load endpoint to make the model active immediately.

    Errors:
        404: job not found, or its ONNX export hasn't been written
        409: a Triton model with this name already exists (use
              ``overwrite=true`` to clobber)
        422: status.json shows the run isn't in a promote-ready state, or
              the promote gate failed. The ``detail`` body always matches
              ``PromoteGateFailedResponse``: ``failures`` is a list of
              ``{code, message}`` (plus ``class_name`` where relevant) so
              a caller never has to string-parse a message, and
              ``force_allowed`` says whether resubmitting with
              ``force=true`` can get past this specific failure.
        502: Triton refused the load (config or weights mismatch)
    """
    if force and not payload.force:
        payload = payload.model_copy(update={'force': True})

    plan = await prepare_promote(job_id, payload)

    if wait:
        response.status_code = 200
        # Shielded: a proxy timeout / client abort must not cancel the load mid-way.
        return await promote_job.run_detached(execute_promote(plan, payload))

    # Async (default): the validation above already ran, so every 4xx a
    # caller can fix is synchronous; only the build/load/warm-up is a job.
    try:
        pjob, created = promote_job.claim(run_job_id=job_id, triton_name=plan.triton_name)
    except promote_job.PromoteJobConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                'code': 'promote_in_progress',
                'message': str(exc),
                'promote_id': exc.promote_id,
            },
        ) from exc
    if created:

        async def _work(on_phase: Callable[[str], None]) -> dict[str, Any]:
            return (await execute_promote(plan, payload, on_phase)).model_dump()

        promote_job.start(pjob, _work)
    else:
        response.status_code = 200  # idempotent: the already-active job
    state = promote_job.read_job(pjob.directory.name)
    assert state is not None  # claim() just wrote it
    return PromoteJobStatus(**state)


@router.get('/promote/{job_id}/jobs/{promote_id}', response_model=PromoteJobStatus)
async def promote_job_status(
    job_id: Annotated[str, PathParam(description='Training job_id the promote belongs to')],
    promote_id: Annotated[str, PathParam(description='promote_id from the 202 response')],
) -> PromoteJobStatus:
    """Phase of a background promote: ``queued`` -> ``exporting`` -> ``loading``
    -> ``building`` -> ``warming`` -> ``done`` | ``failed``. ``result`` (the
    synchronous :class:`PromoteResponse` shape) is set on ``done``; ``error``
    and ``error_status`` (the HTTP status a synchronous promote would have
    returned) on ``failed``. 404 for an unknown id or one that belongs to a
    different run."""
    state = promote_job.read_job(promote_id)
    if state is None or state['job_id'] != job_id:
        raise HTTPException(status_code=404, detail=f'promote job {promote_id!r} not found')
    return PromoteJobStatus(**state)


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
    startup and on its periodic reconcile tick (see ``src/main.py``); this
    route lets an operator trigger it on demand right after bouncing
    Triton, without needing a full API restart. ``make reload-promoted``
    calls this.

    A model that's been through ``DELETE {api_prefix}/models/{name}`` is
    never resurrected here -- that route removes the whole model
    directory, ``promote.json`` included, which is exactly what this
    scan keys off.
    """
    from src.services.training.triton_reload import reload_promoted_models

    result = await reload_promoted_models(honor_unloaded=False)
    return ReloadPromotedResponse(
        status=result.get('status', 'ok'),
        reloaded=result.get('reloaded', []),
        failed=result.get('failed', []),
    )
