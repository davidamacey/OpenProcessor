"""``POST /train/promote/{job_id}``, its job-status poll and ``/reload_promoted``."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Path as PathParam, Query, Response
from pydantic import BaseModel, Field

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.training import jobs as train_jobs, promote_job
from src.services.training.promote_gate import (
    PROMOTE_GATE_MAP50_MIN,
    PROMOTE_GATE_PER_CLASS_PRECISION_MIN,
    PROMOTE_GATE_PER_CLASS_SUPPORT_MIN,
    PromoteGateFailedDetail,
    PromoteGateFailedResponse,
    PromoteGateFailure,
    evaluate_promote_gate,
    registry_ids_contiguous_from_zero,
    resolve_full_registry_for_promote,
)


if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

config = get_curation_config()

router = APIRouter()


class PromoteRequest(BaseModel):
    """Body for ``POST {api_prefix}/train/promote/{job_id}``."""

    triton_name: str = Field(
        ...,
        description=(
            'Desired Triton model name (alphanumeric, underscore, hyphen). The model is served as '
            '`<project>__<triton_name>` (no prefix for `default`); the final name is returned as '
            '`triton_name` in the promote result and job status.'
        ),
        min_length=1,
        max_length=64,
    )
    max_batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description='Triton dynamic-batch upper bound. Must match what the '
        'ONNX export was done with (default 8).',
    )
    input_size: int = Field(
        default=640,
        ge=64,
        le=2048,
        description='Square input dim. Must match the ONNX export.',
    )
    fp16: bool = Field(
        default=True,
        description='Use FP16 precision in the JIT-compiled TensorRT engine.',
    )
    overwrite: bool = Field(
        default=False,
        description='If a Triton model with this name exists, delete it first.',
    )
    force: bool = Field(
        default=False,
        description=(
            'Bypass the promote gate (mAP50 ≥ 0.65, no class precision < 0.50, '
            'no class with support < 5). Use only for known-good experimental runs.'
        ),
    )


class PromoteResponse(BaseModel):
    job_id: str
    triton_name: str
    onnx_path: str
    config_path: str
    labels_path: str
    triton_loaded: bool
    force_used: bool = False
    gate_report: dict[str, Any] | None = None
    lineage_stamped: bool = False
    class_remap_source: str = 'none'
    # Truthful static hint, not a live measurement: see
    # TritonPromoter.PromoteResult.cold_start_expected_on_first_inference.
    cold_start_expected_on_first_inference: bool = True


class PromoteJobStatus(BaseModel):
    """A background promote (``POST /train/promote/{job_id}`` without ``wait=true``)."""

    promote_id: str
    job_id: str
    triton_name: str
    status: Literal['queued', 'exporting', 'loading', 'building', 'warming', 'done', 'failed']
    error: str | None = None
    error_status: int | None = Field(
        default=None, description='HTTP status a synchronous promote would have returned'
    )
    result: PromoteResponse | None = None
    started_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None
    poll_after_s: int | None = None


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

    # Lazy import — keeps the API container slim if no one ever
    # promotes (e.g. a fresh dev box).
    from pathlib import Path

    from src.services.training.class_remap import build_class_id_to_name, resolve_class_remap
    from src.services.training.promote_errors import (
        CheckpointNotFoundError,
        ClassRemapMissingError,
        ClassRemapUnreadableError,
        ModelNameConflictError,
        PromoteError,
        TritonLoadError,
    )
    from src.services.training.triton_promote import promote_yolo26_to_triton

    # Project namespacing (docs/design/openprocessor_internal/
    # projects_plan.md §5.3): the *requested* (unprefixed) name may not
    # itself contain '__' -- that would collide with, or spoof, the
    # namespacing separator once model_prefix is prepended (e.g. a
    # `default` request named 'alpha__x' would resolve to the exact same
    # triton_name as `alpha` legitimately promoting 'x'). `default`'s
    # empty model_prefix means its promoted names are otherwise unchanged.
    if '__' in payload.triton_name:
        raise HTTPException(
            status_code=422,
            detail={
                'code': 'triton_name_reserved_separator',
                'message': (
                    f'triton_name {payload.triton_name!r} may not contain "__" -- reserved '
                    'as the project-namespacing separator'
                ),
            },
        )
    triton_name = f'{config.model_prefix}{payload.triton_name}'

    job_status = await train_jobs.read_status(job_id)
    if job_status is None:
        raise HTTPException(status_code=404, detail=f'job {job_id!r} not found')

    if job_status.state not in {'finished', 'exporting'}:
        raise HTTPException(
            status_code=422,
            detail=PromoteGateFailedDetail(
                message=f'job {job_id!r} is not in a promote-ready state',
                failures=[
                    PromoteGateFailure(
                        code='job_not_promote_ready',
                        message=(
                            f'job {job_id!r} is in state {job_status.state!r}; only '
                            "'finished' or 'exporting' runs can be promoted"
                        ),
                    )
                ],
                # force=true only bypasses the promote-gate score thresholds
                # and a missing class_remap -- it cannot invent a finished
                # training run or an ONNX export that was never written.
                force_allowed=False,
            ).model_dump(),
        )

    if not job_status.checkpoint_path:
        raise HTTPException(
            status_code=422,
            detail=PromoteGateFailedDetail(
                message=f'job {job_id!r} has no checkpoint_path in status.json',
                failures=[
                    PromoteGateFailure(
                        code='no_checkpoint_path',
                        message=f'job {job_id!r} has no checkpoint_path in status.json',
                    )
                ],
                force_allowed=False,
            ).model_dump(),
        )

    # Promote gate. Refuses underqualified runs unless the
    # caller explicitly passes force=true.
    gate_failures = evaluate_promote_gate(job_status.eval)
    gate_thresholds = {
        'map50_min': PROMOTE_GATE_MAP50_MIN,
        'per_class_precision_min': PROMOTE_GATE_PER_CLASS_PRECISION_MIN,
        'per_class_support_min': PROMOTE_GATE_PER_CLASS_SUPPORT_MIN,
    }
    gate_report: dict[str, Any] = {
        'thresholds': gate_thresholds,
        'failures': [f.model_dump() for f in gate_failures],
    }
    if gate_failures and not payload.force:
        raise HTTPException(
            status_code=422,
            detail=PromoteGateFailedDetail(
                message='promote gate failed',
                failures=gate_failures,
                force_allowed=True,
                override='pass force=true in the request body or as ?force=true',
                thresholds=gate_thresholds,
            ).model_dump(),
        )

    # Resolve full-registry class names (id -> name), preferring the
    # snapshot pinned at submit time. Subset-trained models get
    # renumbered inside build_class_id_to_name using the resolved remap
    # (manifest lineage.class_remap first, then the weights-dir file).
    full_registry = await resolve_full_registry_for_promote(job_id)
    job_spec = await train_jobs.read_job_spec(job_id)
    manifest = await train_jobs.read_manifest(job_id)
    is_subset_run = bool((job_spec or {}).get('include_classes')) or bool(
        (job_spec or {}).get('single_cls')
    )

    try:
        class_remap = resolve_class_remap(
            job_id=job_id,
            checkpoint_path=Path(job_status.checkpoint_path),
            manifest=manifest,
        )
    except ClassRemapUnreadableError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    if class_remap.source == 'none' and is_subset_run:
        if not payload.force:
            remap_missing_message = str(ClassRemapMissingError(job_id))
            raise HTTPException(
                status_code=422,
                detail=PromoteGateFailedDetail(
                    message=remap_missing_message,
                    failures=[
                        PromoteGateFailure(
                            code='class_remap_missing', message=remap_missing_message
                        )
                    ],
                    force_allowed=True,
                    override='pass force=true in the request body or as ?force=true',
                ).model_dump(),
            )
        logger.warning(
            'curation_promote_class_remap_missing_force_bypass',
            job_id=job_id,
            note='subset/single_cls run promoted with force=true and no resolvable class_remap',
        )

    if class_remap.source == 'none' and not is_subset_run:
        # Older run, from before the trainer always wrote class_remap.json
        # for full-class runs too. The identity map (labels.txt line i =
        # registry class i's name) is only correct when the pinned
        # registry snapshot has no gap/deprecation for _build_export_id_map
        # to have skipped -- provable from full_registry itself. Anything
        # else is the exact bug this fix closes: refuse unless forced.
        if not registry_ids_contiguous_from_zero(full_registry):
            if not payload.force:
                identity_unproven_message = (
                    f'job {job_id!r} is a full-class run with no resolvable class_remap and its '
                    'pinned registry has a gap or deprecated class -- the identity map '
                    '(labels.txt line i = registry class i) is not provably correct for a '
                    'dense-id-trained model; refusing to promote (pass force=true to bypass -- '
                    'logged distinctly)'
                )
                raise HTTPException(
                    status_code=422,
                    detail=PromoteGateFailedDetail(
                        message=identity_unproven_message,
                        failures=[
                            PromoteGateFailure(
                                code='class_remap_missing_full_class',
                                message=identity_unproven_message,
                            )
                        ],
                        force_allowed=True,
                        override='pass force=true in the request body or as ?force=true',
                    ).model_dump(),
                )
            logger.warning(
                'curation_promote_full_class_identity_unproven_force_bypass',
                job_id=job_id,
                note=(
                    'full-class run promoted with force=true; no class_remap and the pinned '
                    'registry has a gap/deprecation -- labels.txt may be mislabeled'
                ),
            )
        else:
            logger.info(
                'curation_promote_full_class_identity_proven',
                job_id=job_id,
                note='no class_remap, but the pinned registry has no gaps -- identity map is correct',
            )

    include_classes = (job_spec or {}).get('include_classes') or []
    if class_remap.source != 'none' and not class_remap.single_cls and include_classes:
        if len(class_remap.mapping) != len(include_classes):
            raise HTTPException(
                status_code=422,
                detail=(
                    f'class_remap length {len(class_remap.mapping)} != '
                    f'len(include_classes) {len(include_classes)} for job {job_id!r} '
                    f'(source={class_remap.source})'
                ),
            )
        for orig_id in class_remap.mapping:
            remap_name = None
            if class_remap.names:
                remap_name = class_remap.names[class_remap.mapping[orig_id]]
            registry_name = full_registry.get(orig_id)
            if remap_name is not None and registry_name is not None and remap_name != registry_name:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f'class_remap name mismatch for original id {orig_id}: '
                        f'remap says {remap_name!r}, registry says {registry_name!r} '
                        f'(job {job_id!r}, source={class_remap.source})'
                    ),
                )
    class_id_to_name = build_class_id_to_name(
        remap=class_remap,
        full_registry=full_registry,
    )
    if class_remap.single_cls and len(class_id_to_name) != 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f'job {job_id!r} is single_cls but resolved labels.txt would have '
                f'{len(class_id_to_name)} lines, not 1'
            ),
        )

    async def _execute(on_phase: Callable[[str], None] | None = None) -> PromoteResponse:
        try:
            result = await promote_yolo26_to_triton(
                status=job_status,
                triton_name=triton_name,
                class_id_to_name=class_id_to_name,
                max_batch_size=payload.max_batch_size,
                input_size=payload.input_size,
                fp16=payload.fp16,
                overwrite=payload.overwrite,
                class_remap=class_remap,
                project=config.project_slug,
                on_phase=on_phase,
            )
        except CheckpointNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelNameConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TritonLoadError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except PromoteError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        # Stamp the manifest's promoted_to field, including
        # whether the gate was bypassed and the (possibly-failing) report so a
        # forced promote is traceable later. Older runs without a manifest
        # legitimately have nothing to stamp — stamp_manifest_promotion returns
        # False for that case, which is NOT an error. An actual write failure
        # (disk, permissions, corrupt JSON) is a real problem: we don't fail
        # the promote outright (the model is already live in Triton at this
        # point — a 500 here would be misleading), but we surface it loudly via
        # both an ERROR-level log and `lineage_stamped: false` in the response
        # instead of the previous silent WARNING-and-forget.
        from datetime import datetime

        promoted_at = datetime.now(tz=UTC).isoformat()
        lineage_stamped = False
        try:
            lineage_stamped = await train_jobs.stamp_manifest_promotion(
                job_id,
                triton_name=result.triton_name,
                promoted_at=promoted_at,
                force_used=payload.force,
                gate_report=gate_report if gate_failures else None,
            )
            if not lineage_stamped:
                logger.warning(
                    'manifest_stamp_skipped_no_manifest',
                    job_id=job_id,
                    note='no manifest on disk for this job (older run) — nothing to stamp',
                )
        except Exception as exc:
            logger.error('manifest_stamp_failed', job_id=job_id, error=str(exc))

        return PromoteResponse(
            job_id=result.job_id,
            triton_name=result.triton_name,
            onnx_path=result.onnx_path,
            config_path=result.config_path,
            labels_path=result.labels_path,
            triton_loaded=result.triton_loaded,
            force_used=payload.force,
            gate_report=gate_report if gate_failures else None,
            lineage_stamped=lineage_stamped,
            class_remap_source=class_remap.source,
            cold_start_expected_on_first_inference=result.cold_start_expected_on_first_inference,
        )

    if wait:
        response.status_code = 200
        # Shielded: a proxy timeout / client abort must not cancel the load mid-way.
        return await promote_job.run_detached(_execute())

    # Async (default): the validation above already ran, so every 4xx a
    # caller can fix is synchronous; only the build/load/warm-up is a job.
    try:
        pjob, created = promote_job.claim(run_job_id=job_id, triton_name=triton_name)
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
            return (await _execute(on_phase)).model_dump()

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


class ReloadPromotedResponse(BaseModel):
    status: str
    reloaded: list[str] = []
    failed: list[str] = []


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
