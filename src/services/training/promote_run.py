"""Promote a finished training run into the Triton model repo: the validation gates and the
build/load step. The route validates synchronously (every 4xx a caller can fix) and runs
:func:`execute_promote` inline or as a background job."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.schemas.train_promote import PromoteRequest, PromoteResponse
from src.services.training import job_manifest, jobs as train_jobs
from src.services.training.promote_gate import (
    PROMOTE_GATE_MAP50_MIN,
    PROMOTE_GATE_PER_CLASS_PRECISION_MIN,
    PROMOTE_GATE_PER_CLASS_SUPPORT_MIN,
    PromoteGateFailedDetail,
    PromoteGateFailure,
    evaluate_promote_gate,
    registry_ids_contiguous_from_zero,
    resolve_full_registry_for_promote,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.services.training.class_remap import ClassRemapResult
    from src.services.training.job_models import TrainJobStatus

logger = get_logger(__name__)

_FORCE_OVERRIDE = 'pass force=true in the request body or as ?force=true'


@dataclass(frozen=True)
class PromotePlan:
    """A promote that passed every gate, ready to build."""

    job_id: str
    triton_name: str
    job_status: TrainJobStatus
    class_id_to_name: dict[int, str]
    class_remap: ClassRemapResult
    gate_report: dict[str, Any]
    gate_failures: list[PromoteGateFailure]


def _gate_refusal(
    message: str,
    code: str,
    failure_message: str | None = None,
    *,
    force_allowed: bool,
) -> RequestRejectedError:
    """The 422 body every promote gate answers: ``PromoteGateFailedDetail`` with one failure."""
    return RequestRejectedError(
        status_code=422,
        detail=PromoteGateFailedDetail(
            message=message,
            failures=[PromoteGateFailure(code=code, message=failure_message or message)],
            force_allowed=force_allowed,
            override=_FORCE_OVERRIDE if force_allowed else None,
        ).model_dump(),
    )


def _check_triton_name(triton_name: str) -> None:
    """Project namespacing (``projects_plan.md`` §5.3): the *requested* (unprefixed) name may
    not itself contain ``__`` -- that would collide with, or spoof, the namespacing separator
    once ``model_prefix`` is prepended (a ``default`` request named ``alpha__x`` would resolve
    to the same triton_name as ``alpha`` legitimately promoting ``x``)."""
    if '__' in triton_name:
        raise RequestRejectedError(
            status_code=422,
            detail={
                'code': 'triton_name_reserved_separator',
                'message': (
                    f'triton_name {triton_name!r} may not contain "__" -- reserved '
                    'as the project-namespacing separator'
                ),
            },
        )


async def _read_promote_ready_status(job_id: str) -> TrainJobStatus:
    job_status = await train_jobs.read_status(job_id)
    if job_status is None:
        raise RequestRejectedError(status_code=404, detail=f'job {job_id!r} not found')
    if job_status.state not in {'finished', 'exporting'}:
        # force=true only bypasses the promote-gate score thresholds and a missing
        # class_remap -- it cannot invent a finished run or an ONNX export never written.
        raise _gate_refusal(
            f'job {job_id!r} is not in a promote-ready state',
            'job_not_promote_ready',
            f'job {job_id!r} is in state {job_status.state!r}; only '
            "'finished' or 'exporting' runs can be promoted",
            force_allowed=False,
        )
    if not job_status.checkpoint_path:
        raise _gate_refusal(
            f'job {job_id!r} has no checkpoint_path in status.json',
            'no_checkpoint_path',
            force_allowed=False,
        )
    return job_status


def _evaluate_gate(
    job_status: TrainJobStatus, *, force: bool
) -> tuple[dict[str, Any], list[PromoteGateFailure]]:
    """Refuse an underqualified run unless the caller passed ``force``."""
    gate_failures = evaluate_promote_gate(job_status.eval)
    thresholds = {
        'map50_min': PROMOTE_GATE_MAP50_MIN,
        'per_class_precision_min': PROMOTE_GATE_PER_CLASS_PRECISION_MIN,
        'per_class_support_min': PROMOTE_GATE_PER_CLASS_SUPPORT_MIN,
    }
    if gate_failures and not force:
        raise RequestRejectedError(
            status_code=422,
            detail=PromoteGateFailedDetail(
                message='promote gate failed',
                failures=gate_failures,
                force_allowed=True,
                override=_FORCE_OVERRIDE,
                thresholds=thresholds,
            ).model_dump(),
        )
    return {
        'thresholds': thresholds,
        'failures': [f.model_dump() for f in gate_failures],
    }, gate_failures


def _require_remap_for_subset_run(job_id: str, *, force: bool) -> None:
    from src.services.training.promote_errors import ClassRemapMissingError

    if not force:
        message = str(ClassRemapMissingError(job_id))
        raise _gate_refusal(message, 'class_remap_missing', force_allowed=True)
    logger.warning(
        'curation_promote_class_remap_missing_force_bypass',
        job_id=job_id,
        note='subset/single_cls run promoted with force=true and no resolvable class_remap',
    )


def _check_full_class_identity(job_id: str, full_registry: dict[int, str], *, force: bool) -> None:
    """A full-class run from before the trainer always wrote ``class_remap.json``. The
    identity map (labels.txt line i = registry class i's name) is only correct when the
    pinned registry snapshot has no gap or deprecation for ``_build_export_id_map`` to have
    skipped -- provable from ``full_registry`` itself. Anything else is refused unless forced."""
    if registry_ids_contiguous_from_zero(full_registry):
        logger.info(
            'curation_promote_full_class_identity_proven',
            job_id=job_id,
            note='no class_remap, but the pinned registry has no gaps -- identity map is correct',
        )
        return
    if not force:
        message = (
            f'job {job_id!r} is a full-class run with no resolvable class_remap and its '
            'pinned registry has a gap or deprecated class -- the identity map '
            '(labels.txt line i = registry class i) is not provably correct for a '
            'dense-id-trained model; refusing to promote (pass force=true to bypass -- '
            'logged distinctly)'
        )
        raise _gate_refusal(message, 'class_remap_missing_full_class', force_allowed=True)
    logger.warning(
        'curation_promote_full_class_identity_unproven_force_bypass',
        job_id=job_id,
        note=(
            'full-class run promoted with force=true; no class_remap and the pinned '
            'registry has a gap/deprecation -- labels.txt may be mislabeled'
        ),
    )


def _check_remap_against_registry(
    class_remap: ClassRemapResult,
    include_classes: list[int],
    full_registry: dict[int, str],
    job_id: str,
) -> None:
    if len(class_remap.mapping) != len(include_classes):
        raise RequestRejectedError(
            status_code=422,
            detail=(
                f'class_remap length {len(class_remap.mapping)} != '
                f'len(include_classes) {len(include_classes)} for job {job_id!r} '
                f'(source={class_remap.source})'
            ),
        )
    for orig_id in class_remap.mapping:
        remap_name = class_remap.names[class_remap.mapping[orig_id]] if class_remap.names else None
        registry_name = full_registry.get(orig_id)
        if remap_name is not None and registry_name is not None and remap_name != registry_name:
            raise RequestRejectedError(
                status_code=422,
                detail=(
                    f'class_remap name mismatch for original id {orig_id}: '
                    f'remap says {remap_name!r}, registry says {registry_name!r} '
                    f'(job {job_id!r}, source={class_remap.source})'
                ),
            )


async def _resolve_class_names(
    job_id: str, job_status: TrainJobStatus, *, force: bool
) -> tuple[ClassRemapResult, dict[int, str]]:
    """The run's ``class_remap`` and the labels ``labels.txt`` will carry.

    Resolves the full-registry class names (id -> name), preferring the snapshot pinned at
    submit time; a subset-trained model is renumbered by ``build_class_id_to_name`` using the
    resolved remap (manifest lineage first, then the weights-dir file).
    """
    from src.services.training.class_remap import build_class_id_to_name, resolve_class_remap
    from src.services.training.promote_errors import ClassRemapUnreadableError

    full_registry = await resolve_full_registry_for_promote(job_id)
    job_spec = await job_manifest.read_job_spec(job_id)
    manifest = await job_manifest.read_manifest(job_id)
    is_subset_run = bool((job_spec or {}).get('include_classes')) or bool(
        (job_spec or {}).get('single_cls')
    )
    try:
        class_remap = resolve_class_remap(
            job_id=job_id,
            checkpoint_path=Path(job_status.checkpoint_path or ''),
            manifest=manifest,
        )
    except ClassRemapUnreadableError as exc:
        raise RequestRejectedError(status_code=exc.status_code, detail=str(exc)) from exc

    if class_remap.source == 'none':
        if is_subset_run:
            _require_remap_for_subset_run(job_id, force=force)
        else:
            _check_full_class_identity(job_id, full_registry, force=force)

    include_classes = (job_spec or {}).get('include_classes') or []
    if class_remap.source != 'none' and not class_remap.single_cls and include_classes:
        _check_remap_against_registry(class_remap, include_classes, full_registry, job_id)
    class_id_to_name = build_class_id_to_name(remap=class_remap, full_registry=full_registry)
    if class_remap.single_cls and len(class_id_to_name) != 1:
        raise RequestRejectedError(
            status_code=422,
            detail=(
                f'job {job_id!r} is single_cls but resolved labels.txt would have '
                f'{len(class_id_to_name)} lines, not 1'
            ),
        )
    return class_remap, class_id_to_name


async def prepare_promote(job_id: str, payload: PromoteRequest) -> PromotePlan:
    """Run every gate; raise ``RequestRejectedError`` for the first that refuses."""
    _check_triton_name(payload.triton_name)
    triton_name = f'{get_curation_config().model_prefix}{payload.triton_name}'
    job_status = await _read_promote_ready_status(job_id)
    gate_report, gate_failures = _evaluate_gate(job_status, force=payload.force)
    class_remap, class_id_to_name = await _resolve_class_names(
        job_id, job_status, force=payload.force
    )
    return PromotePlan(
        job_id=job_id,
        triton_name=triton_name,
        job_status=job_status,
        class_id_to_name=class_id_to_name,
        class_remap=class_remap,
        gate_report=gate_report,
        gate_failures=gate_failures,
    )


async def _stamp_lineage(plan: PromotePlan, result_triton_name: str, *, force: bool) -> bool:
    """Stamp the manifest's ``promoted_to``, including whether the gate was bypassed and the
    (possibly failing) report so a forced promote is traceable later. A run without a manifest
    (older) has nothing to stamp -- not an error. A real write failure is logged loudly and
    reported as ``lineage_stamped: false``; the promote is not failed, because the model is
    already live in Triton and a 500 would mislead."""
    try:
        stamped = await job_manifest.stamp_manifest_promotion(
            plan.job_id,
            triton_name=result_triton_name,
            promoted_at=datetime.now(tz=UTC).isoformat(),
            force_used=force,
            gate_report=plan.gate_report if plan.gate_failures else None,
        )
    except Exception as exc:
        logger.error('manifest_stamp_failed', job_id=plan.job_id, error=str(exc))
        return False
    if not stamped:
        logger.warning(
            'manifest_stamp_skipped_no_manifest',
            job_id=plan.job_id,
            note='no manifest on disk for this job (older run) — nothing to stamp',
        )
    return stamped


async def execute_promote(
    plan: PromotePlan, payload: PromoteRequest, on_phase: Callable[[str], None] | None = None
) -> PromoteResponse:
    """Build the Triton model for ``plan``, load it, and stamp the run's lineage."""
    from src.services.training.promote_errors import (
        CheckpointNotFoundError,
        ModelNameConflictError,
        PromoteError,
        TritonLoadError,
    )
    from src.services.training.triton_promote import promote_yolo26_to_triton

    try:
        result = await promote_yolo26_to_triton(
            status=plan.job_status,
            triton_name=plan.triton_name,
            class_id_to_name=plan.class_id_to_name,
            max_batch_size=payload.max_batch_size,
            input_size=payload.input_size,
            fp16=payload.fp16,
            overwrite=payload.overwrite,
            class_remap=plan.class_remap,
            project=get_curation_config().project_slug,
            on_phase=on_phase,
        )
    except CheckpointNotFoundError as exc:
        raise RequestRejectedError(status_code=404, detail=str(exc)) from exc
    except ModelNameConflictError as exc:
        raise RequestRejectedError(status_code=409, detail=str(exc)) from exc
    except TritonLoadError as exc:
        raise RequestRejectedError(status_code=502, detail=str(exc)) from exc
    except PromoteError as exc:
        raise RequestRejectedError(status_code=exc.status_code, detail=str(exc)) from exc

    lineage_stamped = await _stamp_lineage(plan, result.triton_name, force=payload.force)
    return PromoteResponse(
        job_id=result.job_id,
        triton_name=result.triton_name,
        onnx_path=result.onnx_path,
        config_path=result.config_path,
        labels_path=result.labels_path,
        triton_loaded=result.triton_loaded,
        force_used=payload.force,
        gate_report=plan.gate_report if plan.gate_failures else None,
        lineage_stamped=lineage_stamped,
        class_remap_source=plan.class_remap.source,
        cold_start_expected_on_first_inference=result.cold_start_expected_on_first_inference,
    )
