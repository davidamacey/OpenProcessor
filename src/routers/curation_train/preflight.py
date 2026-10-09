"""``POST /train/preflight`` and the preflight orchestration shared with ``/start``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import ORJSONResponse

from src.clients.curation_opensearch.registry import get_class_registry
from src.config import get_curation_config
from src.config.curation import items_index
from src.routers.curation._common import OpenSearchDep  # noqa: TC001 - used at runtime by FastAPI
from src.routers.curation._config_common_models import api_error
from src.services.curation.dataset_thresholds import (
    HARD_MIN_CROPS_PER_CLASS,
    MIN_TEST_CROPS_PER_CLASS,
    WARN_MIN_CROPS_PER_CLASS,
)
from src.services.curation.export_readiness import (
    export_class_split_check,
    export_generation_check,
    export_size_check,
    export_splits_check,
    export_unlabeled_objects_check,
    items_index_generation,
)
from src.services.training import jobs as train_jobs
from src.services.training.augmentation_presets import PRESET_IDS
from src.services.training.gpu_arbiter import (
    containers_to_stop,
    docker_client_available,
    needs_service_stop,
    parse_cuda_visible_devices,
    probe_trainer_reachable,
)
from src.services.training.job_models import TrainJobSpec  # noqa: TC001 - FastAPI body type
from src.services.training.preflight_checks import (
    MIN_FREE_DISK_GB,
    SINGLE_CLASS_DATASET_KINDS,
    PreflightCheck,
    PreflightReport,
    append_label_scan_checks,
    append_single_class_data_checks,
    augmentation_preset_error,
    count_pending_ingest,
    count_validated_and_test_per_class,
    free_gb,
    read_export_manifest,
    read_trainer_gpu_order,
    resolve_disk_check_path,
    resolve_target_classes,
    training_volume_mount_sane,
    unresolvable_include_classes,
)
from src.services.training.profiles import RESERVED_OPTIMIZERS_YOLO26


if TYPE_CHECKING:
    from src.services.training.job_models import AugmentationSpec

router = APIRouter(default_response_class=ORJSONResponse)


def refuse_unknown_augmentation_preset(augmentation: AugmentationSpec | None) -> None:
    """``422`` (even with ``force``) before any GPU claim or job write: the
    trainer can never build an unknown preset."""
    error = augmentation_preset_error(augmentation)
    if error is not None:
        raise HTTPException(
            status_code=422,
            detail={
                'message': error,
                'field': 'augmentation.preset',
                'valid_presets': list(PRESET_IDS),
            },
        )


def refuse_empty_val_split(report: PreflightReport) -> None:
    """422 ``empty_val_split`` even with ``force``: the trainer cannot
    validate on an empty val folder, so this is a hard failure, not a
    warning ``force`` may bypass (balance/size warnings still can)."""
    row = next((c for c in report.checks if c.name == 'export_splits_nonempty'), None)
    if row is None or row.severity != 'block':
        return
    if 'val' in ((row.detail or {}).get('empty_splits') or []):
        raise api_error(422, 'empty_val_split', row.message)


def _refuse_export_outside_project(dataset_export_dir: str) -> None:
    """422 ``export_outside_project`` unless the export lives under the
    bound project's own ``export_root``.

    Runs before any check reads the export (manifest, registry, label
    scan), so a foreign or arbitrary path is never read back into the
    report. Not a preflight row: ``force`` must not bypass it.
    """
    cfg = get_curation_config()
    export_root = Path(cfg.export_root).resolve()
    if not Path(dataset_export_dir).resolve().is_relative_to(export_root):
        raise api_error(
            422,
            'export_outside_project',
            f"dataset_export_dir is outside the export root of project '{cfg.project_slug}'",
            project=cfg.project_slug,
        )


async def run_preflight(
    spec: TrainJobSpec,
    opensearch: Any,
) -> PreflightReport:
    """Run every preflight check and return the bundled report.

    Each check appends a row regardless of pass/fail so the UI can render
    the full table (good for the user to know which checks ran).
    """
    checks: list[PreflightCheck] = []

    # ---- 0. dataset_export_dir defaults to the current export ---------------
    # F-73: required-with-no-default forced every caller (including
    # CURATION.md's own worked example) to look up and paste the current
    # export path by hand. Null/omitted now resolves to the same `current`
    # symlink target GET /export/status reports, mutating `spec` in place
    # so every check below (and, on /start, the job.json write) sees the
    # resolved path -- exactly as if the caller had passed it themselves.
    if not spec.dataset_export_dir:
        from src.services.curation.export import resolve_current_export_dir

        try:
            spec.dataset_export_dir = str(resolve_current_export_dir())
        except FileNotFoundError:
            checks.append(
                PreflightCheck(
                    name='dataset_export_dir',
                    severity='block',
                    message=(
                        'no dataset_export_dir was given and no export exists yet '
                        '(data/exports/current). Run POST /export/yolo first, or '
                        'pass dataset_export_dir explicitly.'
                    ),
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    name='dataset_export_dir',
                    severity='ok',
                    message=f'defaulted to the current export: {spec.dataset_export_dir}',
                )
            )

    if spec.dataset_export_dir:
        _refuse_export_outside_project(spec.dataset_export_dir)

    # ---- 1. optimizer != auto -------------------------------------------------
    optimizer = (spec.hyperparameters or {}).get('optimizer')
    if isinstance(optimizer, str) and optimizer.lower() in {
        o.lower() for o in RESERVED_OPTIMIZERS_YOLO26
    }:
        checks.append(
            PreflightCheck(
                name='optimizer_not_auto',
                severity='block',
                message=(
                    "optimizer='auto' is reserved on YOLO26 (Ultralytics issue "
                    '#23696). Use MuSGD explicitly.'
                ),
                detail={'optimizer': optimizer},
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name='optimizer_not_auto',
                severity='ok',
                message=f'optimizer={optimizer or "MuSGD (default)"} is allowed',
            )
        )

    # ---- 1b. augmentation preset is one the trainer can build ------------------
    preset_error = augmentation_preset_error(spec.augmentation)
    if preset_error is not None:
        checks.append(
            PreflightCheck(
                name='augmentation_preset',
                severity='block',
                message=preset_error,
                detail={
                    'preset': spec.augmentation.preset if spec.augmentation else None,
                    'valid_presets': list(PRESET_IDS),
                },
            )
        )
    else:
        enabled = spec.augmentation is not None and spec.augmentation.enabled
        checks.append(
            PreflightCheck(
                name='augmentation_preset',
                severity='ok',
                message=(
                    f'augmentation preset {spec.augmentation.preset!r} is available'
                    if enabled and spec.augmentation is not None
                    else 'augmentation disabled; no preset to check'
                ),
            )
        )

    # ---- 2. free disk space ---------------------------------------------------
    disk_check_path = resolve_disk_check_path(spec)
    if not training_volume_mount_sane(disk_check_path):
        checks.append(
            PreflightCheck(
                name='free_disk',
                severity='block',
                message=(
                    f'{disk_check_path} appears to be on the same filesystem as '
                    "the container's own root — cannot see the trainer data "
                    'volume. Check the bind mount before training.'
                ),
                detail={'path': disk_check_path},
            )
        )
    else:
        free = free_gb(disk_check_path)
        if free is None:
            checks.append(
                PreflightCheck(
                    name='free_disk',
                    severity='unknown',
                    message=f'could not stat {disk_check_path} for free disk space',
                    detail={'path': disk_check_path},
                )
            )
        elif free < MIN_FREE_DISK_GB:
            checks.append(
                PreflightCheck(
                    name='free_disk',
                    severity='block',
                    message=(
                        f'Only {free:.1f} GB free on {disk_check_path} — need '
                        f'≥{MIN_FREE_DISK_GB} GB. Free space before training.'
                    ),
                    detail={'free_gb': round(free, 1), 'required_gb': MIN_FREE_DISK_GB},
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    name='free_disk',
                    severity='ok',
                    message=f'{free:.1f} GB free on {disk_check_path}',
                )
            )

    # ---- 2b. trainer reachable ------------------------------------------
    # Without this, /start writes job.json and the run sits in `queued`
    # forever with no error if the configured trainer container was never started.
    trainer_severity, trainer_detail = await probe_trainer_reachable()
    checks.append(
        PreflightCheck(
            name='trainer_reachable',
            severity=trainer_severity,
            message=trainer_detail,
        )
    )

    # ---- 2c. GPU arbiter can actually stop what this claim requires -----
    # containers_to_stop() names real GPU-resident containers (e.g. a large
    # vLLM/Triton process) this run's GPU claim must free. If the docker
    # SDK/socket isn't usable from this container, claim_gpus_for_training
    # cannot stop them -- proceeding would start training right next to
    # that service on the same GPU. Blocking here (not just at /start)
    # means the frontend form surfaces the failure before the user submits.
    required_stop_names = containers_to_stop(spec.cuda_visible_devices)
    if required_stop_names and not docker_client_available():
        checks.append(
            PreflightCheck(
                name='gpu_arbiter',
                severity='block',
                message=(
                    f'docker SDK/socket unavailable in the API container -- cannot stop '
                    f'{", ".join(required_stop_names)} for this GPU claim'
                ),
            )
        )
    elif required_stop_names:
        checks.append(
            PreflightCheck(
                name='gpu_arbiter',
                severity='ok',
                message=f'can stop: {", ".join(required_stop_names)}',
            )
        )

    # ---- 2d. trainer is actually attached to the requested host GPU(s) --
    # OP_GPU_ALLOWED_IDS only says a host id is *policy-permitted*; it says
    # nothing about which container is physically pinned to it. Without
    # this, a request for a GPU the trainer container was never attached to
    # (device_ids/OP_TRAIN_GPU_ORDER drift, or a GPU-policy change that
    # forgot to redeploy the trainer) writes job.json and sits in
    # `queued`/`starting` forever with no error (the original TR-2 failure
    # mode). The trainer publishes its attachment at startup
    # (write_trainer_capabilities in docker/trainer/trainer.py); an absent
    # file (older image) only downgrades this to a warning.
    trainer_gpu_order = read_trainer_gpu_order()
    requested_ids = parse_cuda_visible_devices(spec.cuda_visible_devices)
    if trainer_gpu_order is None:
        checks.append(
            PreflightCheck(
                name='trainer_gpus',
                severity='warn',
                message=(
                    'no trainer capabilities file found -- cannot verify the trainer '
                    'is attached to the requested GPU(s) (older trainer image?)'
                ),
            )
        )
    elif trainer_gpu_order and (
        unattached := [i for i in requested_ids if i not in trainer_gpu_order]
    ):
        checks.append(
            PreflightCheck(
                name='trainer_gpus',
                severity='block',
                message=(
                    f'trainer is attached to host GPUs {trainer_gpu_order}; requested {unattached}'
                ),
                detail={'trainer_gpu_order': trainer_gpu_order, 'requested': requested_ids},
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name='trainer_gpus',
                severity='ok',
                message=(
                    f'trainer is attached to host GPUs {trainer_gpu_order}'
                    if trainer_gpu_order
                    else 'trainer reports no GPU restriction (attached to every GPU at its host index)'
                ),
            )
        )

    # ---- 3. active run --------------------------------------------------------
    try:
        active = await train_jobs.get_active_job()
    except RuntimeError as exc:
        checks.append(
            PreflightCheck(
                name='active_run',
                severity='block',
                message=str(exc),
            )
        )
        active = None
    else:
        if active is not None and active.state in (
            'queued',
            'starting',
            'running',
            'exporting',
        ):
            checks.append(
                PreflightCheck(
                    name='active_run',
                    severity='block',
                    message=(
                        f'Another run is in progress (job_id={active.job_id}, '
                        f'state={active.state}). Cancel or wait for it to finish.'
                    ),
                    detail={'job_id': active.job_id, 'state': active.state},
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    name='active_run',
                    severity='ok',
                    message='No other training run in progress',
                )
            )

    # ---- 4 & 5. data sufficiency (single-class-aware) ---------------------------------
    # A single-class export validates from its own manifest (disk dataset),
    # not the multi-class registry's class_validated counts.
    _single_class_manifest = read_export_manifest(spec.dataset_export_dir)
    _is_single_class = _single_class_manifest.get('dataset_kind') in SINGLE_CLASS_DATASET_KINDS
    target_classes = [] if _is_single_class else resolve_target_classes(spec)

    # ---- 3b. include_classes resolvable against this export ----------
    # Previously an unresolvable include_classes id (deprecated, typo, or a
    # class this export never had) surfaced as either a 500 or a job that
    # failed deep inside the trainer container (subset_dataset.py's own
    # ValueError, minutes into a run). Catch it here instead, at the API
    # boundary, with a structured preflight check. Skipped entirely for single-class
    # jobs (single-class, no include_classes concept).
    if not _is_single_class and spec.include_classes:
        unresolvable = unresolvable_include_classes(spec.dataset_export_dir, spec.include_classes)
        if unresolvable:
            checks.append(
                PreflightCheck(
                    name='include_classes_resolvable',
                    severity='block',
                    message=(
                        f'include_classes has {len(unresolvable)} id(s) not in this '
                        f"export's active class set (deprecated, typo, or never "
                        f'exported): {unresolvable}'
                    ),
                    detail={'unresolvable_class_ids': unresolvable},
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    name='include_classes_resolvable',
                    severity='ok',
                    message=f'All {len(spec.include_classes)} include_classes ids resolve in this export',
                )
            )
    if _is_single_class:
        append_single_class_data_checks(checks, _single_class_manifest)
    elif not target_classes:
        checks.append(
            PreflightCheck(
                name='class_balance',
                severity='block',
                message='No classes selected and registry is empty',
            )
        )
        # Test-holdout still emitted (skipped) so the UI can render the row.
        checks.append(
            PreflightCheck(
                name='test_holdout',
                severity='block',
                message='Cannot evaluate test holdout — no target classes resolved',
            )
        )
    else:
        # One search covers both per-class validated counts and
        # per-class test-holdout counts (used in check 5 below).
        counts, test_counts = await count_validated_and_test_per_class(opensearch, target_classes)
        registry = get_class_registry()
        sub_blocking: list[dict[str, Any]] = []
        sub_warn: list[dict[str, Any]] = []
        for cid in target_classes:
            n = counts.get(cid, 0)
            entry = registry.get(cid)
            cname = entry.class_name if entry else f'class_{cid}'
            row = {'class_id': cid, 'name': cname, 'validated': n}
            if n < HARD_MIN_CROPS_PER_CLASS:
                sub_blocking.append(row)
            elif n < WARN_MIN_CROPS_PER_CLASS:
                sub_warn.append(row)
        if sub_blocking:
            names = ', '.join(f'{r["name"]} ({r["validated"]})' for r in sub_blocking)
            checks.append(
                PreflightCheck(
                    name='class_balance',
                    severity='block',
                    message=(
                        f'{len(sub_blocking)} class(es) below hard floor of '
                        f'{HARD_MIN_CROPS_PER_CLASS} validated crops: {names}'
                    ),
                    detail={'classes': sub_blocking},
                )
            )
        elif sub_warn:
            names = ', '.join(f'{r["name"]} ({r["validated"]})' for r in sub_warn)
            checks.append(
                PreflightCheck(
                    name='class_balance',
                    severity='warn',
                    message=(
                        f'{len(sub_warn)} class(es) below recommended '
                        f'{WARN_MIN_CROPS_PER_CLASS}: {names}'
                    ),
                    detail={'classes': sub_warn},
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    name='class_balance',
                    severity='ok',
                    message=(
                        f'All {len(target_classes)} classes have '
                        f'≥{WARN_MIN_CROPS_PER_CLASS} validated crops'
                    ),
                )
            )

        # ---- 5. test holdout coverage ----------------------------------------
        # test_counts came from the merged query above.
        thin_test: list[dict[str, Any]] = []
        for cid in target_classes:
            n = test_counts.get(cid, 0)
            if n < MIN_TEST_CROPS_PER_CLASS:
                entry = registry.get(cid)
                cname = entry.class_name if entry else f'class_{cid}'
                thin_test.append({'class_id': cid, 'name': cname, 'test_crops': n})
        if thin_test:
            names = ', '.join(f'{r["name"]} ({r["test_crops"]})' for r in thin_test)
            checks.append(
                PreflightCheck(
                    name='test_holdout',
                    severity='block' if any(r['test_crops'] == 0 for r in thin_test) else 'warn',
                    message=(
                        f'{len(thin_test)} class(es) below {MIN_TEST_CROPS_PER_CLASS} '
                        f'test crops: {names}. Refreeze the test holdout.'
                    ),
                    detail={'classes': thin_test},
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    name='test_holdout',
                    severity='ok',
                    message=f'All classes have ≥{MIN_TEST_CROPS_PER_CLASS} test crops',
                )
            )

    append_label_scan_checks(checks, spec, _single_class_manifest, _is_single_class)

    # ---- export readiness: not empty, trainable splits, built from
    # the current index. Per-class coverage is multi-class only: a
    # single-class export has one target class, which the overall
    # train/val check already covers. So are the unlabeled-object counts
    # (export_unlabeled_objects): only the multi-class exporter records them.
    export_manifest = read_export_manifest(spec.dataset_export_dir)
    class_split_result = (
        (
            'ok',
            'not applicable for this dataset kind (single-class: covered by '
            'export_splits_nonempty)',
            {},
        )
        if _is_single_class
        else export_class_split_check(export_manifest, spec.include_classes)
    )
    unlabeled_result = (
        ('ok', 'not applicable for this dataset kind (single-class)', {})
        if _is_single_class
        else export_unlabeled_objects_check(export_manifest)
    )
    for name, (severity, message, detail) in (
        ('export_not_empty', export_size_check(export_manifest)),
        ('export_splits_nonempty', export_splits_check(export_manifest)),
        ('export_class_split_coverage', class_split_result),
        ('export_unlabeled_objects', unlabeled_result),
        (
            'export_generation',
            export_generation_check(
                export_manifest,
                await items_index_generation(opensearch, items_index()),
            ),
        ),
    ):
        checks.append(
            PreflightCheck(name=name, severity=severity, message=message, detail=detail or None)
        )

    # ---- validated items the export dropped for unregistered class ids ------
    dropped = export_manifest.get('dropped_unregistered_class_ids')
    if isinstance(dropped, dict):
        total = sum(int(v) for v in dropped.values())
        checks.append(
            PreflightCheck(
                name='unregistered_class_ids',
                severity='warn' if total else 'ok',
                message=(
                    f'{total:,} item(s) were left out of this export because their '
                    f'class id is not in the registry ({", ".join(sorted(dropped))}). '
                    'Relabel or undo them in curation, then re-export.'
                    if total
                    else 'every exported item has a registered class id'
                ),
                detail={'dropped_unregistered_class_ids': dropped},
            )
        )

    # ---- active-ingest warning (claim stops GPU-scoped ingest containers) -
    if needs_service_stop(spec.cuda_visible_devices):
        stopped = containers_to_stop(spec.cuda_visible_devices)
        pending = await count_pending_ingest(opensearch)
        stopped_names = ', '.join(stopped)
        if pending > 0:
            checks.append(
                PreflightCheck(
                    name='ingest_idle',
                    severity='warn',
                    message=(
                        f'{pending:,} crops are still pending region '
                        f'detection/verification. This run stops {stopped_names}, '
                        'pausing that ingest until the run finishes (it auto-resumes '
                        'afterward).'
                    ),
                    detail={'pending_ingest': pending, 'stops_containers': list(stopped)},
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    name='ingest_idle',
                    severity='ok',
                    message=f'No ingest backlog — safe to stop {stopped_names} for training',
                    detail={'stops_containers': list(stopped)},
                )
            )

    blocked = any(c.severity == 'block' for c in checks)
    summary = (
        'BLOCKED — fix the highlighted checks before training'
        if blocked
        else 'All checks passed (warnings allowed)'
    )
    return PreflightReport(blocked=blocked, checks=checks, summary=summary)


# =============================================================================
# Endpoints
# =============================================================================


@router.post('/preflight', response_model=PreflightReport)
async def preflight(spec: TrainJobSpec, opensearch: OpenSearchDep) -> PreflightReport:
    """Run every preflight gate and return the bundled report.

    The frontend uses this for inline validation while the user is
    filling the form (no side effects). ``/start`` calls this internally.
    """
    return await run_preflight(spec, opensearch)
