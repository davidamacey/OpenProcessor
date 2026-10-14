"""The training preflight: every check, in the published order, bundled into one report.

Each check appends a row whether it passes or not, so the UI can render the full
table (and show which checks ran). ``POST /train/preflight`` returns the report;
``/start`` runs the same function and refuses on a blocking row.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.registry import get_class_registry
from src.config import get_curation_config
from src.config.curation import items_index
from src.core.exceptions import RequestRejectedError
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
from src.services.rejections import rejection
from src.services.training import jobs as train_jobs
from src.services.training.augmentation_presets import PRESET_IDS
from src.services.training.gpu_arbiter_docker import docker_client_available
from src.services.training.gpu_arbiter_policy import (
    containers_to_stop,
    needs_service_stop,
    parse_cuda_visible_devices,
)
from src.services.training.preflight_checks import (
    MIN_FREE_DISK_GB,
    SINGLE_CLASS_DATASET_KINDS,
    ExportManifestUnreadableError,
    PreflightCheck,
    PreflightReport,
    append_label_scan_checks,
    append_single_class_data_checks,
    augmentation_preset_error,
    count_pending_ingest,
    count_validated_and_test_per_class,
    free_gb,
    ingest_idle_check,
    read_export_manifest,
    read_trainer_gpu_order,
    resolve_disk_check_path,
    resolve_target_classes,
    training_volume_mount_sane,
    unresolvable_include_classes,
)
from src.services.training.profiles import RESERVED_OPTIMIZERS_YOLO26
from src.services.training.trainer_reachability import probe_trainer_reachable


if TYPE_CHECKING:
    from src.services.training.job_models import AugmentationSpec, TrainJobSpec


def refuse_unknown_augmentation_preset(augmentation: AugmentationSpec | None) -> None:
    """``422`` (even with ``force``) before any GPU claim or job write: the
    trainer can never build an unknown preset."""
    error = augmentation_preset_error(augmentation)
    if error is not None:
        raise RequestRejectedError(
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
        raise rejection(422, 'empty_val_split', row.message)


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
        raise rejection(
            422,
            'export_outside_project',
            f"dataset_export_dir is outside the export root of project '{cfg.project_slug}'",
            project=cfg.project_slug,
        )


def _default_export_dir_check(spec: TrainJobSpec) -> PreflightCheck | None:
    """A null ``dataset_export_dir`` resolves to the same ``current`` symlink target
    ``GET /export/status`` reports, mutating ``spec`` in place so every later check
    (and, on /start, the ``job.json`` write) sees the resolved path as if the caller
    had passed it. ``None`` when the caller gave one."""
    if spec.dataset_export_dir:
        return None
    from src.services.curation.export import resolve_current_export_dir

    try:
        spec.dataset_export_dir = str(resolve_current_export_dir())
    except FileNotFoundError:
        return PreflightCheck(
            name='dataset_export_dir',
            severity='block',
            message=(
                'no dataset_export_dir was given and no export exists yet '
                '(data/exports/current). Run POST /export/yolo first, or '
                'pass dataset_export_dir explicitly.'
            ),
        )
    return PreflightCheck(
        name='dataset_export_dir',
        severity='ok',
        message=f'defaulted to the current export: {spec.dataset_export_dir}',
    )


def _optimizer_check(spec: TrainJobSpec) -> PreflightCheck:
    optimizer = (spec.hyperparameters or {}).get('optimizer')
    if isinstance(optimizer, str) and optimizer.lower() in {
        o.lower() for o in RESERVED_OPTIMIZERS_YOLO26
    }:
        return PreflightCheck(
            name='optimizer_not_auto',
            severity='block',
            message=(
                "optimizer='auto' is reserved on YOLO26 (Ultralytics issue "
                '#23696). Use MuSGD explicitly.'
            ),
            detail={'optimizer': optimizer},
        )
    return PreflightCheck(
        name='optimizer_not_auto',
        severity='ok',
        message=f'optimizer={optimizer or "MuSGD (default)"} is allowed',
    )


def _augmentation_check(spec: TrainJobSpec) -> PreflightCheck:
    """The augmentation preset is one the trainer can build."""
    preset_error = augmentation_preset_error(spec.augmentation)
    if preset_error is not None:
        return PreflightCheck(
            name='augmentation_preset',
            severity='block',
            message=preset_error,
            detail={
                'preset': spec.augmentation.preset if spec.augmentation else None,
                'valid_presets': list(PRESET_IDS),
            },
        )
    enabled = spec.augmentation is not None and spec.augmentation.enabled
    return PreflightCheck(
        name='augmentation_preset',
        severity='ok',
        message=(
            f'augmentation preset {spec.augmentation.preset!r} is available'
            if enabled and spec.augmentation is not None
            else 'augmentation disabled; no preset to check'
        ),
    )


def _free_disk_check(spec: TrainJobSpec) -> PreflightCheck:
    disk_check_path = resolve_disk_check_path(spec)
    if not training_volume_mount_sane(disk_check_path):
        return PreflightCheck(
            name='free_disk',
            severity='block',
            message=(
                f'{disk_check_path} appears to be on the same filesystem as '
                "the container's own root — cannot see the trainer data "
                'volume. Check the bind mount before training.'
            ),
            detail={'path': disk_check_path},
        )
    free = free_gb(disk_check_path)
    if free is None:
        return PreflightCheck(
            name='free_disk',
            severity='unknown',
            message=f'could not stat {disk_check_path} for free disk space',
            detail={'path': disk_check_path},
        )
    if free < MIN_FREE_DISK_GB:
        return PreflightCheck(
            name='free_disk',
            severity='block',
            message=(
                f'Only {free:.1f} GB free on {disk_check_path} — need '
                f'≥{MIN_FREE_DISK_GB} GB. Free space before training.'
            ),
            detail={'free_gb': round(free, 1), 'required_gb': MIN_FREE_DISK_GB},
        )
    return PreflightCheck(
        name='free_disk', severity='ok', message=f'{free:.1f} GB free on {disk_check_path}'
    )


async def _trainer_reachable_check() -> PreflightCheck:
    """Without this, /start writes job.json and the run sits in ``queued``
    forever with no error if the configured trainer container was never started."""
    severity, detail = await probe_trainer_reachable()
    return PreflightCheck(name='trainer_reachable', severity=severity, message=detail)


def _gpu_arbiter_check(spec: TrainJobSpec) -> PreflightCheck | None:
    """The GPU arbiter can actually stop what this claim requires.

    ``containers_to_stop()`` names real GPU-resident containers (e.g. a large
    vLLM/Triton process) this run's GPU claim must free. If the docker SDK/socket
    isn't usable from this container, ``claim_gpus_for_training`` cannot stop them
    -- proceeding would start training right next to that service on the same GPU.
    Blocking here (not just at /start) means the form surfaces the failure before
    the user submits.
    """
    required_stop_names = containers_to_stop(spec.cuda_visible_devices)
    if not required_stop_names:
        return None
    if not docker_client_available():
        return PreflightCheck(
            name='gpu_arbiter',
            severity='block',
            message=(
                f'docker SDK/socket unavailable in the API container -- cannot stop '
                f'{", ".join(required_stop_names)} for this GPU claim'
            ),
        )
    return PreflightCheck(
        name='gpu_arbiter',
        severity='ok',
        message=f'can stop: {", ".join(required_stop_names)}',
    )


def _trainer_gpus_check(spec: TrainJobSpec) -> PreflightCheck:
    """The trainer is actually attached to the requested host GPU(s).

    ``OP_GPU_ALLOWED_IDS`` only says a host id is *policy-permitted*; it says
    nothing about which container is physically pinned to it. Without this, a
    request for a GPU the trainer container was never attached to (device_ids /
    ``OP_TRAIN_GPU_ORDER`` drift, or a GPU-policy change that forgot to redeploy
    the trainer) writes job.json and sits in ``queued``/``starting`` forever with
    no error. The trainer publishes its attachment at startup
    (``write_trainer_capabilities`` in docker/trainer/gpu_select.py); an absent file
    (older image) only downgrades this to a warning.
    """
    trainer_gpu_order = read_trainer_gpu_order()
    requested_ids = parse_cuda_visible_devices(spec.cuda_visible_devices)
    if trainer_gpu_order is None:
        return PreflightCheck(
            name='trainer_gpus',
            severity='warn',
            message=(
                'no trainer capabilities file found -- cannot verify the trainer '
                'is attached to the requested GPU(s) (older trainer image?)'
            ),
        )
    if trainer_gpu_order and (
        unattached := [i for i in requested_ids if i not in trainer_gpu_order]
    ):
        return PreflightCheck(
            name='trainer_gpus',
            severity='block',
            message=f'trainer is attached to host GPUs {trainer_gpu_order}; requested {unattached}',
            detail={'trainer_gpu_order': trainer_gpu_order, 'requested': requested_ids},
        )
    return PreflightCheck(
        name='trainer_gpus',
        severity='ok',
        message=(
            f'trainer is attached to host GPUs {trainer_gpu_order}'
            if trainer_gpu_order
            else 'trainer reports no GPU restriction (attached to every GPU at its host index)'
        ),
    )


async def _active_run_check() -> PreflightCheck:
    try:
        active = await train_jobs.get_active_job()
    except RuntimeError as exc:
        return PreflightCheck(name='active_run', severity='block', message=str(exc))
    if active is not None and active.state in ('queued', 'starting', 'running', 'exporting'):
        return PreflightCheck(
            name='active_run',
            severity='block',
            message=(
                f'Another run is in progress (job_id={active.job_id}, '
                f'state={active.state}). Cancel or wait for it to finish.'
            ),
            detail={'job_id': active.job_id, 'state': active.state},
        )
    return PreflightCheck(
        name='active_run', severity='ok', message='No other training run in progress'
    )


def _include_classes_check(spec: TrainJobSpec, include_classes: list[int]) -> PreflightCheck:
    """``include_classes`` all resolve against this export. An unresolvable id
    (deprecated, typo, or a class this export never had) used to surface as a 500
    or a job that failed minutes into a run; it is caught here, at the API
    boundary. Not run for single-class jobs, which have no ``include_classes``."""
    unresolvable = unresolvable_include_classes(spec.dataset_export_dir, include_classes)
    if unresolvable:
        return PreflightCheck(
            name='include_classes_resolvable',
            severity='block',
            message=(
                f'include_classes has {len(unresolvable)} id(s) not in this '
                f"export's active class set (deprecated, typo, or never "
                f'exported): {unresolvable}'
            ),
            detail={'unresolvable_class_ids': unresolvable},
        )
    return PreflightCheck(
        name='include_classes_resolvable',
        severity='ok',
        message=f'All {len(include_classes)} include_classes ids resolve in this export',
    )


def _class_name(registry: Any, class_id: int) -> str:
    entry = registry.get(class_id)
    return entry.class_name if entry else f'class_{class_id}'


def _class_balance_check(
    target_classes: list[int], counts: dict[int, int], registry: Any
) -> PreflightCheck:
    below_hard: list[dict[str, Any]] = []
    below_recommended: list[dict[str, Any]] = []
    for cid in target_classes:
        n = counts.get(cid, 0)
        row = {'class_id': cid, 'name': _class_name(registry, cid), 'validated': n}
        if n < HARD_MIN_CROPS_PER_CLASS:
            below_hard.append(row)
        elif n < WARN_MIN_CROPS_PER_CLASS:
            below_recommended.append(row)
    if below_hard:
        names = ', '.join(f'{r["name"]} ({r["validated"]})' for r in below_hard)
        return PreflightCheck(
            name='class_balance',
            severity='block',
            message=(
                f'{len(below_hard)} class(es) below hard floor of '
                f'{HARD_MIN_CROPS_PER_CLASS} validated crops: {names}'
            ),
            detail={'classes': below_hard},
        )
    if below_recommended:
        names = ', '.join(f'{r["name"]} ({r["validated"]})' for r in below_recommended)
        return PreflightCheck(
            name='class_balance',
            severity='warn',
            message=(
                f'{len(below_recommended)} class(es) below recommended '
                f'{WARN_MIN_CROPS_PER_CLASS}: {names}'
            ),
            detail={'classes': below_recommended},
        )
    return PreflightCheck(
        name='class_balance',
        severity='ok',
        message=(
            f'All {len(target_classes)} classes have ≥{WARN_MIN_CROPS_PER_CLASS} validated crops'
        ),
    )


def _test_holdout_check(
    target_classes: list[int], test_counts: dict[int, int], registry: Any
) -> PreflightCheck:
    thin: list[dict[str, Any]] = []
    for cid in target_classes:
        n = test_counts.get(cid, 0)
        if n < MIN_TEST_CROPS_PER_CLASS:
            thin.append({'class_id': cid, 'name': _class_name(registry, cid), 'test_crops': n})
    if thin:
        names = ', '.join(f'{r["name"]} ({r["test_crops"]})' for r in thin)
        return PreflightCheck(
            name='test_holdout',
            severity='block' if any(r['test_crops'] == 0 for r in thin) else 'warn',
            message=(
                f'{len(thin)} class(es) below {MIN_TEST_CROPS_PER_CLASS} '
                f'test crops: {names}. Refreeze the test holdout.'
            ),
            detail={'classes': thin},
        )
    return PreflightCheck(
        name='test_holdout',
        severity='ok',
        message=f'All classes have ≥{MIN_TEST_CROPS_PER_CLASS} test crops',
    )


async def _multi_class_data_checks(
    opensearch: Any, target_classes: list[int]
) -> list[PreflightCheck]:
    """Per-class validated counts and test-holdout coverage (one search covers both)."""
    if not target_classes:
        # Test-holdout is still emitted so the UI can render the row.
        return [
            PreflightCheck(
                name='class_balance',
                severity='block',
                message='No classes selected and registry is empty',
            ),
            PreflightCheck(
                name='test_holdout',
                severity='block',
                message='Cannot evaluate test holdout — no target classes resolved',
            ),
        ]
    counts, test_counts = await count_validated_and_test_per_class(opensearch, target_classes)
    registry = get_class_registry()
    return [
        _class_balance_check(target_classes, counts, registry),
        _test_holdout_check(target_classes, test_counts, registry),
    ]


def _export_readiness_inputs(
    manifest: dict[str, Any], spec: TrainJobSpec, *, is_single_class: bool
) -> list[tuple[str, tuple[str, str, dict[str, Any]]]]:
    """Export readiness: not empty, trainable splits, built from the current index.

    Per-class coverage and the unlabeled-object counts are multi-class only: a
    single-class export has one target class, which the overall train/val check
    already covers, and only the multi-class exporter records the unlabeled counts.
    """
    class_split_result = (
        (
            'ok',
            (
                'not applicable for this dataset kind (single-class: covered by '
                'export_splits_nonempty)'
            ),
            {},
        )
        if is_single_class
        else export_class_split_check(manifest, spec.include_classes)
    )
    unlabeled_result = (
        ('ok', 'not applicable for this dataset kind (single-class)', {})
        if is_single_class
        else export_unlabeled_objects_check(manifest)
    )
    return [
        ('export_not_empty', export_size_check(manifest)),
        ('export_splits_nonempty', export_splits_check(manifest)),
        ('export_class_split_coverage', class_split_result),
        ('export_unlabeled_objects', unlabeled_result),
    ]


async def _export_readiness_checks(
    manifest: dict[str, Any], spec: TrainJobSpec, opensearch: Any, *, is_single_class: bool
) -> list[PreflightCheck]:
    results = _export_readiness_inputs(manifest, spec, is_single_class=is_single_class)
    results.append(
        (
            'export_generation',
            export_generation_check(
                manifest, await items_index_generation(opensearch, items_index())
            ),
        )
    )
    return [
        PreflightCheck(name=name, severity=severity, message=message, detail=detail or None)
        for name, (severity, message, detail) in results
    ]


def _unregistered_class_ids_check(manifest: dict[str, Any]) -> PreflightCheck | None:
    """Validated items the export dropped for an unregistered class id."""
    dropped = manifest.get('dropped_unregistered_class_ids')
    if not isinstance(dropped, dict):
        return None
    total = sum(int(v) for v in dropped.values())
    return PreflightCheck(
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


def _read_manifest(spec: TrainJobSpec, checks: list[PreflightCheck]) -> dict[str, Any]:
    """The export manifest ({} when unreadable, with a blocking row). A single-class
    export validates from its own manifest (disk dataset), not the multi-class
    registry's ``class_validated`` counts."""
    try:
        return read_export_manifest(spec.dataset_export_dir)
    except ExportManifestUnreadableError as exc:
        checks.append(PreflightCheck(name='export_manifest', severity='block', message=str(exc)))
        return {}


async def _data_checks(
    spec: TrainJobSpec,
    opensearch: Any,
    manifest: dict[str, Any],
    checks: list[PreflightCheck],
    *,
    is_single_class: bool,
) -> None:
    """Data sufficiency (single-class-aware), then the label scan."""
    target_classes = [] if is_single_class else resolve_target_classes(spec)
    if not is_single_class and spec.include_classes:
        checks.append(_include_classes_check(spec, spec.include_classes))
    if is_single_class:
        append_single_class_data_checks(checks, manifest)
    else:
        checks.extend(await _multi_class_data_checks(opensearch, target_classes))
    append_label_scan_checks(checks, spec, manifest, is_single_class)


async def run_preflight(spec: TrainJobSpec, opensearch: Any) -> PreflightReport:
    """Run every preflight check and return the bundled report."""
    checks: list[PreflightCheck] = []
    if (default_dir := _default_export_dir_check(spec)) is not None:
        checks.append(default_dir)
    if spec.dataset_export_dir:
        _refuse_export_outside_project(spec.dataset_export_dir)

    checks.append(_optimizer_check(spec))
    checks.append(_augmentation_check(spec))
    checks.append(_free_disk_check(spec))
    checks.append(await _trainer_reachable_check())
    if (arbiter := _gpu_arbiter_check(spec)) is not None:
        checks.append(arbiter)
    checks.append(_trainer_gpus_check(spec))
    checks.append(await _active_run_check())

    manifest = _read_manifest(spec, checks)
    is_single_class = manifest.get('dataset_kind') in SINGLE_CLASS_DATASET_KINDS
    await _data_checks(spec, opensearch, manifest, checks, is_single_class=is_single_class)
    checks.extend(
        await _export_readiness_checks(manifest, spec, opensearch, is_single_class=is_single_class)
    )
    if (unregistered := _unregistered_class_ids_check(manifest)) is not None:
        checks.append(unregistered)
    # Claiming the GPUs stops GPU-scoped ingest containers: warn when ingest is active.
    if needs_service_stop(spec.cuda_visible_devices):
        stopped = containers_to_stop(spec.cuda_visible_devices)
        checks.append(ingest_idle_check(await count_pending_ingest(opensearch), stopped))

    blocked = any(c.severity == 'block' for c in checks)
    summary = (
        'BLOCKED — fix the highlighted checks before training'
        if blocked
        else 'All checks passed (warnings allowed)'
    )
    return PreflightReport(blocked=blocked, checks=checks, summary=summary)
