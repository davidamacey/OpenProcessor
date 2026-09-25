"""Training-pipeline router.

Mounts at ``{api_prefix}/train`` and exposes the ten endpoints the labeler
frontend's train page consumes. The router is intentionally thin: every file-system
write/read goes through :mod:`src.services.training.jobs`, and every
hyperparameter table lives in :mod:`src.services.training.profiles`.

Endpoints:

    POST   {api_prefix}/train/preflight          → PreflightReport
    POST   {api_prefix}/train/start              → {job_id}        (writes job.json)
    POST   {api_prefix}/train/start_campaign     → {campaign_id, job_ids}
    GET    {api_prefix}/train/status             → most recent TrainJobStatus
    GET    {api_prefix}/train/status/{job_id}    → TrainJobStatus
    GET    {api_prefix}/train/runs               → paginated list
    GET    {api_prefix}/train/log/tail/{job_id}  → tail N lines
    POST   {api_prefix}/train/cancel/{job_id}    → drop cancel sentinel
    POST   {api_prefix}/train/cancel_campaign/{campaign_id}
    GET    {api_prefix}/train/profiles           → profile table
    GET    {api_prefix}/train/presets            → class-subset presets
    GET    {api_prefix}/train/augmentation_presets → augmentation preset catalog
    GET    {api_prefix}/train/gpus               → TrainGpuOptionsResponse
    GET    {api_prefix}/train/manifest/{job_id}   → run lineage manifest
    GET    {api_prefix}/train/artifacts/{job_id}/{name}
        → whitelisted run artifact (confusion_matrix.png, results.csv, ...)

Pre-flight contract: ``/start`` calls ``/preflight``
internally and refuses to write ``job.json`` if any check has severity
``block``. Pass ``?force=true`` to bypass; the report is still returned in
the 422 body so the UI can render it inline.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Path as PathParam, Query, status
from fastapi.responses import FileResponse, ORJSONResponse
from pydantic import BaseModel, Field

from src.config import (
    IndexRole,
    get_curation_config,
    get_gpu_arbiter_config,
    get_region_fields,
    index_name,
)
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.routers.curation import get_class_registry
from src.routers.curation._common import OpenSearchDep  # noqa: TC001 - used at runtime by FastAPI
from src.services.curation.dataset_thresholds import (
    HARD_MIN_CROPS_PER_CLASS,
    MIN_TEST_CROPS_PER_CLASS,
    WARN_MIN_CROPS_PER_CLASS,
    dataset_thresholds,
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
from src.services.training.augmentation_presets import (
    AUGMENTATION_PRESETS,
    DEFAULT_AUGMENTATION_PRESET,
    PRESET_IDS,
    unknown_preset_error,
)
from src.services.training.gpu_arbiter import (
    GpuArbiterStopFailedError,
    containers_to_stop,
    docker_client_available,
    needs_service_stop,
    parse_cuda_visible_devices,
    probe_trainer_reachable,
)
from src.services.training.jobs import (
    AugmentationSpec,
    Profile,
    TrainCampaignSpec,
    TrainJobSpec,
    TrainJobStatus,
)
from src.services.training.profiles import (
    PROFILES_YOLO26,
    RESERVED_OPTIMIZERS_YOLO26,
    get_class_subset_presets,
    get_profiles,
)


logger = get_logger(__name__)


config = get_curation_config()
F = get_region_fields()
CURATION_ITEMS_INDEX = index_name(config, IndexRole.ITEMS)

router = APIRouter(
    prefix=f'{config.api_prefix}/train',
    tags=[f'{config.api_tag} - Train'],
    default_response_class=ORJSONResponse,
)


# =============================================================================
# Pre-flight model
# =============================================================================


PreflightSeverity = Literal['ok', 'warn', 'block', 'unknown']


class PreflightCheck(BaseModel):
    """Single row in the preflight report."""

    name: str
    severity: PreflightSeverity
    message: str
    detail: dict[str, Any] | None = None


class PreflightReport(BaseModel):
    """Bundled preflight result the frontend renders inline on the form."""

    blocked: bool
    checks: list[PreflightCheck]
    summary: str = ''
    # The per-class cut points the class_balance / test_holdout checks use.
    thresholds: dict[str, int] = Field(default_factory=dataset_thresholds)


# Disk-space threshold (≥50 GB free on the training-data volume).
MIN_FREE_DISK_GB = 50

# Per-class crop minimums live in dataset_thresholds.py,
# which also serves them to clients.


# =============================================================================
# Helpers
# =============================================================================


async def _count_validated_and_test_per_class(
    opensearch: Any,
    class_ids: list[int],
) -> tuple[dict[int, int], dict[int, int]]:
    """Return ``({class_id: validated_crop_count}, {class_id: test_holdout_count})``.

    Preflight used to issue these as two separate ``_search`` round trips
    (identical ``class_id`` scope, one with an extra ``test_holdout``
    filter) -- merged into one ``_search`` with two sibling ``filter``
    aggs, each with its own ``by_class`` terms sub-agg, since both share
    the same base document set.

    The items-index schema uses a boolean ``label_validated`` field (set
    true by both human-confirmation and auto-promotion) as the source of
    truth; both human and auto-promoted labels count as eligible training
    data. Test-holdout coverage (≥5 per class) is a subset of validated
    crops.
    """
    if not class_ids:
        return {}, {}
    size = max(len(class_ids), 1)
    body = {
        'size': 0,
        'query': {'bool': {'filter': [{'terms': {'class_id': class_ids}}]}},
        'aggs': {
            'validated_by_class': {
                'filter': {'term': {'class_validated': True}},
                'aggs': {'by_class': {'terms': {'field': 'class_id', 'size': size}}},
            },
            'test_by_class': {
                'filter': {
                    'bool': {
                        'filter': [
                            {'term': {'class_validated': True}},
                            {'term': {'test_holdout': True}},
                        ]
                    }
                },
                'aggs': {'by_class': {'terms': {'field': 'class_id', 'size': size}}},
            },
        },
    }
    empty = dict.fromkeys(class_ids, 0)
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        logger.warning('train_class_count_failed', error=str(exc))
        return dict(empty), dict(empty)
    aggs = resp.get('aggregations') or {}

    def _by_class(agg_name: str) -> dict[int, int]:
        counts = dict(empty)
        for bucket in (aggs.get(agg_name) or {}).get('by_class', {}).get('buckets', []):
            cid = bucket.get('key')
            if isinstance(cid, int):
                counts[cid] = int(bucket.get('doc_count', 0))
        return counts

    return _by_class('validated_by_class'), _by_class('test_by_class')


async def _count_pending_ingest(opensearch: Any) -> int:
    """Count crops still awaiting region detection/verification.

    A training claim that stops GPU-resident ingest containers
    (``gpu_arbiter.containers_to_stop``) pauses that detection/verification
    until the run ends. This lets preflight warn the operator how much
    in-flight ingest that will stall. Legacy status names included so a
    mid-migration backlog is still counted.
    """
    body = {
        'query': {
            'terms': {
                F.status: [
                    RegionStatus.PENDING_DETECTION,
                    RegionStatus.PENDING_VERIFICATION,
                    'pending',
                    'pending_verify',
                ]
            }
        }
    }
    try:
        resp = await opensearch.count(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        logger.warning('train_pending_count_failed', error=str(exc))
        return 0
    return int(resp.get('count', 0))


def _resolve_target_classes(spec: TrainJobSpec) -> list[int]:
    """Resolve the effective class list for a spec.

    ``include_classes=None`` → every non-deprecated class in the registry.
    """
    if spec.include_classes:
        return list(spec.include_classes)
    registry = get_class_registry()
    return [c.class_id for c in registry.load().classes if not c.deprecated]


def _augmentation_preset_error(augmentation: AugmentationSpec | None) -> str | None:
    """Error for an enabled augmentation block naming an unknown preset.

    A disabled block's preset is never built by the trainer, so it isn't
    judged. Checked by preflight and, ahead of every side effect, by
    ``/start`` and ``/start_campaign``.
    """
    if augmentation is None or not augmentation.enabled:
        return None
    return unknown_preset_error(augmentation.preset)


def _refuse_unknown_augmentation_preset(augmentation: AugmentationSpec | None) -> None:
    """``422`` (even with ``force``) before any GPU claim or job write: the
    trainer can never build an unknown preset."""
    error = _augmentation_preset_error(augmentation)
    if error is not None:
        raise HTTPException(
            status_code=422,
            detail={
                'message': error,
                'field': 'augmentation.preset',
                'valid_presets': list(PRESET_IDS),
            },
        )


def _free_gb(path: str) -> float | None:
    """Free disk space on ``path``'s filesystem, in GB.

    This used to fail OPEN on any ``OSError`` (return ``float('inf')``,
    i.e. "infinite free space") — the exact opposite of a safe default.
    ``None`` now means "couldn't determine", which the caller reports as
    ``severity='unknown'``, never ``'ok'``.
    """
    try:
        usage = shutil.disk_usage(_nearest_existing(path))
    except OSError:
        return None
    return usage.free / (1024**3)


def _nearest_existing(path: str) -> Path:
    """``path`` or its closest existing ancestor -- a staging dir on a fresh
    volume doesn't exist until the first run, but its mount does. Raises
    ``OSError`` for anything other than a missing component."""
    for candidate in (Path(path), *Path(path).parents):
        try:
            candidate.stat()
        except FileNotFoundError:
            continue
        return candidate
    return Path('/')


def _resolve_disk_check_path(spec: TrainJobSpec) -> str:
    """Pick the path to stat for the free-disk check.

    Previously hardcoded to a host data-volume root — inside the yolo-api
    container, only specific subpaths under that root (e.g. a
    deployment-specific training-data mount, see the deployment's own
    compose overlay) are bind-mounted from the real training-data volume;
    the root itself resolves to the container's own overlay filesystem, which
    almost always has plenty of headroom regardless of whether the real
    training volume is anywhere near full. Prefer the export dir itself
    (guaranteed to be on the real volume once a job names one) and fall
    back to ``OP_TRAIN_STAGING`` (the same env var the trainer/API compose
    services already use for the training-data root).
    """
    if spec.dataset_export_dir and Path(spec.dataset_export_dir).exists():
        return str(spec.dataset_export_dir)
    return os.environ.get('OP_TRAIN_STAGING', str(config.state_dir / 'training_staging'))


def _training_volume_mount_sane(path: str) -> bool:
    """False if ``path`` is on the same device as ``/``.

    A strong signal the real training-data volume isn't actually mounted
    into this container at ``path`` — e.g. a dev box or misconfigured
    compose file where the bind mount silently didn't take, leaving
    ``path`` resolving to the container's own root filesystem. A path that
    doesn't exist yet (fresh volume, staging dir not created) is judged by
    its nearest existing ancestor. Any other ``OSError`` is treated as
    "can't confirm it's sane" (False), not a soft pass.
    """
    try:
        existing = _nearest_existing(path)
        if existing == Path('/'):
            return False
        return existing.stat().st_dev != Path('/').stat().st_dev
    except OSError:
        return False


# Written by docker/trainer/trainer.py's write_trainer_capabilities() at
# startup, next to job.json in the shared /jobs volume. Kept in sync with
# TRAINER_CAPABILITIES_FILENAME there -- there is no shared import (the
# trainer ships as its own image with no src/ dependency), so
# test_trainer_protocol.py conformance-tests both halves against the
# literal name.
TRAINER_CAPABILITIES_FILENAME = '.trainer_capabilities.json'


def _read_trainer_capabilities() -> dict[str, Any] | None:
    """The trainer's published ``gpu_order``/``visible_count``/``build_sha``.

    ``None`` (not an error) when the file is absent -- an older trainer
    image that predates this fix, or a container that hasn't started yet.
    Callers must treat that as "can't verify" (a warning), not "no GPUs
    attached" (which would incorrectly block every request).
    """
    path = train_jobs._resolve_jobs_dir() / TRAINER_CAPABILITIES_FILENAME
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _trainer_gpu_order() -> list[int] | None:
    """``gpu_order`` from the trainer capabilities file, or ``None`` if
    unknown (missing file, or a non-empty-but-unparseable field)."""
    caps = _read_trainer_capabilities()
    if caps is None:
        return None
    order = caps.get('gpu_order')
    if not isinstance(order, list) or not all(isinstance(i, int) for i in order):
        return None
    return order


def _read_export_manifest(dataset_export_dir: str | None) -> dict[str, Any]:
    """Best-effort read of an export's ``manifest.json`` (``{}`` on any failure)."""
    if not dataset_export_dir:
        return {}
    import json
    from pathlib import Path

    try:
        return json.loads((Path(dataset_export_dir) / 'manifest.json').read_text())
    except Exception:
        return {}


def _unresolvable_include_classes(
    dataset_export_dir: str | None, include_classes: list[int]
) -> list[int]:
    """Return the subset of ``include_classes`` this export can't resolve.

    Mirrors ``docker/trainer/subset_dataset.py::_read_export_id_map``'s own
    "unknown" guard, just run at API preflight time instead of minutes into
    a trainer-container run. If the export's ``class_registry.json`` or its
    ``export_id_map`` is missing/unreadable, every requested id is reported
    unresolvable — that's a real blocker (an export without a Phase 5
    dense-id map can't be subset-trained at all), not a soft skip.
    """
    if not dataset_export_dir:
        return list(include_classes)
    import json
    from pathlib import Path

    try:
        payload = json.loads((Path(dataset_export_dir) / 'class_registry.json').read_text())
    except Exception:
        return list(include_classes)
    export_id_map = payload.get('export_id_map')
    if not isinstance(export_id_map, dict):
        return list(include_classes)
    return [c for c in include_classes if str(c) not in export_id_map]


# Export manifests whose data sufficiency must be judged from the manifest
# itself rather than the multi-class registry. ``single_class`` is what
# :mod:`src.services.curation.export_single_class` writes. The retired
# ``lpr_single_class`` alias is not accepted: a manifest carrying it
# must be re-exported, not silently treated as single-class.
SINGLE_CLASS_DATASET_KINDS: frozenset[str] = frozenset({'single_class'})


def _single_class_label(manifest: dict[str, Any]) -> str:
    """Human-readable name of a single-class export's target class."""
    name = manifest.get('class_name')
    if name:
        return str(name)
    names = manifest.get('class_names')
    if isinstance(names, list) and names:
        return ', '.join(str(n) for n in names)
    return 'target class'


def _append_single_class_data_checks(
    checks: list[PreflightCheck], manifest: dict[str, Any]
) -> None:
    """Single-class data-sufficiency checks read from the export manifest.

    A narrowed export's labels live on disk, and the multi-class
    ``class_validated`` counts for its target class are typically ~0 — which
    would wrongly block. Validate from the manifest's positive + test-split
    counts instead.
    """
    label = _single_class_label(manifest)
    pos = int(manifest.get('positive_images') or 0)
    if pos < HARD_MIN_CROPS_PER_CLASS:
        checks.append(
            PreflightCheck(
                name='class_balance',
                severity='block',
                message=(
                    f'{label} has {pos} labeled positive frames '
                    f'(< hard floor {HARD_MIN_CROPS_PER_CLASS})'
                ),
                detail={'positive_images': pos},
            )
        )
    elif pos < WARN_MIN_CROPS_PER_CLASS:
        checks.append(
            PreflightCheck(
                name='class_balance',
                severity='warn',
                message=(
                    f'{label} has {pos} labeled positive frames (<{WARN_MIN_CROPS_PER_CLASS})'
                ),
                detail={'positive_images': pos},
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name='class_balance',
                severity='ok',
                message=f'{label}: {pos} labeled positive frames',
            )
        )

    test_n = int((manifest.get('split_counts') or {}).get('test') or 0)
    if test_n < MIN_TEST_CROPS_PER_CLASS:
        checks.append(
            PreflightCheck(
                name='test_holdout',
                severity='block' if test_n == 0 else 'warn',
                message=(
                    f'Single-class test split has {test_n} frames '
                    f'(<{MIN_TEST_CROPS_PER_CLASS}). Refreeze the test holdout.'
                ),
                detail={'test_frames': test_n},
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name='test_holdout',
                severity='ok',
                message=f'Single-class test split: {test_n} frames',
            )
        )


# =============================================================================
# Pre-flight implementation
# =============================================================================


async def _run_preflight(
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
    preset_error = _augmentation_preset_error(spec.augmentation)
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
    disk_check_path = _resolve_disk_check_path(spec)
    if not _training_volume_mount_sane(disk_check_path):
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
        free = _free_gb(disk_check_path)
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
    trainer_gpu_order = _trainer_gpu_order()
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
    _single_class_manifest = _read_export_manifest(spec.dataset_export_dir)
    _is_single_class = _single_class_manifest.get('dataset_kind') in SINGLE_CLASS_DATASET_KINDS
    target_classes = [] if _is_single_class else _resolve_target_classes(spec)

    # ---- 3b. include_classes resolvable against this export ----------
    # Previously an unresolvable include_classes id (deprecated, typo, or a
    # class this export never had) surfaced as either a 500 or a job that
    # failed deep inside the trainer container (subset_dataset.py's own
    # ValueError, minutes into a run). Catch it here instead, at the API
    # boundary, with a structured preflight check. Skipped entirely for single-class
    # jobs (single-class, no include_classes concept).
    if not _is_single_class and spec.include_classes:
        unresolvable = _unresolvable_include_classes(spec.dataset_export_dir, spec.include_classes)
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
        _append_single_class_data_checks(checks, _single_class_manifest)
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
        counts, test_counts = await _count_validated_and_test_per_class(opensearch, target_classes)
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

    # ---- 6. empty-label / region-pairing (real scan, not a stub) --------
    # Both used to be hardcoded to 'ok' with no scan ever run. Single-class
    # exports are handled by their own additive branch — background/negative
    # frames are a legitimate, expected empty-label case there (accounted
    # for via the manifest's own counts), and there are no parent item
    # boxes to pair against by construction.
    if _is_single_class:
        positive = int(_single_class_manifest.get('positive_images') or 0)
        total_single_class_images = int(_single_class_manifest.get('total_images') or 0) or None
        background_note = (
            f' ({total_single_class_images - positive} background/negative frames)'
            if total_single_class_images is not None
            else ''
        )
        checks.append(
            PreflightCheck(
                name='empty_labels',
                severity='ok',
                message=(
                    f'single-class export: {positive} positive frames{background_note} — '
                    'background/negative frames are expected here, not scanned as '
                    "'empty labels'"
                ),
            )
        )
        checks.append(
            PreflightCheck(
                name='region_pairing',
                severity='ok',
                message=(
                    'not applicable for this dataset kind (a single-class export '
                    'has no parent item boxes by construction)'
                ),
            )
        )
    else:
        from src.services.training.preflight_scan import scan_export_labels

        scan = (
            scan_export_labels(Path(spec.dataset_export_dir), include_classes=spec.include_classes)
            if spec.dataset_export_dir
            else None
        )
        if scan is None or scan.status == 'unknown':
            reason = scan.reason if scan else 'no dataset_export_dir on this spec'
            checks.append(
                PreflightCheck(
                    name='empty_labels',
                    severity='unknown',
                    message=f'could not scan label files: {reason}',
                )
            )
            checks.append(
                PreflightCheck(
                    name='region_pairing',
                    severity='unknown',
                    message=f'could not scan label files: {reason}',
                )
            )
        else:
            if scan.total_images > 0 and scan.empty_label_images == scan.total_images:
                checks.append(
                    PreflightCheck(
                        name='empty_labels',
                        severity='block',
                        message=(
                            f'every one of {scan.total_images} images has 0 label rows '
                            'after the include_classes filter — 0 training rows would '
                            'survive'
                        ),
                        detail={
                            'total_images': scan.total_images,
                            'empty_label_images': scan.empty_label_images,
                        },
                    )
                )
            elif scan.empty_label_images > 0:
                checks.append(
                    PreflightCheck(
                        name='empty_labels',
                        severity='warn',
                        message=(
                            f'{scan.empty_label_images}/{scan.total_images} images have '
                            '0 label rows after the include_classes filter'
                        ),
                        detail={
                            'total_images': scan.total_images,
                            'empty_label_images': scan.empty_label_images,
                        },
                    )
                )
            else:
                checks.append(
                    PreflightCheck(
                        name='empty_labels',
                        severity='ok',
                        message=f'all {scan.total_images} images have at least one label row',
                    )
                )

            if scan.region_boxes == 0:
                checks.append(
                    PreflightCheck(
                        name='region_pairing',
                        severity='ok',
                        message='no region boxes in this export/subset',
                    )
                )
            elif scan.unpaired_region_boxes > 0:
                checks.append(
                    PreflightCheck(
                        name='region_pairing',
                        severity='warn',
                        message=(
                            f'{scan.unpaired_region_boxes}/{scan.region_boxes} region '
                            'boxes have no matching parent item box in the same image'
                        ),
                        detail={
                            'region_boxes': scan.region_boxes,
                            'unpaired_region_boxes': scan.unpaired_region_boxes,
                        },
                    )
                )
            else:
                checks.append(
                    PreflightCheck(
                        name='region_pairing',
                        severity='ok',
                        message=f'all {scan.region_boxes} region boxes are paired',
                    )
                )

    # ---- export readiness: not empty, trainable splits, built from
    # the current index. Per-class coverage is multi-class only: a
    # single-class export has one target class, which the overall
    # train/val check already covers. So are the unlabeled-object counts
    # (export_unlabeled_objects): only the multi-class exporter records them.
    export_manifest = _read_export_manifest(spec.dataset_export_dir)
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
                await items_index_generation(opensearch, CURATION_ITEMS_INDEX),
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
        pending = await _count_pending_ingest(opensearch)
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
    return await _run_preflight(spec, opensearch)


class StartTrainResponse(BaseModel):
    job_id: str
    preflight: PreflightReport


@router.post(
    '/start',
    response_model=StartTrainResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_train(
    spec: TrainJobSpec,
    opensearch: OpenSearchDep,
    force: Annotated[bool, Query(description='Bypass blocking preflight checks')] = False,
) -> StartTrainResponse:
    """Validate, run preflight, and write ``job.json``.

    Returns 422 with the full preflight report if any check is blocking
    and ``force=False``, and 422 for an unknown augmentation preset even
    with ``force`` (before the GPU claim). The trainer picks up the file
    out-of-band.
    """
    _refuse_unknown_augmentation_preset(spec.augmentation)
    report = await _run_preflight(spec, opensearch)
    # Active run gets 409 specifically (precedes the generic 422). Without
    # ``force``, active-run is non-overridable: the trainer only handles
    # one run at a time.
    active_check = next(
        (c for c in report.checks if c.name == 'active_run' and c.severity == 'block'),
        None,
    )
    if active_check is not None and not force:
        raise HTTPException(
            status_code=409,
            detail={'message': active_check.message, 'preflight': report.model_dump()},
        )
    if report.blocked and not force:
        raise HTTPException(
            status_code=422,
            detail={'message': 'preflight blocked', 'preflight': report.model_dump()},
        )
    # GPU arbiter — pause the VLM worker (single-GPU) or stop the container
    # (dual-GPU) BEFORE the trainer picks the job up, and BEFORE job.json
    # is written. Fails closed -- claim_gpus_for_training raises
    # GpuArbiterStopFailedError when a claim needs to stop a configured
    # GPU-resident container and can't (docker SDK/socket unavailable, or
    # the stop itself failed). Starting anyway would run training right
    # next to that service on the same GPU, so refuse with 409 and never
    # reach write_job.
    from src.services.training.gpu_arbiter import claim_gpus_for_training

    try:
        await claim_gpus_for_training(spec.cuda_visible_devices)
    except GpuArbiterStopFailedError as exc:
        logger.warning('gpu_arbiter_claim_failed', error=str(exc))
        raise HTTPException(
            status_code=409,
            detail={
                'message': (
                    'cannot claim the requested GPU(s): a configured GPU-resident '
                    f'container could not be stopped ({exc})'
                ),
            },
        ) from exc
    try:
        job_id = await train_jobs.write_job(spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StartTrainResponse(job_id=job_id, preflight=report)


class StartCampaignResponse(BaseModel):
    campaign_id: str
    job_ids: list[str]


@router.post(
    '/start_campaign',
    response_model=StartCampaignResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_campaign(
    campaign: TrainCampaignSpec,
    opensearch: OpenSearchDep,
    force: Annotated[bool, Query(description='Bypass blocking preflight checks')] = False,
) -> StartCampaignResponse:
    """Submit a multi-size training campaign.

    Preflight runs once on a synthetic spec built from the first run; the
    rest of the runs share the same dataset / class set so a single
    preflight covers them all.
    """
    if not campaign.runs:
        raise HTTPException(status_code=400, detail='campaign requires at least one run')
    _refuse_unknown_augmentation_preset(campaign.augmentation)

    first = campaign.runs[0]
    probe_spec = TrainJobSpec(
        dataset_export_dir=campaign.dataset_export_dir,
        include_classes=campaign.include_classes,
        single_cls=campaign.single_cls,
        cuda_visible_devices=campaign.cuda_visible_devices,
        model_size=first.model_size or 'm',
        profile=first.profile if first.profile in PROFILES_YOLO26 else 'custom',  # type: ignore[arg-type]
        hyperparameters=first.hyperparameters,
        augmentation=campaign.augmentation,
    )
    report = await _run_preflight(probe_spec, opensearch)
    if report.blocked and not force:
        raise HTTPException(
            status_code=422,
            detail={'message': 'preflight blocked', 'preflight': report.model_dump()},
        )

    # GPU arbiter — claim once for the entire campaign. The reconcile loop
    # in src/main.py releases when no run is left in a non-terminal state.
    # Fails closed -- see the matching comment in start_train above.
    from src.services.training.gpu_arbiter import claim_gpus_for_training

    try:
        await claim_gpus_for_training(campaign.cuda_visible_devices)
    except GpuArbiterStopFailedError as exc:
        logger.warning('gpu_arbiter_claim_failed', error=str(exc))
        raise HTTPException(
            status_code=409,
            detail={
                'message': (
                    'cannot claim the requested GPU(s): a configured GPU-resident '
                    f'container could not be stopped ({exc})'
                ),
            },
        ) from exc
    try:
        campaign_id, job_ids = await train_jobs.write_campaign(campaign)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StartCampaignResponse(campaign_id=campaign_id, job_ids=job_ids)


@router.get('/status', response_model=TrainJobStatus | None)
async def latest_status() -> TrainJobStatus | None:
    """Return the active or most-recent run's ``status.json``.

    Returns ``null`` (HTTP 200) if there are no runs yet.
    """
    try:
        return await train_jobs.get_active_job()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get('/status/{job_id}', response_model=TrainJobStatus)
async def status_by_id(
    job_id: Annotated[str, PathParam(description='Training job_id')],
) -> TrainJobStatus:
    """Return the ``status.json`` for a specific run.

    404 if no spec or status exists for that id.
    """
    try:
        result = await train_jobs.read_status(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail=f'job_id not found: {job_id}')
    return result


class RunsListResponse(BaseModel):
    items: list[TrainJobStatus]
    total: int


@router.get('/runs', response_model=RunsListResponse)
async def list_runs_endpoint(
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RunsListResponse:
    """Paginated past-runs list (newest first)."""
    items = await train_jobs.list_runs(limit=limit, offset=offset)
    # ``list_runs`` is page-bounded; for total we re-query a larger chunk.
    # The labeler's infinite-scroll only needs ``len(items)`` to know if
    # there's more — we report that via the ``total`` field.
    deeper = await train_jobs.list_runs(limit=limit + offset + 1, offset=0)
    return RunsListResponse(items=items, total=len(deeper))


class LogTailResponse(BaseModel):
    job_id: str
    lines: list[str]


@router.get('/log/tail/{job_id}', response_model=LogTailResponse)
async def tail_log(
    job_id: Annotated[str, PathParam(description='Training job_id')],
    lines: Annotated[int, Query(ge=1, le=5000)] = 200,
) -> LogTailResponse:
    """Return the last N lines of the run log."""
    try:
        out = await train_jobs.tail_run_log(job_id, lines=lines)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return LogTailResponse(job_id=job_id, lines=out)


class CancelResponse(BaseModel):
    cancelled: bool
    job_id: str


@router.post('/cancel/{job_id}', response_model=CancelResponse)
async def cancel_run(
    job_id: Annotated[str, PathParam(description='Training job_id')],
) -> CancelResponse:
    """Drop the cancel sentinel; trainer picks it up between epochs."""
    try:
        await train_jobs.write_cancel(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CancelResponse(cancelled=True, job_id=job_id)


class CancelCampaignResponse(BaseModel):
    cancelled: int
    campaign_id: str


@router.post('/cancel_campaign/{campaign_id}', response_model=CancelCampaignResponse)
async def cancel_campaign_endpoint(
    campaign_id: Annotated[str, PathParam(description='Campaign id')],
) -> CancelCampaignResponse:
    """Cancel every non-terminal job in a campaign."""
    try:
        n = await train_jobs.cancel_campaign(campaign_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CancelCampaignResponse(cancelled=n, campaign_id=campaign_id)


class ProfilesResponse(BaseModel):
    profiles: list[Profile]


@router.get('/profiles', response_model=ProfilesResponse)
async def list_profiles() -> ProfilesResponse:
    """Return the YOLO26 profile table for the form picker."""
    rows = [Profile(**p) for p in get_profiles()]
    return ProfilesResponse(profiles=rows)


class AugmentationPresetOption(BaseModel):
    """One selectable ``augmentation.preset``."""

    id: str
    label: str
    description: str
    orientation_sensitive: bool = Field(
        description='Horizontal flip is disabled for the whole run with this preset.'
    )


class AugmentationPresetsResponse(BaseModel):
    presets: list[AugmentationPresetOption]
    default: str = Field(description='Preset used when a job omits augmentation.preset.')


@router.get('/augmentation_presets', response_model=AugmentationPresetsResponse)
async def list_augmentation_presets() -> AugmentationPresetsResponse:
    """The augmentation presets the trainer can build, for the form picker.

    Served from the catalog the trainer itself builds from
    (``src/services/training/augmentation_presets.py``); ``/preflight`` and
    ``/start`` reject any other id.
    """
    return AugmentationPresetsResponse(
        presets=[
            AugmentationPresetOption(
                id=p.id,
                label=p.label,
                description=p.description,
                orientation_sensitive=p.orientation_sensitive,
            )
            for p in AUGMENTATION_PRESETS
        ],
        default=DEFAULT_AUGMENTATION_PRESET,
    )


class PresetsResponse(BaseModel):
    class_subset_presets: list[dict[str, Any]] = Field(default_factory=list)


@router.get('/presets', response_model=PresetsResponse)
async def list_presets() -> PresetsResponse:
    """Return server-side class-subset presets."""
    return PresetsResponse(class_subset_presets=get_class_subset_presets())


# =============================================================================
# GET /gpus -- served training GPU picker (backend owns the decisions)
# =============================================================================


class TrainGpuOption(BaseModel):
    """One selectable ``cuda_visible_devices`` value for the train form."""

    value: str
    gpu_ids: list[int]
    label: str
    advisory: str
    stops_containers: list[str] = Field(default_factory=list)
    default: bool = False


class TrainGpuOptionsResponse(BaseModel):
    options: list[TrainGpuOption]
    allowed_ids: list[int]
    unrestricted: bool


def _train_gpu_option_label(gpu_ids: list[int], gpu_labels: dict[int, str]) -> str:
    ids_str = ','.join(str(i) for i in gpu_ids)
    if len(gpu_ids) == 1:
        gid = gpu_ids[0]
        card = gpu_labels.get(gid)
        return f'{card} (GPU {gid})' if card else f'GPU {gid}'
    names = {gpu_labels.get(gid) for gid in gpu_ids}
    if len(names) == 1 and (only := next(iter(names))):
        return f'{len(gpu_ids)}× {only} (GPUs {ids_str})'  # noqa: RUF001 - intentional display glyph
    return f'GPUs {ids_str}'


def _train_gpu_option_advisory(stopped: tuple[str, ...]) -> str:
    if stopped:
        return f'Stops {", ".join(stopped)} for the run; restarted when it ends.'
    return 'Shares GPU(s) with running services; background workers pause for the run.'


def _build_train_gpu_option(gpu_ids: list[int], gpu_labels: dict[int, str]) -> TrainGpuOption:
    value = ','.join(str(i) for i in gpu_ids)
    stopped = containers_to_stop(value)
    return TrainGpuOption(
        value=value,
        gpu_ids=gpu_ids,
        label=_train_gpu_option_label(gpu_ids, gpu_labels),
        advisory=_train_gpu_option_advisory(stopped),
        stops_containers=list(stopped),
    )


@router.get('/gpus', response_model=TrainGpuOptionsResponse)
async def list_train_gpu_options() -> TrainGpuOptionsResponse:
    """Serve the training GPU picker: values, labels, and stop advisories.

    Backend owns these decisions -- the frontend must never hardcode a
    deployment's GPU topology. Unrestricted installs (no
    ``OP_GPU_ALLOWED_IDS``) get exactly one option: the resolved default.

    When the trainer's published ``gpu_order`` (see ``_trainer_gpu_order``)
    is known and non-empty, the option list is intersected with it -- a
    trainer physically attached to only host GPU 2 must not offer GPU 0 as
    selectable, even if ``OP_GPU_ALLOWED_IDS`` (a policy allowlist, not a
    physical-attachment fact) says otherwise.
    """
    from src.services.training.jobs import default_train_gpu_value

    arbiter_cfg = get_gpu_arbiter_config()
    allowed_ids = sorted(arbiter_cfg.allowed_gpu_ids)
    trainer_gpu_order = _trainer_gpu_order()
    if trainer_gpu_order:
        allowed_ids = (
            [i for i in allowed_ids if i in trainer_gpu_order]
            if allowed_ids
            else sorted(trainer_gpu_order)
        )
    default_value = default_train_gpu_value()

    if not allowed_ids:
        option = _build_train_gpu_option(
            [int(t) for t in default_value.split(',')], arbiter_cfg.gpu_labels
        )
        return TrainGpuOptionsResponse(
            options=[option.model_copy(update={'default': True})],
            allowed_ids=[],
            unrestricted=True,
        )

    options = [_build_train_gpu_option([gid], arbiter_cfg.gpu_labels) for gid in allowed_ids]
    if len(allowed_ids) > 1:
        options.append(_build_train_gpu_option(allowed_ids, arbiter_cfg.gpu_labels))

    default_idx = next(
        (i for i, o in enumerate(options) if o.value == default_value),
        0,
    )
    options[default_idx] = options[default_idx].model_copy(update={'default': True})
    return TrainGpuOptionsResponse(options=options, allowed_ids=allowed_ids, unrestricted=False)


# =============================================================================
# Promote — Phase 4 (Triton handoff)
# =============================================================================


class PromoteRequest(BaseModel):
    """Body for ``POST {api_prefix}/train/promote/{job_id}``."""

    triton_name: str = Field(
        ...,
        description='Desired Triton model name. Alphanumeric + underscore + hyphen.',
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


# Promote-gate thresholds. Mirrored as named constants so
# tests can monkey-patch them without re-parsing the router source.
PROMOTE_GATE_MAP50_MIN = 0.65
PROMOTE_GATE_PER_CLASS_PRECISION_MIN = 0.50
PROMOTE_GATE_PER_CLASS_SUPPORT_MIN = 5


class PromoteGateFailure(BaseModel):
    """One machine-readable promote-gate failure.

    F-64 (fresh-start E2E findings 2026-09-25): the 422 used to carry only
    free-text strings in ``failures``; a UI-only user saw just "API 422"
    because there was nothing structured to render. ``code`` is a stable
    identifier a frontend can switch on without string-parsing;
    ``message`` is still the human-readable detail for display.
    """

    code: str
    message: str
    class_name: str | None = None


class PromoteGateFailedDetail(BaseModel):
    """The ``detail`` body of every promote-blocking 422.

    ``force_allowed`` tells the caller (and the UI) whether re-submitting
    with ``force=true`` can get past *this specific* failure — some 422s
    (e.g. the job simply isn't finished yet) force cannot bypass.
    """

    message: str
    failures: list[PromoteGateFailure]
    force_allowed: bool
    override: str | None = None
    thresholds: dict[str, float | int] | None = None


class PromoteGateFailedResponse(BaseModel):
    """Documents the actual FastAPI error envelope: ``{"detail": ...}``."""

    detail: PromoteGateFailedDetail


def _evaluate_promote_gate(eval_block: dict[str, Any] | None) -> list[PromoteGateFailure]:
    """Return a list of structured gate failures, [] when the gate passes.

    The gate runs against ``status.json``'s ``eval`` block:
    mAP50 floor + per-class precision floor + per-class support floor.
    """
    failures: list[PromoteGateFailure] = []
    if not eval_block:
        failures.append(
            PromoteGateFailure(
                code='no_eval_block',
                message='no eval block in status.json — trainer never ran val()',
            )
        )
        return failures

    map50 = eval_block.get('map50')
    if not isinstance(map50, int | float) or map50 < PROMOTE_GATE_MAP50_MIN:
        failures.append(
            PromoteGateFailure(
                code='map50_below_floor',
                message=f'mAP50 {map50!r} < {PROMOTE_GATE_MAP50_MIN} (promote-gate floor)',
            )
        )

    per_class = eval_block.get('per_class') or []
    if not isinstance(per_class, list):
        failures.append(
            PromoteGateFailure(code='per_class_not_list', message='eval.per_class is not a list')
        )
        return failures

    for row in per_class:
        if not isinstance(row, dict):
            continue
        name = row.get('name') or f'class_id={row.get("class_id")}'
        precision = row.get('precision')
        support = row.get('support')
        # A non-numeric metric (null, string, missing) is a gate FAILURE,
        # not a skip — an unreadable eval is exactly the case the gate
        # exists to catch. isinstance(x, bool) is deliberately not
        # excluded from the numeric check for precision since Triton/trainer
        # never emits bool there; support uses `int` so a JSON `true`/`false`
        # would (correctly) still gate-fail on the < comparison below it if
        # it ever slipped through as a bool.
        if not isinstance(precision, int | float):
            failures.append(
                PromoteGateFailure(
                    code='per_class_precision_not_numeric',
                    message=f'{name}: precision {precision!r} is not numeric (promote-gate floor)',
                    class_name=name,
                )
            )
        elif precision < PROMOTE_GATE_PER_CLASS_PRECISION_MIN:
            failures.append(
                PromoteGateFailure(
                    code='per_class_precision_below_floor',
                    message=(
                        f'{name}: precision {precision:.3f} '
                        f'< {PROMOTE_GATE_PER_CLASS_PRECISION_MIN}'
                    ),
                    class_name=name,
                )
            )
        if not isinstance(support, int):
            failures.append(
                PromoteGateFailure(
                    code='per_class_support_not_int',
                    message=f'{name}: support {support!r} is not an int (promote-gate floor)',
                    class_name=name,
                )
            )
        elif support < PROMOTE_GATE_PER_CLASS_SUPPORT_MIN:
            failures.append(
                PromoteGateFailure(
                    code='per_class_support_below_floor',
                    message=(
                        f'{name}: support {support} < {PROMOTE_GATE_PER_CLASS_SUPPORT_MIN} '
                        'test crops (promoting on no test data)'
                    ),
                    class_name=name,
                )
            )
    return failures


async def _resolve_full_registry_for_promote(job_id: str) -> dict[int, str]:
    """Resolve class_id -> name for ``labels.txt``, preferring the registry
    snapshot pinned at submit time over the live registry.

    ``labels.txt`` used to be rebuilt from the *live* registry at promote
    time — a rename between export and promote silently mislabeled the
    served model. Falls back to the live registry only when no pin is
    available (older runs, or a pin/read failure), and always logs loudly
    when it does so the gap is visible in the API logs.
    """
    job_raw = await train_jobs.read_job_spec(job_id)
    snapshot_path = (job_raw or {}).get('registry_snapshot_path')
    if snapshot_path:
        try:
            import json
            from pathlib import Path as _Path

            data = json.loads(_Path(snapshot_path).read_text(encoding='utf-8'))
            pinned = {
                int(c['class_id']): str(c['class_name'])
                for c in data.get('classes') or []
                if not c.get('deprecated')
            }
        except Exception as exc:
            logger.warning(
                'curation_promote_registry_pin_unreadable',
                job_id=job_id,
                snapshot_path=snapshot_path,
                error=str(exc),
            )
        else:
            if pinned:
                return pinned
            logger.warning(
                'curation_promote_registry_pin_empty', job_id=job_id, snapshot_path=snapshot_path
            )
    else:
        logger.warning(
            'curation_promote_registry_pin_missing',
            job_id=job_id,
            note=(
                'no registry_snapshot_path on this job — falling back to the LIVE '
                'registry. A class rename since submit would silently relabel '
                'this model.'
            ),
        )
    registry_snapshot = get_class_registry().load()
    return {c.class_id: c.class_name for c in registry_snapshot.classes if not c.deprecated}


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


@router.post(
    '/promote/{job_id}',
    response_model=PromoteResponse,
    responses={422: {'model': PromoteGateFailedResponse}},
)
async def promote_run(
    payload: PromoteRequest,
    job_id: Annotated[str, PathParam(description='Training job_id from {api_prefix}/train/runs')],
) -> PromoteResponse:
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
    # Lazy import — keeps the API container slim if no one ever
    # promotes (e.g. a fresh dev box).
    from pathlib import Path

    from src.services.training.triton_promote import (
        CheckpointNotFoundError,
        ClassRemapMissingError,
        ClassRemapUnreadableError,
        ModelNameConflictError,
        PromoteError,
        TritonLoadError,
        build_class_id_to_name,
        promote_yolo26_to_triton,
        resolve_class_remap,
    )

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
    gate_failures = _evaluate_promote_gate(job_status.eval)
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
                override='pass force=true in the request body',
                thresholds=gate_thresholds,
            ).model_dump(),
        )

    # Resolve full-registry class names (id -> name), preferring the
    # snapshot pinned at submit time. Subset-trained models get
    # renumbered inside build_class_id_to_name using the resolved remap
    # (manifest lineage.class_remap first, then the weights-dir file).
    full_registry = await _resolve_full_registry_for_promote(job_id)
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
                    override='pass force=true in the request body',
                ).model_dump(),
            )
        logger.warning(
            'curation_promote_class_remap_missing_force_bypass',
            job_id=job_id,
            note='subset/single_cls run promoted with force=true and no resolvable class_remap',
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

    try:
        result = await promote_yolo26_to_triton(
            status=job_status,
            triton_name=payload.triton_name,
            class_id_to_name=class_id_to_name,
            max_batch_size=payload.max_batch_size,
            input_size=payload.input_size,
            fp16=payload.fp16,
            overwrite=payload.overwrite,
            class_remap=class_remap,
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
    )


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
    from src.services.training.triton_promote import reload_promoted_models

    result = await reload_promoted_models()
    return ReloadPromotedResponse(
        status=result.get('status', 'ok'),
        reloaded=result.get('reloaded', []),
        failed=result.get('failed', []),
    )


# =============================================================================
# /manifest/{job_id}
# =============================================================================


@router.get('/manifest/{job_id}')
async def get_manifest(
    job_id: Annotated[str, PathParam(description='Training job_id from {api_prefix}/train/runs')],
) -> ORJSONResponse:
    """Return the run's ``manifest.json`` (lineage envelope).

    404 if the manifest doesn't exist yet — older runs that finished
    before the manifest writer landed simply lack one.
    """
    manifest = await train_jobs.read_manifest(job_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f'no manifest for job {job_id!r}')
    return ORJSONResponse(content=manifest)


# =============================================================================
# /artifacts/{job_id}/{name}
# =============================================================================


@router.get('/artifacts/{job_id}/{name}', response_class=FileResponse)
async def get_run_artifact(
    job_id: Annotated[str, PathParam(description='Training job_id from {api_prefix}/train/runs')],
    name: Annotated[
        str,
        PathParam(
            description=(
                'Whitelisted artifact filename '
                '(see src.services.training.jobs.RUN_ARTIFACT_WHITELIST), '
                'e.g. confusion_matrix.png'
            )
        ),
    ],
) -> FileResponse:
    """Serve one whitelisted metrics/plot artifact from a run's directory.

    This is the only sanctioned way to reach these files -- the server
    filesystem path itself never appears on the wire (``eval.
    confusion_matrix_url`` on ``{api_prefix}/train/status*``/``manifest``
    points here instead). 404 alike for an unwhitelisted name, an unknown
    job, or a file that hasn't been written yet — nothing here
    distinguishes those cases to a caller.
    """
    try:
        path = await train_jobs.read_artifact(job_id, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if path is None:
        raise HTTPException(status_code=404, detail=f'no artifact {name!r} for job {job_id!r}')
    return FileResponse(path, media_type=train_jobs.artifact_media_type(name))
