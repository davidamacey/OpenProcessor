"""Training-pipeline router.

Mounts at ``{api_prefix}/train`` and exposes the ten endpoints the labeler
frontend's train page consumes. The router is intentionally thin: every file-system
write/read goes through :mod:`src.services.training.jobs`, and every
hyperparameter table lives in :mod:`src.services.training.profiles`.

Endpoints (per design table §7):

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

Pre-flight contract (design §15.1): ``/start`` calls ``/preflight``
internally and refuses to write ``job.json`` if any check has severity
``block``. Pass ``?force=true`` to bypass; the report is still returned in
the 422 body so the UI can render it inline.
"""

from __future__ import annotations

import os
import shutil
from datetime import UTC
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Path as PathParam, Query, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.config import IndexRole, get_curation_config, get_region_fields, index_name
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.routers.curation import get_class_registry
from src.routers.curation._common import OpenSearchDep  # noqa: TC001 - used at runtime by FastAPI
from src.services.training import jobs as train_jobs
from src.services.training.gpu_arbiter import needs_multi_gpu_stop, probe_trainer_reachable
from src.services.training.jobs import Profile, TrainCampaignSpec, TrainJobSpec, TrainJobStatus
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
    """Single row in the preflight report (design §15.1)."""

    name: str
    severity: PreflightSeverity
    message: str
    detail: dict[str, Any] | None = None


class PreflightReport(BaseModel):
    """Bundled preflight result the frontend renders inline on the form."""

    blocked: bool
    checks: list[PreflightCheck]
    summary: str = ''


# Disk-space threshold (design §15.1: ≥50 GB free on /data).
MIN_FREE_DISK_GB = 50

# Per-class crop minimum thresholds (design §15.1: hard-fail at <20,
# warn at <500).
HARD_MIN_CROPS_PER_CLASS = 20
WARN_MIN_CROPS_PER_CLASS = 500
MIN_TEST_CROPS_PER_CLASS = 5


# =============================================================================
# Helpers
# =============================================================================


async def _count_validated_per_class(
    opensearch: Any,
    class_ids: list[int],
) -> dict[int, int]:
    """Return ``{class_id: validated_crop_count}`` for each requested class.

    Uses the ``legacy_vehicle_crops`` index. Validated crops are those with
    ``label_state == 'confirmed'``. We issue one bool-filter aggregation
    rather than N count requests.
    """
    if not class_ids:
        return {}
    # The legacy_vehicle_crops schema uses a boolean ``label_validated`` field
    # (set true by both human-confirmation and auto-promotion). The design
    # doc's earlier reference to ``label_state == 'confirmed'`` predated
    # the schema settling on the boolean — we keep the boolean as the
    # source of truth and treat both human and auto-promoted labels as
    # eligible training data.
    body = {
        'size': 0,
        'query': {
            'bool': {
                'must': [
                    {'term': {'class_validated': True}},
                    {'terms': {'class_id': class_ids}},
                ]
            }
        },
        'aggs': {
            'by_class': {
                'terms': {'field': 'class_id', 'size': max(len(class_ids), 1)},
            }
        },
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        logger.warning('train_class_count_failed', error=str(exc))
        return dict.fromkeys(class_ids, 0)
    counts = dict.fromkeys(class_ids, 0)
    for bucket in (resp.get('aggregations') or {}).get('by_class', {}).get('buckets', []):
        cid = bucket.get('key')
        if isinstance(cid, int):
            counts[cid] = int(bucket.get('doc_count', 0))
    return counts


async def _count_test_per_class(
    opensearch: Any,
    class_ids: list[int],
) -> dict[int, int]:
    """``{class_id: test_holdout_count}`` — design §15.1 requires ≥5 each."""
    if not class_ids:
        return {}
    body = {
        'size': 0,
        'query': {
            'bool': {
                'must': [
                    {'term': {'class_validated': True}},
                    {'term': {'test_holdout': True}},
                    {'terms': {'class_id': class_ids}},
                ]
            }
        },
        'aggs': {
            'by_class': {'terms': {'field': 'class_id', 'size': max(len(class_ids), 1)}},
        },
    }
    try:
        resp = await opensearch.search(index=CURATION_ITEMS_INDEX, body=body)
    except Exception as exc:
        logger.warning('train_test_count_failed', error=str(exc))
        return dict.fromkeys(class_ids, 0)
    counts = dict.fromkeys(class_ids, 0)
    for bucket in (resp.get('aggregations') or {}).get('by_class', {}).get('buckets', []):
        cid = bucket.get('key')
        if isinstance(cid, int):
            counts[cid] = int(bucket.get('doc_count', 0))
    return counts


async def _count_pending_ingest(opensearch: Any) -> int:
    """Count crops still awaiting the SAM3/Gemma plate pipeline.

    A dual-GPU run stops the SAM3 + Gemma containers (gpu_arbiter), pausing
    plate detection/verification. This lets preflight warn the operator how
    much in-flight ingest that will stall. Legacy status names included so a
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


def _free_gb(path: str) -> float | None:
    """Free disk space on ``path``'s filesystem, in GB.

    P2-8: this used to fail OPEN on any ``OSError`` (return ``float('inf')``,
    i.e. "infinite free space") — the exact opposite of a safe default.
    ``None`` now means "couldn't determine", which the caller reports as
    ``severity='unknown'``, never ``'ok'``.
    """
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return usage.free / (1024**3)


def _resolve_disk_check_path(spec: TrainJobSpec) -> str:
    """Pick the path to stat for the free-disk check (P2-8).

    Previously hardcoded to ``/data`` — inside the yolo-api container,
    only specific subpaths under ``/data`` (e.g.
    a deployment-specific training-data mount, see the deployment's own
    compose overlay)
    are bind-mounted from the real training-data volume; ``/data``
    itself resolves to the container's own overlay filesystem, which
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
    """False if ``path`` is on the same device as ``/`` (P2-8).

    A strong signal the real training-data volume isn't actually mounted
    into this container at ``path`` — e.g. a dev box or misconfigured
    compose file where the bind mount silently didn't take, leaving
    ``path`` resolving to the container's own root filesystem. Any
    ``OSError`` here is treated as "can't confirm it's sane" (False), not
    a soft pass.
    """
    try:
        return Path(path).stat().st_dev != Path('/').stat().st_dev
    except OSError:
        return False


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
# :mod:`src.services.curation.export_single_class` writes;
# ``lpr_single_class`` is the reference implementation's own value for the
# same shape, accepted so an export produced before that exporter existed
# still preflights.
SINGLE_CLASS_DATASET_KINDS: frozenset[str] = frozenset({'single_class', 'lpr_single_class'})


def _single_class_label(manifest: dict[str, Any]) -> str:
    """Human-readable name of a single-class export's target class."""
    name = manifest.get('class_name')
    if name:
        return str(name)
    names = manifest.get('class_names')
    if isinstance(names, list) and names:
        return ', '.join(str(n) for n in names)
    return 'target class'


def _append_lpr_data_checks(checks: list[PreflightCheck], manifest: dict[str, Any]) -> None:
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

    # ---- 2b. trainer reachable (P1-8) ------------------------------------------
    # Without this, /start writes job.json and the run sits in `queued`
    # forever with no error if the configured trainer container was never started.
    trainer_up, trainer_detail = await probe_trainer_reachable()
    checks.append(
        PreflightCheck(
            name='trainer_reachable',
            severity='ok' if trainer_up else 'block',
            message=trainer_detail,
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

    # ---- 4 & 5. data sufficiency (LPR-aware) ---------------------------------
    # A single-class LPR export validates from its own manifest (disk dataset),
    # not the multi-class registry's class_validated counts.
    _lpr_manifest = _read_export_manifest(spec.dataset_export_dir)
    _is_lpr = _lpr_manifest.get('dataset_kind') in SINGLE_CLASS_DATASET_KINDS
    target_classes = [] if _is_lpr else _resolve_target_classes(spec)

    # ---- 3b. include_classes resolvable against this export (P2-8) ----------
    # Previously an unresolvable include_classes id (deprecated, typo, or a
    # class this export never had) surfaced as either a 500 or a job that
    # failed deep inside the trainer container (subset_dataset.py's own
    # ValueError, minutes into a run). Catch it here instead, at the API
    # boundary, with a structured preflight check. Skipped entirely for LPR
    # jobs (single-class, no include_classes concept).
    if not _is_lpr and spec.include_classes:
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
    if _is_lpr:
        _append_lpr_data_checks(checks, _lpr_manifest)
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
        counts = await _count_validated_per_class(opensearch, target_classes)
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
        test_counts = await _count_test_per_class(opensearch, target_classes)
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

    # ---- 6. empty-label / plate-pairing (P2-8: real scan, not a stub) --------
    # Both used to be hardcoded to 'ok' with no scan ever run. LPR
    # (single-class plate exports) is handled by its own additive branch —
    # background/negative frames are a legitimate, expected empty-label
    # case there (accounted for via the manifest's own counts), and there
    # are no parent vehicle boxes to pair against by construction.
    if _is_lpr:
        positive = int(_lpr_manifest.get('positive_images') or 0)
        total_lpr_images = int(_lpr_manifest.get('total_images') or 0) or None
        background_note = (
            f' ({total_lpr_images - positive} background/negative frames)'
            if total_lpr_images is not None
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

            if scan.plate_boxes == 0:
                checks.append(
                    PreflightCheck(
                        name='region_pairing',
                        severity='ok',
                        message='no region boxes in this export/subset',
                    )
                )
            elif scan.unpaired_plate_boxes > 0:
                checks.append(
                    PreflightCheck(
                        name='region_pairing',
                        severity='warn',
                        message=(
                            f'{scan.unpaired_plate_boxes}/{scan.plate_boxes} region '
                            'boxes have no matching parent item box in the same image'
                        ),
                        detail={
                            'region_boxes': scan.plate_boxes,
                            'unpaired_region_boxes': scan.unpaired_plate_boxes,
                        },
                    )
                )
            else:
                checks.append(
                    PreflightCheck(
                        name='region_pairing',
                        severity='ok',
                        message=f'all {scan.plate_boxes} region boxes are paired',
                    )
                )

    # ---- active-ingest warning (dual-GPU runs stop SAM3 + Gemma) ----------
    if needs_multi_gpu_stop(spec.cuda_visible_devices):
        pending = await _count_pending_ingest(opensearch)
        if pending > 0:
            checks.append(
                PreflightCheck(
                    name='ingest_idle',
                    severity='warn',
                    message=(
                        f'{pending:,} crops are still pending region '
                        'detection/verification. A dual-GPU run stops the segmenter '
                        'and VLM containers, pausing that ingest until the run '
                        'finishes (it auto-resumes afterward).'
                    ),
                    detail={'pending_ingest': pending},
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    name='ingest_idle',
                    severity='ok',
                    message='No ingest backlog — safe to stop the segmenter + VLM for training',
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
    and ``force=False``. The trainer picks up the file out-of-band.
    """
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
    # GPU arbiter — pause Gemma worker (single-GPU) or stop the container
    # (dual-GPU) BEFORE the trainer picks the job up. Best-effort: a missing
    # docker socket falls back to sentinel-only and never fails /start.
    from src.services.training.gpu_arbiter import claim_gpus_for_training

    try:
        await claim_gpus_for_training(spec.cuda_visible_devices)
    except Exception as exc:
        logger.warning('gpu_arbiter_claim_failed', error=str(exc))
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
    """Submit a multi-size training campaign (design §14).

    Preflight runs once on a synthetic spec built from the first run; the
    rest of the runs share the same dataset / class set so a single
    preflight covers them all.
    """
    if not campaign.runs:
        raise HTTPException(status_code=400, detail='campaign requires at least one run')

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
    from src.services.training.gpu_arbiter import claim_gpus_for_training

    try:
        await claim_gpus_for_training(campaign.cuda_visible_devices)
    except Exception as exc:
        logger.warning('gpu_arbiter_claim_failed', error=str(exc))
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


class PresetsResponse(BaseModel):
    class_subset_presets: list[dict[str, Any]] = Field(default_factory=list)


@router.get('/presets', response_model=PresetsResponse)
async def list_presets() -> PresetsResponse:
    """Return server-side class-subset presets (design §12.3)."""
    return PresetsResponse(class_subset_presets=get_class_subset_presets())


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


# Promote-gate thresholds (design §15.2). Mirrored as named constants so
# tests can monkey-patch them without re-parsing the router source.
PROMOTE_GATE_MAP50_MIN = 0.65
PROMOTE_GATE_PER_CLASS_PRECISION_MIN = 0.50
PROMOTE_GATE_PER_CLASS_SUPPORT_MIN = 5


def _evaluate_promote_gate(eval_block: dict[str, Any] | None) -> list[str]:
    """Return a list of human-readable failure messages, [] when the gate passes.

    The gate runs against ``status.json``'s ``eval`` block (design §15.2):
    mAP50 floor + per-class precision floor + per-class support floor.
    """
    failures: list[str] = []
    if not eval_block:
        failures.append('no eval block in status.json — trainer never ran val()')
        return failures

    map50 = eval_block.get('map50')
    if not isinstance(map50, int | float) or map50 < PROMOTE_GATE_MAP50_MIN:
        failures.append(f'mAP50 {map50!r} < {PROMOTE_GATE_MAP50_MIN} (promote-gate floor)')

    per_class = eval_block.get('per_class') or []
    if not isinstance(per_class, list):
        failures.append('eval.per_class is not a list')
        return failures

    for row in per_class:
        if not isinstance(row, dict):
            continue
        name = row.get('name') or f'class_id={row.get("class_id")}'
        precision = row.get('precision')
        support = row.get('support')
        # A non-numeric metric (null, string, missing) is a gate FAILURE,
        # not a skip — an unreadable eval is exactly the case the gate
        # exists to catch (P1-9). isinstance(x, bool) is deliberately not
        # excluded from the numeric check for precision since Triton/trainer
        # never emits bool there; support uses `int` so a JSON `true`/`false`
        # would (correctly) still gate-fail on the < comparison below it if
        # it ever slipped through as a bool.
        if not isinstance(precision, int | float):
            failures.append(f'{name}: precision {precision!r} is not numeric (promote-gate floor)')
        elif precision < PROMOTE_GATE_PER_CLASS_PRECISION_MIN:
            failures.append(
                f'{name}: precision {precision:.3f} < {PROMOTE_GATE_PER_CLASS_PRECISION_MIN}'
            )
        if not isinstance(support, int):
            failures.append(f'{name}: support {support!r} is not an int (promote-gate floor)')
        elif support < PROMOTE_GATE_PER_CLASS_SUPPORT_MIN:
            failures.append(
                f'{name}: support {support} < {PROMOTE_GATE_PER_CLASS_SUPPORT_MIN} '
                'test crops (promoting on no test data)'
            )
    return failures


async def _resolve_full_registry_for_promote(job_id: str) -> dict[int, str]:
    """Resolve class_id -> name for ``labels.txt``, preferring the registry
    snapshot pinned at submit time over the live registry (P1-12).

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
                'legacy_promote_registry_pin_unreadable',
                job_id=job_id,
                snapshot_path=snapshot_path,
                error=str(exc),
            )
        else:
            if pinned:
                return pinned
            logger.warning(
                'legacy_promote_registry_pin_empty', job_id=job_id, snapshot_path=snapshot_path
            )
    else:
        logger.warning(
            'legacy_promote_registry_pin_missing',
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


@router.post('/promote/{job_id}', response_model=PromoteResponse)
async def promote_run(
    payload: PromoteRequest,
    job_id: Annotated[str, PathParam(description='Training job_id from {api_prefix}/train/runs')],
) -> PromoteResponse:
    """Promote a finished training run into the Triton model repo.

    Reads the run's ``status.json`` for the ONNX export the trainer
    produced during the ``exporting`` state, copies it into
    ``models/<triton_name>/1/model.onnx``, writes ``config.pbtxt``
    using the YOLO26 single-output template (no NMS plugin — NMS is
    internal to the YOLO26 forward pass per design §10), writes
    ``labels.txt`` honoring any subset-training class_remap, and POSTs
    Triton's load endpoint to make the model active immediately.

    Errors:
        404: job not found, or its ONNX export hasn't been written
        409: a Triton model with this name already exists (use
              ``overwrite=true`` to clobber)
        422: status.json shows the run isn't in a promote-ready state
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
            detail=(
                f'job {job_id!r} is in state {job_status.state!r}; only '
                "'finished' or 'exporting' runs can be promoted"
            ),
        )

    if not job_status.checkpoint_path:
        raise HTTPException(
            status_code=422,
            detail=f'job {job_id!r} has no checkpoint_path in status.json',
        )

    # Promote gate (design §15.2). Refuses underqualified runs unless the
    # caller explicitly passes force=true.
    gate_failures = _evaluate_promote_gate(job_status.eval)
    gate_report: dict[str, Any] = {
        'thresholds': {
            'map50_min': PROMOTE_GATE_MAP50_MIN,
            'per_class_precision_min': PROMOTE_GATE_PER_CLASS_PRECISION_MIN,
            'per_class_support_min': PROMOTE_GATE_PER_CLASS_SUPPORT_MIN,
        },
        'failures': gate_failures,
    }
    if gate_failures and not payload.force:
        raise HTTPException(
            status_code=422,
            detail={
                'message': 'promote gate failed',
                **gate_report,
                'override': 'pass force=true in the request body',
            },
        )

    # Resolve full-registry class names (id -> name), preferring the
    # snapshot pinned at submit time (P1-12). Subset-trained models get
    # renumbered inside build_class_id_to_name using the resolved remap
    # (P2-7: manifest lineage.class_remap first, then the weights-dir file).
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
            raise HTTPException(
                status_code=422,
                detail=str(ClassRemapMissingError(job_id)),
            )
        logger.warning(
            'legacy_promote_class_remap_missing_force_bypass',
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

    # Stamp the manifest's promoted_to field (design §15.4), including
    # whether the gate was bypassed and the (possibly-failing) report so a
    # forced promote is traceable later. Older runs without a manifest
    # legitimately have nothing to stamp — stamp_manifest_promotion returns
    # False for that case, which is NOT an error. An actual write failure
    # (disk, permissions, corrupt JSON) is a real problem: we don't fail
    # the promote outright (the model is already live in Triton at this
    # point — a 500 here would be misleading), but we surface it loudly via
    # both an ERROR-level log and `lineage_stamped: false` in the response
    # instead of the previous silent WARNING-and-forget (P1-10).
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
# /manifest/{job_id}
# =============================================================================


@router.get('/manifest/{job_id}')
async def get_manifest(
    job_id: Annotated[str, PathParam(description='Training job_id from {api_prefix}/train/runs')],
) -> ORJSONResponse:
    """Return the run's ``manifest.json`` (design §15.4 lineage envelope).

    404 if the manifest doesn't exist yet — older runs that finished
    before the manifest writer landed simply lack one.
    """
    manifest = await train_jobs.read_manifest(job_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f'no manifest for job {job_id!r}')
    return ORJSONResponse(content=manifest)
