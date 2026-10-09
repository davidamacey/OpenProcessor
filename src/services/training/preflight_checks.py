"""Preflight models and check helpers for training jobs (no FastAPI dependency)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from src.clients.curation_opensearch import get_class_registry
from src.config import get_curation_config, get_region_fields
from src.config.curation import items_index
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation.dataset_thresholds import (
    HARD_MIN_CROPS_PER_CLASS,
    MIN_TEST_CROPS_PER_CLASS,
    WARN_MIN_CROPS_PER_CLASS,
    dataset_thresholds,
)
from src.services.training.augmentation_presets import unknown_preset_error
from src.services.training.job_files import trainer_root_dir


if TYPE_CHECKING:
    from src.services.training.job_models import AugmentationSpec, TrainJobSpec


logger = get_logger(__name__)

config = get_curation_config()
F = get_region_fields()


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


async def count_validated_and_test_per_class(
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
        resp = await opensearch.search(index=items_index(), body=body)
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


async def count_pending_ingest(opensearch: Any) -> int:
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
        resp = await opensearch.count(index=items_index(), body=body)
    except Exception as exc:
        logger.warning('train_pending_count_failed', error=str(exc))
        return 0
    return int(resp.get('count', 0))


def resolve_target_classes(spec: TrainJobSpec) -> list[int]:
    """Resolve the effective class list for a spec.

    ``include_classes=None`` → every non-deprecated item class in the registry
    (the active profile's region class labels sub-boxes, never a whole item, so
    it has no per-item samples to require).
    """
    if spec.include_classes:
        return list(spec.include_classes)
    from src.services.curation.region_class import item_classes

    return [c.class_id for c in item_classes(get_class_registry().load().classes)]


def augmentation_preset_error(augmentation: AugmentationSpec | None) -> str | None:
    """Error for an enabled augmentation block naming an unknown preset.

    A disabled block's preset is never built by the trainer, so it isn't
    judged. Checked by preflight and, ahead of every side effect, by
    ``/start`` and ``/start_campaign``.
    """
    if augmentation is None or not augmentation.enabled:
        return None
    return unknown_preset_error(augmentation.preset)


def free_gb(path: str) -> float | None:
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


def resolve_disk_check_path(spec: TrainJobSpec) -> str:
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
    # P1-deferred: nested under the bound project's own state dir (rather
    # than the global state_dir) so a fallback disk-space check never
    # points at another project's volume.
    return os.environ.get('OP_TRAIN_STAGING', str(config.project_state_dir / 'training_staging'))


def training_volume_mount_sane(path: str) -> bool:
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
    path = trainer_root_dir() / TRAINER_CAPABILITIES_FILENAME
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def read_trainer_gpu_order() -> list[int] | None:
    """``gpu_order`` from the trainer capabilities file, or ``None`` if
    unknown (missing file, or a non-empty-but-unparseable field)."""
    caps = _read_trainer_capabilities()
    if caps is None:
        return None
    order = caps.get('gpu_order')
    if not isinstance(order, list) or not all(isinstance(i, int) for i in order):
        return None
    return order


def read_export_manifest(dataset_export_dir: str | None) -> dict[str, Any]:
    """Best-effort read of an export's ``manifest.json`` (``{}`` on any failure)."""
    if not dataset_export_dir:
        return {}
    import json
    from pathlib import Path

    try:
        return json.loads((Path(dataset_export_dir) / 'manifest.json').read_text())
    except Exception:
        return {}


def unresolvable_include_classes(
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


def append_single_class_data_checks(checks: list[PreflightCheck], manifest: dict[str, Any]) -> None:
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


def append_label_scan_checks(
    checks: list[PreflightCheck],
    spec: TrainJobSpec,
    single_class_manifest: dict[str, Any],
    is_single_class: bool,
) -> None:
    """Empty-label and region-pairing rows from a real scan of the export."""
    # ---- 6. empty-label / region-pairing (real scan, not a stub) --------
    # Both used to be hardcoded to 'ok' with no scan ever run. Single-class
    # exports are handled by their own additive branch — background/negative
    # frames are a legitimate, expected empty-label case there (accounted
    # for via the manifest's own counts), and there are no parent item
    # boxes to pair against by construction.
    if is_single_class:
        positive = int(single_class_manifest.get('positive_images') or 0)
        total_single_class_images = int(single_class_manifest.get('total_images') or 0) or None
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
