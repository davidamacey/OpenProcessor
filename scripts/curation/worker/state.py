"""Auto-split sub-module of the curation detection worker.

See ``scripts/curation/sam_worker_main.py`` for the entry point and
the ``scripts/curation/worker/`` package for the rest of the split.
"""

from __future__ import annotations

import asyncio
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.detection.cascade_detect import DEFAULT_PROFILE


logger = get_logger('curation_worker')

_config = get_curation_config()

CURATION_ITEMS_INDEX = _config.items_index
DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')
DEFAULT_TRITON = os.environ.get('TRITON_URL', 'triton-server:8001')
DEFAULT_SAM3 = os.environ.get('SAM3_URL', 'http://sam3:8000')
# Dual-GPU SAM3: SAM3_URLS=http://sam3-gpu0:8000,http://sam3-gpu1:8000
# When set, the worker round-robins requests across all listed URLs so
# the parallelism across GPUs adds up. SAM3_URL is kept as the
# single-URL fallback for backwards compatibility.
DEFAULT_SAM3_URLS = os.environ.get('SAM3_URLS', '').strip()
DEFAULT_GEMMA = os.environ.get('GEMMA_URL', os.environ.get('OPENWEBUI_BASE_URL', ''))
DEFAULT_PAUSE_SENTINEL = Path(
    os.environ.get(
        'OP_WORKER_PAUSE_SENTINEL',
        str(_config.state_dir / 'vlm_worker' / 'pause.sentinel'),
    )
)
JPEG_QUALITY = 90

# Crop classes to skip the primary-detector retry on. Source of truth is
# the class registry (see ``CurationConfig.class_registry_path``, group
# field). The accessor surface doesn't expose a dedicated groups helper,
# so the profile pins the set directly (see
# DetectionProfile.secondary_shape_groups on the reference license-plate
# profile) — update there when the registry adds / renames a group.
# The reference detector is known weak on this shape class; the
# secondary segmenter is the better bet.
SECONDARY_SHAPE_GROUPS: frozenset[str] = DEFAULT_PROFILE.secondary_shape_groups

# Status names (task #7 rename — see plan / task #7 description).
# Worker emits the new long-form names everywhere; reads accept both
# legacy and new names until the OS migration completes.
STATUS_PENDING_DETECTION = 'pending_detection'
STATUS_PENDING_VERIFICATION = 'pending_verification'
_PENDING_DETECTION_ALIASES = frozenset({STATUS_PENDING_DETECTION, 'pending'})
_PENDING_VERIFICATION_ALIASES = frozenset({STATUS_PENDING_VERIFICATION, 'pending_verify'})

# Terminal statuses the worker MUST NOT override.
_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        'detected',
        'no_plate_visible',
        'verify_rejected',
        'no_plate_box',
        'detection_failed',
        # Human-marked bad detection (box kept for FP analysis / hard-
        # negative training). Terminal — worker must not re-run it.
        'false_positive',
    }
)


# =============================================================================
# _ItemTask
# =============================================================================


@dataclass
class _ItemTask:
    """One item pulled from OpenSearch + its in-flight detector results."""

    crop_id: str
    image_path: str
    vehicle_bbox_norm: tuple[float, float, float, float]
    plate_status: str | None
    class_name: str
    group: str
    # X-Request-ID carried over from the HTTP ingest call that produced
    # this crop ('-' for non-HTTP ingests). Bound to structlog contextvars
    # in every consumer's per-task processing block so worker logs can
    # be joined back to the originating request (Phase 4a).
    request_id: str = '-'
    # B-PR5 cohort marker. ``coco_yolo11_proposal`` means the primary
    # classifier missed and we fell back to a generic proposal — these
    # crops still need a class label, so combining class + region-verify
    # + OCR in one ``VlmLabeler.label_combined`` call cuts ~2 round-trips
    # per crop.
    class_source: str = ''
    # Primary classifier confidence (or fallback proposal score) at
    # ingest time. Used by the combined-call cohort gate to decide
    # whether a low-confidence crop should re-ask the VLM (low conf) or
    # take the legacy two-call path (high conf, class trusted).
    class_confidence: float = 0.0
    # Human-provenance guard (P0-2): True when a human already confirmed
    # this crop's class. ``_should_classify`` must never let the worker
    # reclassify a human-validated crop regardless of class_source.
    class_validated: bool = False
    # test_holdout guard (P0-3): True when this crop is frozen as
    # evaluation ground truth. Scoped to CLASS fields only — region
    # detection/writes must stay unconditional (test_holdout protects
    # class-label ground truth, not region state), so this only feeds
    # ``_should_classify``, never ``_build_pending_query`` (which would
    # also starve holdout crops of region detection).
    test_holdout: bool = False
    # Existing primary-detector candidate (already in source frame) for
    # pending_verify.
    lpr_plate_in_source: tuple[float, float, float, float] | None = None
    lpr_score: float = 0.0
    # Cropped JPEG bytes — built lazily so we don't load images we'd skip.
    crop_jpeg: bytes | None = None
    # Two-stage pipeline state — Stage A (primary/secondary/OCR-det)
    # writes these, Stage B (VLM verify) consumes them.
    candidate_source: str = ''  # 'lpr' / 'lpr_existing' / 'sam3' / 'paddle' / 'paddleocr_rec' / ''
    candidate_in_crop: tuple[float, float, float, float] | None = None
    candidate_in_source: tuple[float, float, float, float] | None = None
    candidate_score: float = 0.0
    # text-hint: when the OCR pipeline produced the candidate, its
    # recognized text rides along so writers can persist it even when
    # the downstream VLM verify returns an empty read.
    candidate_text: str | None = None
    candidate_text_confidence: float | None = None
    # Detection trace — list of "<detector>:<tag>" strings the task
    # accumulates as it moves through the cascade, serialized as
    # ``RegionFields.detector_chain`` on every write that produces a
    # region bbox.
    detection_trace: list[str] = field(default_factory=list)
    # B-PR5: class-side update from a combined VLM call. Layered onto
    # ``update_doc`` at write time so subsequent region-cascade writes
    # don't clobber the class fields (cohort = primary-detector-missed).
    combined_class_update: dict[str, Any] = field(default_factory=dict)
    # Final outcome to write back. Empty dict means "no update for this crop".
    update_doc: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Crop IO
# =============================================================================

# Phase A: RAM-backed crop cache populated by yolo-api at ingest time.
# The worker reads <CROP_CACHE_DIR>/<crop_id>.jpg first; only falls
# back to opening + cropping the source image when the cache misses
# (e.g. crops ingested before this code shipped). When the cache hits
# we skip the HDD read AND the EXIF + decode + crop work.
CROP_CACHE_DIR = str(_config.crop_cache_dir)


# Cache hit/miss counters — process-local, reset on restart. Aggregated
# into the periodic metrics log so we can audit /dev/shm utilization
# without a separate Prometheus endpoint.
_cache_hits = 0
_cache_misses = 0


def _crop_jpeg_from_cache(crop_id: str) -> bytes | None:
    """Return the cached item-crop JPEG bytes or None on miss."""
    global _cache_hits, _cache_misses  # noqa: PLW0603 — counter intentionally module-level
    if not CROP_CACHE_DIR:
        _cache_misses += 1
        return None
    p = Path(CROP_CACHE_DIR) / f'{crop_id}.jpg'
    try:
        b = p.read_bytes()
        _cache_hits += 1
        return b
    except FileNotFoundError:
        _cache_misses += 1
        return None
    except OSError as exc:
        _cache_misses += 1
        logger.warning('crop_cache_read_error', crop_id=crop_id, error=str(exc))
        return None


def _crop_jpeg_from_disk(image_path: str, bbox: tuple[float, float, float, float]) -> bytes | None:
    """Extract a JPEG of the item crop from the source image on disk.

    This is the slow path — re-opens the source image, EXIF-transposes,
    crops, re-encodes JPEG. Used only when the RAM crop cache misses.
    """
    p = Path(image_path)
    if not p.is_file():
        logger.warning('image_missing', path=image_path)
        return None
    try:
        with p.open('rb') as f:
            img = Image.open(f)
            img.load()
            img = ImageOps.exif_transpose(img)
            if img.mode != 'RGB':
                img = img.convert('RGB')
    except UnidentifiedImageError:
        logger.warning('image_unreadable', path=image_path)
        return None
    except OSError as exc:
        logger.warning('image_io_error', path=image_path, error=str(exc))
        return None

    full_w, full_h = img.size
    x1, y1, x2, y2 = bbox
    x1i = max(0, round(x1 * full_w))
    y1i = max(0, round(y1 * full_h))
    x2i = max(x1i + 1, round(x2 * full_w))
    y2i = max(y1i + 1, round(y2 * full_h))
    if x2i <= x1i or y2i <= y1i:
        return None
    crop = img.crop((x1i, y1i, x2i, y2i))
    buf = io.BytesIO()
    crop.save(buf, format='JPEG', quality=JPEG_QUALITY)
    return buf.getvalue()


def _crop_jpeg_for_task(
    crop_id: str, image_path: str, bbox: tuple[float, float, float, float]
) -> bytes | None:
    """Cache-first crop fetch: try RAM cache, fall back to source image."""
    cached = _crop_jpeg_from_cache(crop_id)
    if cached is not None:
        return cached
    return _crop_jpeg_from_disk(image_path, bbox)


def _is_secondary_shape(task: _ItemTask) -> bool:
    """True if the crop's class belongs to a secondary-shape group.

    See :data:`SECONDARY_SHAPE_GROUPS` for the source-of-truth comment.
    """
    if task.group:
        return task.group in SECONDARY_SHAPE_GROUPS
    # Some old crops have no ``group`` field stored. Best-effort fallback:
    # a name suffix of ``bike`` is a strong signal under the reference
    # naming convention (cruiserbike, sportbike, dirtbike, etc.). Keep
    # this narrow — false positives just shift more crops to the
    # secondary segmenter unnecessarily.
    return task.class_name.endswith('bike')


# =============================================================================
# Sentinel pause helper
# =============================================================================


async def _wait_for_sentinel_clear(sentinel: Path, *, sleep_s: float = 10.0) -> None:
    """Block while ``sentinel`` exists. Re-checks every ``sleep_s`` seconds.

    The trainer arbiter writes this file during dual-GPU runs to free a
    shared GPU for the trainer. We honor it here because the secondary
    segmenter and the VLM sit on GPUs the trainer needs.
    """
    first = True
    while sentinel.exists():
        if first:
            logger.info('sentinel_present_pausing', path=str(sentinel))
            first = False
        await asyncio.sleep(sleep_s)
    if not first:
        logger.info('sentinel_cleared_resuming', path=str(sentinel))
