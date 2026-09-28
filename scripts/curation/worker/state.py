"""Auto-split sub-module of the curation detection worker.

See ``scripts/curation/region_worker_main.py`` for the entry point and
the ``scripts/curation/worker/`` package for the rest of the split.
"""

from __future__ import annotations

import asyncio
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import TERMINAL_STATUSES, RegionStatus, get_curation_config, get_region_fields
from src.config.curation import items_index  # noqa: F401 - re-exported for the worker package
from src.core.logging import get_logger
from src.services.detection.profile_registry import get_active_region_profile


if TYPE_CHECKING:
    from scripts.curation.worker.verify import TaskBoxInput
    from src.config import DetectionProfile
    from src.services.curation.region_boxes import RegionBox
    from src.services.detection.region_text import OcrLine


logger = get_logger('curation_worker')

# Stored ``region_source`` / ``candidate_source`` provenance values are
# defined in src/config/region_source.py (imported above) so both this
# worker and the API router (region_vocabulary.py) share one source of
# truth without a scripts -> src layering violation.

_config = get_curation_config()

DEFAULT_OPENSEARCH = os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200')
DEFAULT_TRITON = os.environ.get('TRITON_URL', 'triton-server:8001')
DEFAULT_SEGMENTER_URL = os.environ.get('OP_SEGMENTER_URL', 'http://sam3:8000')
# Dual-GPU SAM3: OP_SEGMENTER_URLS=http://sam3-gpu0:8000,http://sam3-gpu1:8000
# When set, the worker round-robins requests across all listed URLs so
# the parallelism across GPUs adds up. OP_SEGMENTER_URL is the
# single-URL fallback when OP_SEGMENTER_URLS is unset.
DEFAULT_SEGMENTER_URLS = os.environ.get('OP_SEGMENTER_URLS', '').strip()
DEFAULT_VLM_URL = os.environ.get('OP_VLM_URL', '')
DEFAULT_PAUSE_SENTINEL = Path(
    os.environ.get(
        'OP_WORKER_PAUSE_SENTINEL',
        # Must match CurationConfig.pause_sentinel_path -- the
        # gpu_arbiter writer resolves through the same property.
        str(_config.pause_sentinel_path),
    )
)
JPEG_QUALITY = 90


class RegionProfileNotConfiguredError(RuntimeError):
    """The worker's region cascade ran with no active region profile."""


def region_profile() -> DetectionProfile:
    """The deployment's active region :class:`DetectionProfile`.

    Resolved from ``OP_REGION_PROFILE`` / ``OP_REGION_DETECTION_*`` (see
    :mod:`src.services.detection.profile_registry`). :func:`run` refuses to
    start the cascade without one, so reaching this with none configured
    is a programming error, not a deployment state.
    """
    profile = get_active_region_profile()
    if profile is None:
        msg = (
            'no region profile configured -- set OP_REGION_PROFILE (e.g. to a built-in '
            'reference profile) or OP_REGION_DETECTION_* to enable region detection'
        )
        raise RegionProfileNotConfiguredError(msg)
    return profile


# Status names (renamed to the long form). Worker emits the new
# long-form names everywhere; reads accept both
# legacy and new names until the OS migration completes.
STATUS_PENDING_DETECTION = RegionStatus.PENDING_DETECTION
STATUS_PENDING_VERIFICATION = RegionStatus.PENDING_VERIFICATION
_PENDING_DETECTION_ALIASES = frozenset({STATUS_PENDING_DETECTION, 'pending'})
_PENDING_VERIFICATION_ALIASES = frozenset({STATUS_PENDING_VERIFICATION, 'pending_verify'})

# Terminal statuses the worker MUST NOT override.
_TERMINAL_STATUSES: frozenset[str] = TERMINAL_STATUSES


# =============================================================================
# _ItemTask
# =============================================================================


@dataclass
class _ItemTask:
    """One item pulled from OpenSearch + its in-flight detector results."""

    crop_id: str
    image_path: str
    item_bbox_norm: tuple[float, float, float, float]
    region_status: str | None
    class_name: str
    # Dead field -- nothing writes or maps a ``group`` item field, so
    # this was always empty in production. Kept (default '', never read by
    # _is_secondary_shape) purely for source/test-fixture back-compat;
    # cascade.py's _fetch_pending no longer requests it from OpenSearch.
    group: str = ''
    # X-Request-ID carried over from the HTTP ingest call that produced
    # this crop ('-' for non-HTTP ingests). Bound to structlog contextvars
    # in every consumer's per-task processing block so worker logs can
    # be joined back to the originating request (Phase 4a).
    request_id: str = '-'
    # Cohort marker. ``coco_yolo11_proposal`` means the primary
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
    # Class state this task was read in (class_state_token). The writer
    # applies class fields only if the item still has exactly this state.
    class_token: tuple[Any, ...] | None = None
    # Existing primary-detector candidate (already in source frame) for
    # pending_verify.
    detector_region_in_source: tuple[float, float, float, float] | None = None
    detector_score: float = 0.0
    # Cropped JPEG bytes — built lazily so we don't load images we'd skip.
    crop_jpeg: bytes | None = None
    # Two-stage pipeline state — Stage A (primary/secondary/OCR-det)
    # writes these, Stage B (VLM verify) consumes them.
    candidate_source: str = (
        ''  # 'detector' / 'detector_existing' / 'segmenter' / 'segmenter_text_hint' / ''
    )
    candidate_in_crop: tuple[float, float, float, float] | None = None
    candidate_in_source: tuple[float, float, float, float] | None = None
    candidate_score: float = 0.0
    # text-hint: when the OCR pipeline produced the candidate, its
    # recognized text rides along so writers can persist it even when
    # the downstream VLM verify returns an empty read.
    candidate_text: str | None = None
    candidate_text_confidence: float | None = None
    # W8: every selected candidate for this item (select_region_candidates
    # output, wrapped as TaskBoxInput -- box_id set only when read back
    # from an existing pending_verification box). Drives the multi-box
    # VLM overlay + verdicts_to_boxes write path. The singular
    # candidate_* fields above stay populated with candidates[0] (best
    # candidate) for the no-VLM-configured fallback (accept_without_vlm)
    # and the region-embedding stage, which are still single-box.
    candidates: list[TaskBoxInput] = field(default_factory=list)
    # W8 B1 fix: this item's full stored ``region_boxes`` list, read
    # alongside this task's other fields at fetch time. Path 1
    # (pending_verification) builds its VLM candidates from this list's
    # ``proposed`` boxes -- the real source of truth -- rather than the
    # legacy single scalar. This is a FETCH-TIME SNAPSHOT, used only to
    # decide what to re-verify; the actual write-time merge (never
    # discarding an untouched sibling box) re-reads the live list fresh
    # (see ``bulk_writer._merge`` / M1).
    stored_boxes: list[RegionBox] = field(default_factory=list)
    # Current stored region_revision / region_box_seq high-water marks
    # (read alongside this task's other fields), kept for logging/back-
    # compat -- the actual revision/seq bump at write time now always
    # reads the CURRENT live doc (M1 fix), never this snapshot.
    region_revision: int = 0
    region_box_seq: int = 0
    # W8 B1 + M1 fix: this pass's own resolved box list (not yet merged
    # with any concurrently-stored siblings, ids not yet finalized) plus
    # the status to fall back to once the write-time merge is empty.
    # ``None`` means this write doesn't touch the box list at all (e.g.
    # ``unreadable_crop_update``). Set by ``runner._box_list_doc`` and
    # ``region_text_stage.accept_without_vlm``; consumed by
    # ``bulk_writer._merge``, which re-reads the live doc immediately
    # before the write and merges/mints ids/derives the final status
    # against THAT, never this snapshot.
    pending_boxes: list[RegionBox] | None = None
    pending_empty_status: RegionStatus | None = None
    # W8 B1 fix: whether the write-time merge (bulk_writer._merge) should
    # touch `pending_boxes` against the CURRENT stored list at all
    # (preserving a sibling box this pass never itself decided) or
    # REPLACE the stored list outright with exactly `pending_boxes`.
    # Default False (replace) -- matches every pre-B1 write path's
    # existing behaviour (e.g. `accept_without_vlm`'s DETECTION_FAILED
    # sanity-reject branch, which owns the whole list it writes).
    #
    # True covers TWO distinct cases, disambiguated by `reverify` below
    # (W8c B1/M1 fix, 2026-09-28 re-review): `reverify=True` is Path 1
    # (re-verifying a stored `proposed` box) -- `bulk_writer._merge` calls
    # `merge_boxes_for_write`, which keeps every stored sibling this pass
    # never touched, by id. `reverify=False` is a FRESH detection pass
    # (Path 2/3, or the text-hint re-pass they fall into) -- "merge" is
    # the wrong semantic there: a fresh detection is a new answer to
    # "where are the regions?", so `bulk_writer._merge` instead keeps
    # only stored siblings a human owns (`region_boxes.is_human_owned`)
    # and REPLACES every machine-sourced one with this pass's own
    # `pending_boxes` -- otherwise a requeued item's stale rejected
    # machine box would accumulate forever and keep overriding the new
    # pass's derived status (M1).
    pending_merge: bool = False
    # W8c B1 fix (2026-09-28 re-review): True ONLY for Path 1
    # (`runner.py`'s pending-verification branch, re-verifying a stored
    # `proposed` box). Read at the VLM `region_visible=False` branch
    # (`runner.py`, combined-verify handling) to pick "resolve the
    # re-verified candidate(s) as `rejected`, keeping their stored ids"
    # instead of the fresh-detection "no box, terminal `no_region_visible`"
    # branch. Before this flag existed, that decision was (incorrectly)
    # keyed off `pending_merge` alone -- which every fresh-detection pass
    # ALSO sets (see above) -- so a fresh item with no stored boxes at all
    # that got `region_visible=False` was misrouted into the re-verify
    # branch and wrote a phantom `rejected` box instead of the correct
    # empty `no_region_visible` (B1, the dev/test stack's `fake_vlm`
    # defaults to `region_visible=False`, so this was not an edge case).
    reverify: bool = False
    # A FORCED final status that must win over whatever
    # ``derive_status(merged_boxes)`` would otherwise compute -- only
    # ``accept_without_vlm``'s sanity-gate-reject branch uses this
    # (``detection_failed`` is not part of ``derive_status``'s box-state
    # vocabulary). ``None`` means "derive it from pending_empty_status".
    pending_status: RegionStatus | None = None
    # Detection trace — list of "<detector>:<tag>" strings the task
    # accumulates as it moves through the cascade, serialized as
    # ``RegionFields.detector_chain`` on every write that produces a
    # region bbox.
    detection_trace: list[str] = field(default_factory=list)
    # Class-side update from a combined VLM call. Layered onto
    # ``update_doc`` at write time so subsequent region-cascade writes
    # don't clobber the class fields (cohort = primary-detector-missed).
    combined_class_update: dict[str, Any] = field(default_factory=dict)
    # Item-text OCR: every line read on the item crop this pass (None =
    # not read / read failed), and the item-text fields layered onto
    # whatever region write this pass produces.
    item_ocr_lines: list[OcrLine] | None = None
    item_text_update: dict[str, Any] = field(default_factory=dict)
    # Final outcome to write back. Empty dict means "no update for this crop".
    update_doc: dict[str, Any] = field(default_factory=dict)
    # Which project this item belongs to (projects_plan.md §5.1). Every
    # downstream call for this task -- Triton/segmenter/VLM config reads,
    # OpenSearch reads/writes -- must run inside ``with
    # bind_project(task.project):`` so it resolves this item's own
    # project, not whatever project a sibling task on another consumer
    # happens to be processing. The producer always stamps it; a task
    # without one is a bug and :func:`bind_task_project` refuses it.
    project: Any = None
    # Minor 5 (W2 review, 2026-09-27): True once a VLM call actually ran
    # for this task this pass (visibility gate or combined class+region
    # -- see runner.py's call sites; W8 deleted the separate per-crop
    # verify_region cascade in cascade.py/combined.py). The bulk writer
    # only stamps ``vlm_prompt_pack`` on a write when this is
    # True, so a deployment with no VLM configured (or a write path that
    # skipped the VLM, e.g. the high-confidence segmenter auto-skip) never
    # gets a stamp implying a VLM ran.
    vlm_called: bool = False


def bind_task_project(task: _ItemTask) -> None:
    """Bind ``task``'s project for the rest of the calling consumer task.

    Each pipeline consumer is its own asyncio task, so its contextvars
    binding is private to it; rebinding on every dequeue means per-item
    work always resolves the item's own project.
    """
    from src.config.project_context import set_bound_project

    if task.project is None:
        msg = f'item task {task.crop_id!r} has no project'
        raise ValueError(msg)
    set_bound_project(task.project)


def bound_class_catalog() -> tuple[list[str], dict[str, int]]:
    """The bound project's VLM class catalog: the non-deprecated class
    names (the list sent in the prompt; a reply's ``class_id`` indexes it)
    and ``name -> registry class_id``.

    Resolved per call from the bound project's class registry (cached per
    project, mtime-invalidated), never from a process-wide list: items of
    different projects must be classified against their own registry.
    An unreadable registry yields an empty catalog, which every caller
    already treats as "do not classify".
    """
    from src.clients.curation_opensearch import get_class_registry

    try:
        registry = get_class_registry().load()
    except Exception as exc:
        logger.warning('class_registry_load_failed', error=str(exc))
        return [], {}
    from src.services.curation.region_class import item_classes

    # The region class only ever labels sub-boxes, never a whole item.
    active = item_classes(registry.classes)
    return [c.class_name for c in active], {c.class_name: int(c.class_id) for c in active}


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


def unreadable_crop_update(task: _ItemTask) -> dict[str, Any]:
    """Region update for an item whose source image could not be read.

    ``detection_failed`` (retryable via the requeue tooling), never
    ``no_region_box``: nothing was looked at, so recording "no region
    found" would be a false verdict -- a missing mount once turned every
    item in a batch into one.
    """
    F = get_region_fields()
    return {
        F.status: RegionStatus.DETECTION_FAILED,
        F.reason: 'image_unavailable',
        F.detector_chain: [*task.detection_trace, 'worker:image_unavailable'],
    }


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

    Groups come from the active profile's ``secondary_shape_groups``
    (``OP_REGION_DETECTION_SECONDARY_SHAPE_GROUPS``), which must match the
    class registry's ``group`` values. A profile with no groups routes
    nothing to the secondary-shape path.

    ``task.group`` came from a ``group`` field on the item doc that
    nothing ever writes or maps -- it was always empty, so this always
    fell through to the name-suffix heuristic below. Resolve the group
    from the class registry (the actual source of truth for
    ``class_name -> group``) instead.
    """
    groups = region_profile().secondary_shape_groups
    if not groups:
        return False
    group = _class_group(task.class_name)
    if group:
        return group in groups
    # No registry entry for this class_name (e.g. a stale/renamed class,
    # or a deployment that hasn't backfilled `group` yet). No fallback:
    # a group-less item is simply not routed to the secondary-shape path.
    return False


def _class_group(class_name: str) -> str | None:
    """``class_name -> group`` via the class registry, or ``None`` if the
    class isn't registered."""
    from src.clients.curation_opensearch import get_class_registry

    if not class_name:
        return None
    for entry in get_class_registry().load().classes:
        if entry.class_name == class_name:
            return entry.group or None
    return None


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
