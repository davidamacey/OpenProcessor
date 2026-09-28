"""Auto-split sub-module of the curation detection worker.

See ``scripts/curation/region_worker_main.py`` for the entry point and
the ``scripts/curation/worker/`` package for the rest of the split.

The pre-W8 single-candidate cascade (``_process_crop`` + its
``combined._run_combined_cohort_path`` cohort routing) has been deleted
-- ``runner.py``'s streaming pipeline is the only production cascade
(confirmed zero production callers before removal; see the W8 pipeline-
wiring handback report). This module now only keeps the pending-fetch
query, the crop/geometry helpers ``runner.py`` still calls, and the
``SegmenterClient`` / ``SegmenterAllHostsDown`` re-exports other modules
import from here.
"""

from __future__ import annotations

# ruff: noqa: E402
import base64  # noqa: F401  — kept for back-compat re-export surface
import io
from typing import TYPE_CHECKING, Any

from PIL import Image

from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation.class_write_guard import CLASS_GUARD_SOURCE_FIELDS, class_state_token
from src.services.curation.region_boxes import read_boxes
from src.services.curation.region_scope import parent_classes_clause
from src.services.detection.cascade_detect import RegionCandidate
from src.services.detection.profile_registry import get_active_region_profile


logger = get_logger('curation_worker')


from scripts.curation.worker.client import (
    SegmenterAllHostsDown,  # noqa: F401  # back-compat re-export for runner/tests
    SegmenterClient,  # noqa: TC001  # runtime back-compat re-export for shim + tests
)
from scripts.curation.worker.state import JPEG_QUALITY, _ItemTask, items_index


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


def _build_pending_query(exclude_ids: list[str] | None = None) -> dict[str, Any]:
    """Crops needing the worker's attention — pending or pending_verify.

    Skips crops whose region status is already terminal so we never
    overwrite a human or detector verdict on a re-run.

    Deliberately **not** filtered on ``test_holdout`` (P0-3): this query
    feeds the whole worker pipeline, which writes region fields
    unconditionally for every fetched crop. Excluding holdout crops here
    would silently starve them of region detection too, violating the
    region-fields-stay-unconditional rule. The test_holdout guard for
    this worker is scoped to the class-field write path only — see
    ``runner.py:_should_classify`` (checks ``task.test_holdout``).

    None of these clauses score, so they belong in filter context
    (cacheable, no scoring pass) rather than ``must``. ``exclude_ids`` —
    the caller's in-flight set — is pushed server-side via
    ``must_not: {ids: ...}`` instead of being filtered out in Python
    after over-fetching ``batch_size + len(in_flight)`` docs. Items outside
    the profile's ``parent_classes`` are skipped even if seeded pending.
    """
    F = get_region_fields()
    query: dict[str, Any] = {
        'bool': {
            'filter': [
                {'exists': {'field': 'image_path'}},
                {'exists': {'field': 'bbox_norm'}},
                # Pull both legacy short names AND the renamed forms
                # introduced by task #7. Compatibility window: worker
                # consumes whichever name OS happens to carry today;
                # writes only ever emit the new long-form names below.
                {
                    'terms': {
                        F.status: [
                            'pending',
                            RegionStatus.PENDING_DETECTION,
                            'pending_verify',
                            RegionStatus.PENDING_VERIFICATION,
                        ]
                    }
                },
            ],
        },
    }
    profile = get_active_region_profile()
    scope = parent_classes_clause(profile.parent_classes) if profile is not None else None
    if scope is not None:
        query['bool']['filter'].append(scope)
    if exclude_ids:
        query['bool']['must_not'] = [{'ids': {'values': exclude_ids}}]
    return query


async def _fetch_pending(
    opensearch: AsyncOpenSearch,
    *,
    batch_size: int,
    exclude_ids: list[str] | None = None,
    project: Any = None,
) -> list[_ItemTask]:
    """Pull up to ``batch_size`` pending crops, oldest first.

    ``track_total_hits: False`` (the exact match count is never
    read here) and a ``crop_id`` sort tiebreaker for stable ordering
    among same-``created_at`` crops. ``_source`` stays an explicit
    includes list (unlike the VLM worker's ids-only fetch) — this
    worker needs bbox/class fields for every task it dispatches.
    """
    F = get_region_fields()
    body = {
        'size': batch_size,
        '_source': [
            'crop_id',
            'image_path',
            'bbox_norm',
            F.status,
            F.bbox_norm,
            F.score,
            F.revision,
            F.box_seq,
            F.boxes,
            'class_name',
            'class_source',
            'class_validated',
            'confidence',
            'request_id',
            'test_holdout',
            *CLASS_GUARD_SOURCE_FIELDS,
        ],
        'track_total_hits': False,
        'query': _build_pending_query(exclude_ids=exclude_ids),
        'sort': [{'created_at': {'order': 'asc', 'unmapped_type': 'date'}}, {'crop_id': 'asc'}],
    }
    resp = await opensearch.search(index=items_index(), body=body)
    hits = (resp.get('hits') or {}).get('hits') or []
    tasks: list[_ItemTask] = []
    for h in hits:
        src = h.get('_source') or {}
        bbox = src.get('bbox_norm')
        if not bbox or len(bbox) != 4:
            continue
        region_bbox = src.get(F.bbox_norm)
        detector_region_in_source: tuple[float, float, float, float] | None = None
        if isinstance(region_bbox, list) and len(region_bbox) == 4:
            detector_region_in_source = (
                float(region_bbox[0]),
                float(region_bbox[1]),
                float(region_bbox[2]),
                float(region_bbox[3]),
            )
        tasks.append(
            _ItemTask(
                crop_id=h['_id'],
                image_path=str(src.get('image_path') or ''),
                item_bbox_norm=(
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ),
                region_status=src.get(F.status),
                class_name=str(src.get('class_name') or ''),
                class_source=str(src.get('class_source') or ''),
                class_confidence=float(src.get('confidence') or 0.0),
                class_validated=bool(src.get('class_validated') or False),
                test_holdout=bool(src.get('test_holdout') or False),
                detector_region_in_source=detector_region_in_source,
                detector_score=float(src.get(F.score) or 0.0),
                stored_boxes=read_boxes(src, F),
                region_revision=int(src.get(F.revision) or 0),
                region_box_seq=int(src.get(F.box_seq) or 0),
                request_id=str(src.get('request_id') or '-'),
                class_token=class_state_token(src),
                project=project,
            )
        )
    return tasks


# fetch_pending_multi_project lives in fairness.py (700 LOC ceiling).


# =============================================================================
# Per-crop pipeline
# =============================================================================


def _crop_region_jpeg(crop_jpeg: bytes, region_in_crop: tuple[float, float, float, float]) -> bytes:
    """Extract just the region of interest from a crop's JPEG bytes for VLM verify."""
    img = Image.open(io.BytesIO(crop_jpeg))
    img.load()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    cw, ch = img.size
    px1, py1, px2, py2 = region_in_crop
    x1 = max(0, round(px1 * cw))
    y1 = max(0, round(py1 * ch))
    x2 = max(x1 + 1, round(px2 * cw))
    y2 = max(y1 + 1, round(py2 * ch))
    region = img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    region.save(buf, format='JPEG', quality=JPEG_QUALITY)
    return buf.getvalue()


def _source_to_crop(
    region_in_source: tuple[float, float, float, float],
    item_in_source: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Inverse of :func:`crop_norm_to_source_norm` — needed to verify a
    primary-detector region that was already projected into source
    coordinates.

    Used only for ``pending_verify`` crops where the upstream ingest
    wrote the region bbox in source frame; the VLM needs a tight region
    JPEG, which we can only carve from the item crop.
    """
    vx1, vy1, vx2, vy2 = item_in_source
    vw = max(vx2 - vx1, 1e-6)
    vh = max(vy2 - vy1, 1e-6)
    sx1, sy1, sx2, sy2 = region_in_source
    return (
        max(0.0, min(1.0, (sx1 - vx1) / vw)),
        max(0.0, min(1.0, (sy1 - vy1) / vh)),
        max(0.0, min(1.0, (sx2 - vx1) / vw)),
        max(0.0, min(1.0, (sy2 - vy1) / vh)),
    )


# Sub-crop margin around an OCR text hint before re-running the
# secondary segmenter. The segmenter works better with a tighter view
# of the text-bearing region than with the full item crop. 1.8x bounds
# the neighborhood so the region keeps some context (mounting bracket,
# body-color edge) which the grounding model uses to disambiguate
# badges from the real region of interest.
_TEXT_HINT_SUBCROP_MARGIN = 1.8


def _expand_bbox(
    bbox_in_crop: tuple[float, float, float, float], margin: float
) -> tuple[float, float, float, float]:
    """Return an axis-aligned expansion of ``bbox_in_crop`` by ``margin``.

    ``margin=1.8`` means the resulting box is 1.8x the original size,
    centered on the same point, clipped to ``[0, 1]``.
    """
    x1, y1, x2, y2 = bbox_in_crop
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    hw = (x2 - x1) * 0.5 * margin
    hh = (y2 - y1) * 0.5 * margin
    return (
        max(0.0, cx - hw),
        max(0.0, cy - hh),
        min(1.0, cx + hw),
        min(1.0, cy + hh),
    )


def _project_subcrop_box_to_parent(
    box_in_sub: tuple[float, float, float, float],
    sub_in_parent: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Project a sub-crop-frame bbox back into its parent-crop frame."""
    sx1, sy1, sx2, sy2 = sub_in_parent
    sw = max(sx2 - sx1, 1e-6)
    sh = max(sy2 - sy1, 1e-6)
    bx1, by1, bx2, by2 = box_in_sub
    return (
        max(0.0, min(1.0, sx1 + bx1 * sw)),
        max(0.0, min(1.0, sy1 + by1 * sh)),
        max(0.0, min(1.0, sx1 + bx2 * sw)),
        max(0.0, min(1.0, sy1 + by2 * sh)),
    )


async def _resegment_from_text_hint(
    crop_jpeg: bytes,
    hint_in_crop: tuple[float, float, float, float],
    segmenter: SegmenterClient,
) -> tuple[RegionCandidate | None, tuple[float, float, float, float]]:
    """Run the secondary segmenter on a tight sub-crop around an OCR text hint.

    Returns ``(candidate_in_parent_crop, sub_in_parent)``. The candidate
    is projected back to the parent-crop frame so the rest of the worker
    treats it identically to a global hit. ``sub_in_parent`` is
    surfaced for trace logging.
    """
    sub_box = _expand_bbox(hint_in_crop, _TEXT_HINT_SUBCROP_MARGIN)
    sub_jpeg = _crop_region_jpeg(crop_jpeg, sub_box)
    sub_cand = await segmenter.segment(sub_jpeg)
    if sub_cand is None:
        return None, sub_box
    projected = _project_subcrop_box_to_parent(sub_cand.bbox_norm, sub_box)
    return (
        RegionCandidate(bbox_norm=projected, score=sub_cand.score, source=sub_cand.source),
        sub_box,
    )


# =============================================================================
# Bulk write
# =============================================================================
