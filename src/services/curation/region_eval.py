"""Evaluate the region-detection cascade against a ground-truth dataset.

The region cascade (``OP_REGION_PROFILE``) writes one region box per item
into the items index (:class:`~src.config.region_fields.RegionFields`).
This module checks those boxes against YOLO ground truth drawn on the
*whole source frame* (e.g. a single-class license-plate dataset with
background images) and reports recall / precision / F1 / mean IoU, the
false-positive gate on background images, and why each missed box was
missed.

Coordinate frames. Region boxes are written normalized to the source frame
(``RegionFields.bbox_frame == 'source'``); an absent frame is read the same
way (pre-provenance rows). A crop-relative frame (``'crop'`` / ``'item'``)
is re-projected through the item's own ``bbox_norm`` (the item crop in
source coordinates) with
:func:`~src.services.detection.cascade_detect.crop_norm_to_source_norm`.
Any other frame value raises :class:`RegionFrameError` instead of scoring
boxes in an unknown coordinate system.

Image states (a frame is one cohort image; its items are the objects the
ingest detector found on it):

- ``not_ingested`` — no images-index doc for the path; excluded from metrics.
- ``pending`` — at least one item still in a pending region status; the
  cascade has not finished the frame, so it is excluded from metrics (and
  reported) rather than read as a miss.
- ``no_items`` — ingested, but the ingest detector found no parent item, so
  no region could be detected. Evaluated: every GT box on it is a miss.
- ``settled`` — evaluated.

Items with no region status at all (never queued for region detection) are
counted under ``(none)`` and are not treated as pending.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.config.region_state import PENDING_STATUSES, RegionStatus
from src.services.curation.export_support import scroll_hits
from src.services.detection.cascade_detect import crop_norm_to_source_norm


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Sequence

    from src.config.curation import CurationConfig
    from src.config.region_fields import RegionFields


Box = tuple[float, float, float, float]

DEFAULT_ACCEPTED_STATUSES: tuple[str, ...] = (RegionStatus.DETECTED.value,)
# Pre-rename status strings the worker still reads as pending.
LEGACY_STATUS_ALIASES: dict[str, str] = {
    'pending': RegionStatus.PENDING_DETECTION.value,
    'pending_verify': RegionStatus.PENDING_VERIFICATION.value,
}
PENDING_VALUES: frozenset[str] = frozenset(s.value for s in PENDING_STATUSES)
SOURCE_FRAMES: frozenset[str] = frozenset({'source'})
CROP_FRAMES: frozenset[str] = frozenset({'crop', 'item'})
NO_STATUS = '(none)'
NO_DETECTOR = '(none)'
TERMS_CHUNK = 1000


class RegionFrameError(ValueError):
    """A region box is stored in a coordinate frame this evaluator cannot map."""


# =============================================================================
# Geometry
# =============================================================================


def as_box(value: Any) -> Box | None:
    """``[x1, y1, x2, y2]`` -> tuple, or None when absent / malformed / degenerate."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float) -> Box:
    """YOLO ``cx, cy, w, h`` (normalized) -> ``x1, y1, x2, y2``, clamped to 0..1."""

    def _c(v: float) -> float:
        return min(max(v, 0.0), 1.0)

    return (_c(cx - w / 2), _c(cy - h / 2), _c(cx + w / 2), _c(cy + h / 2))


def parse_yolo_labels(text: str, class_ids: Iterable[int] | None = None) -> list[Box]:
    """Parse a YOLO label file into normalized xyxy boxes.

    Detection rows (``cls cx cy w h``) and segmentation rows (``cls x1 y1 x2
    y2 ...``, reduced to their bounding box) are both accepted. ``class_ids``
    keeps only those classes (``None`` = every row).
    """
    wanted = None if class_ids is None else {int(c) for c in class_ids}
    out: list[Box] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0].startswith('#'):
            continue
        try:
            cls = int(float(parts[0]))
            nums = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if wanted is not None and cls not in wanted:
            continue
        if len(nums) == 4:
            box = yolo_to_xyxy(*nums)
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            xs, ys = nums[0::2], nums[1::2]
            box = (min(xs), min(ys), max(xs), max(ys))
        else:
            continue
        if box[2] > box[0] and box[3] > box[1]:
            out.append(box)
    return out


def greedy_match(
    gt: Sequence[Box], pred: Sequence[Box], thr: float
) -> list[tuple[int, int, float]]:
    """One-to-one matching: highest-IoU pairs first, each box used once, IoU >= ``thr``."""
    pairs = sorted(
        (
            (v, gi, pi)
            for gi, g in enumerate(gt)
            for pi, p in enumerate(pred)
            if (v := iou(g, p)) >= thr and v > 0.0
        ),
        reverse=True,
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    out = []
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi, v))
    return out


def to_source_frame(box: Box, frame: str | None, item_box: Box | None) -> Box:
    """Map a stored region box to source-frame coordinates, or refuse."""
    if frame is None or frame in SOURCE_FRAMES:
        return box
    if frame in CROP_FRAMES:
        if item_box is None:
            raise RegionFrameError(
                f'region box is in frame {frame!r} but its item has no bbox_norm to map it through'
            )
        return crop_norm_to_source_norm(box, item_box)
    raise RegionFrameError(
        f'unknown region bbox frame {frame!r} (known: '
        f'{sorted(SOURCE_FRAMES | CROP_FRAMES)}); refusing to score boxes in an unknown '
        'coordinate system'
    )


# =============================================================================
# Cohort + index reads
# =============================================================================


@dataclass
class CohortImage:
    """One ground-truth frame. ``key`` is the local path the dataset is read from."""

    key: str
    server_path: str
    gt: list[Box]
    split: str = ''
    image_id: str | None = None

    @property
    def positive(self) -> bool:
        return bool(self.gt)


@dataclass
class RegionRecord:
    crop_id: str
    status: str
    detector: str
    score: float | None
    box: Box | None  # source frame; None when the item carries no region box


def normalize_status(raw: Any) -> str:
    if raw is None or raw == '':
        return NO_STATUS
    value = str(raw.value if isinstance(raw, RegionStatus) else raw)
    return LEGACY_STATUS_ALIASES.get(value, value)


def region_record(src: dict[str, Any], fields: RegionFields, doc_id: str = '') -> RegionRecord:
    box = as_box(src.get(fields.bbox_norm))
    if box is not None:
        frame = src.get(fields.bbox_frame)
        box = to_source_frame(
            box, None if frame is None else str(frame), as_box(src.get('bbox_norm'))
        )
    score = src.get(fields.score)
    return RegionRecord(
        crop_id=str(src.get('crop_id') or doc_id),
        status=normalize_status(src.get(fields.status)),
        detector=str(src.get(fields.detector) or NO_DETECTOR),
        score=float(score) if isinstance(score, (int, float)) else None,
        box=box,
    )


def _chunks(values: list[str], n: int = TERMS_CHUNK) -> Iterable[list[str]]:
    for i in range(0, len(values), n):
        yield values[i : i + n]


async def resolve_image_ids(
    opensearch: Any, cohort: list[CohortImage], *, config: CurationConfig
) -> None:
    """Fill ``image_id`` from the images index for entries that lack one."""
    todo = sorted({c.server_path for c in cohort if not c.image_id})
    found: dict[str, str] = {}
    for chunk in _chunks(todo):
        hits = await scroll_hits(
            opensearch,
            index=config.images_index,
            query={'terms': {'image_path': chunk}},
            source=['image_id', 'image_path'],
        )
        for hit in hits:
            src = hit.get('_source') or {}
            if src.get('image_path') and src.get('image_id'):
                found[str(src['image_path'])] = str(src['image_id'])
    for c in cohort:
        if not c.image_id:
            c.image_id = found.get(c.server_path)


async def fetch_regions(
    opensearch: Any,
    image_ids: Iterable[str],
    *,
    config: CurationConfig,
    fields: RegionFields,
) -> dict[str, list[RegionRecord]]:
    """Every item on the given frames, as region records keyed by image_id."""
    source = [
        'crop_id',
        'image_id',
        'image_path',
        'bbox_norm',
        fields.status,
        fields.bbox_norm,
        fields.bbox_frame,
        fields.detector,
        fields.score,
    ]
    out: dict[str, list[RegionRecord]] = defaultdict(list)
    for chunk in _chunks(sorted(set(image_ids))):
        hits = await scroll_hits(
            opensearch,
            index=config.items_index,
            query={'terms': {'image_id': chunk}},
            source=source,
        )
        for hit in hits:
            src = hit.get('_source') or {}
            out[str(src.get('image_id'))].append(region_record(src, fields, str(hit.get('_id'))))
    return dict(out)


# =============================================================================
# Scoring
# =============================================================================


@dataclass
class _Tally:
    images: int = 0
    positives: int = 0
    backgrounds: int = 0
    gt_boxes: int = 0
    not_ingested: int = 0
    pending_images: int = 0
    pending_gt_boxes: int = 0
    no_item_images: int = 0
    no_item_gt_boxes: int = 0
    evaluated_images: int = 0
    evaluated_gt_boxes: int = 0
    predictions: int = 0
    duplicates_merged: int = 0
    tp_05: int = 0
    tp: int = 0
    iou_sum: float = 0.0
    evaluated_backgrounds: int = 0
    backgrounds_with_detection: int = 0
    background_fp_regions: int = 0
    by_detector: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    def metrics(self, thr: float) -> dict[str, Any]:
        def _r(n: int, d: int) -> float | None:
            return round(n / d, 4) if d else None

        recall = _r(self.tp, self.evaluated_gt_boxes)
        precision = _r(self.tp, self.predictions)
        if recall is None or precision is None:
            f1 = 0.0 if 0.0 in (recall, precision) else None
        else:
            f1 = (
                round(2 * recall * precision / (recall + precision), 4)
                if recall + precision
                else 0.0
            )
        out: dict[str, Any] = {
            k: v for k, v in vars(self).items() if k not in ('by_detector', 'iou_sum')
        }
        out.update(
            {
                'iou_threshold': thr,
                'recall_at_0.5': _r(self.tp_05, self.evaluated_gt_boxes),
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'false_positives': self.predictions - self.tp,
                'mean_iou_matched': round(self.iou_sum / self.tp, 4) if self.tp else None,
                'background_fp_image_rate': _r(
                    self.backgrounds_with_detection, self.evaluated_backgrounds
                ),
                'by_detector': {
                    det: {
                        'predictions': c['predictions'],
                        'tp': c['tp'],
                        'fp': c['predictions'] - c['tp'],
                        'precision': _r(c['tp'], c['predictions']),
                    }
                    for det, c in sorted(self.by_detector.items())
                },
            }
        )
        return out


def dedup_predictions(
    records: list[RegionRecord], dedup_iou: float
) -> tuple[list[RegionRecord], int]:
    """Collapse near-identical boxes (overlapping items detecting the same region)."""
    if dedup_iou <= 0.0:
        return records, 0
    kept: list[RegionRecord] = []
    for r in sorted(records, key=lambda r: -(r.score if r.score is not None else -1.0)):
        if r.box is None or any(k.box is not None and iou(k.box, r.box) >= dedup_iou for k in kept):
            continue
        kept.append(r)
    return kept, len(records) - len(kept)


@dataclass
class EvalResult:
    summary: dict[str, Any]
    misses: list[dict[str, Any]]
    false_positives: list[dict[str, Any]]

    @property
    def pending_images(self) -> int:
        return int(self.summary['total']['pending_images'])


def evaluate(
    cohort: list[CohortImage],
    regions: dict[str, list[RegionRecord]],
    *,
    accepted: Iterable[str] = DEFAULT_ACCEPTED_STATUSES,
    iou_threshold: float = 0.5,
    dedup_iou: float = 0.7,
) -> EvalResult:
    """Score the cohort. Pure — ``regions`` is :func:`fetch_regions`' output."""
    accepted_set = {normalize_status(s) for s in accepted}
    tallies: dict[str, _Tally] = defaultdict(_Tally)
    status_items: Counter[str] = Counter()
    status_boxes: Counter[str] = Counter()
    rejected_hits: Counter[str] = Counter()
    misses: list[dict[str, Any]] = []
    fps: list[dict[str, Any]] = []

    for img in cohort:
        groups = (
            [tallies['total'], tallies[f'split:{img.split}']] if img.split else [tallies['total']]
        )
        n_gt = len(img.gt)

        def _bump(attr: str, n: int = 1, *, _groups: list[_Tally] = groups) -> None:
            for t in _groups:
                setattr(t, attr, getattr(t, attr) + n)

        _bump('images')
        _bump('positives' if img.positive else 'backgrounds')
        _bump('gt_boxes', n_gt)
        if not img.image_id:
            _bump('not_ingested')
            continue
        records = regions.get(img.image_id, [])
        for r in records:
            status_items[r.status] += 1
            if r.box is not None:
                status_boxes[r.status] += 1
        if any(r.status in PENDING_VALUES for r in records):
            _bump('pending_images')
            _bump('pending_gt_boxes', n_gt)
            continue
        if not records:
            _bump('no_item_images')
            _bump('no_item_gt_boxes', n_gt)

        preds, merged = dedup_predictions(
            [r for r in records if r.status in accepted_set and r.box is not None], dedup_iou
        )
        others = [r for r in records if r.status not in accepted_set and r.box is not None]
        pred_boxes = [r.box for r in preds if r.box is not None]
        _bump('evaluated_images')
        _bump('evaluated_gt_boxes', n_gt)
        _bump('predictions', len(preds))
        _bump('duplicates_merged', merged)
        _bump('tp_05', len(greedy_match(img.gt, pred_boxes, 0.5)))
        matched = greedy_match(img.gt, pred_boxes, iou_threshold)
        _bump('tp', len(matched))
        for t in groups:
            t.iou_sum += sum(v for _g, _p, v in matched)
        matched_p = {p for _g, p, _v in matched}
        for pi, r in enumerate(preds):
            for t in groups:
                t.by_detector[r.detector]['predictions'] += 1
                t.by_detector[r.detector]['tp'] += int(pi in matched_p)
        if not img.positive:
            _bump('evaluated_backgrounds')
            if preds:
                _bump('backgrounds_with_detection')
                _bump('background_fp_regions', len(preds))
        for pi, r in enumerate(preds):
            if pi not in matched_p:
                fps.append(
                    {
                        'split': img.split,
                        'image': img.key,
                        'server_path': img.server_path,
                        'background': not img.positive,
                        'crop_id': r.crop_id,
                        'detector': r.detector,
                        'score': r.score,
                        'box': list(r.box or ()),
                        'best_gt_iou': round(max((iou(g, r.box) for g in img.gt), default=0.0), 4)
                        if r.box
                        else 0.0,
                    }
                )

        matched_g = {g for g, _p, _v in matched}
        for gi, g in enumerate(img.gt):
            if gi in matched_g:
                continue
            best = max(
                ((iou(g, b), r) for r, b in zip(preds, pred_boxes, strict=True)),
                default=None,
                key=lambda t: t[0],
            )
            best_other = max(
                ((iou(g, r.box), r) for r in others if r.box), default=None, key=lambda t: t[0]
            )
            if not records:
                reason = 'no_items'
            elif best_other is not None and best_other[0] >= iou_threshold:
                reason = f'status:{best_other[1].status}'
                rejected_hits[best_other[1].status] += 1
            elif best is not None and best[0] > 0.0:
                reason = 'low_iou'
            else:
                reason = 'no_region'
            misses.append(
                {
                    'split': img.split,
                    'image': img.key,
                    'server_path': img.server_path,
                    'image_id': img.image_id,
                    'gt_box': [round(v, 6) for v in g],
                    'reason': reason,
                    'best_iou': round(best[0], 4) if best else 0.0,
                    'best_box': list(best[1].box or ()) if best else None,
                    'best_detector': best[1].detector if best else None,
                    'best_other_status': best_other[1].status if best_other else None,
                    'best_other_iou': round(best_other[0], 4) if best_other else None,
                }
            )

    misses.sort(key=lambda m: (m['best_iou'], m['image']))
    total = tallies['total']
    summary: dict[str, Any] = {
        'accepted_statuses': sorted(accepted_set),
        'iou_threshold': iou_threshold,
        'dedup_iou': dedup_iou,
        'total': total.metrics(iou_threshold),
        'splits': {
            k.removeprefix('split:'): t.metrics(iou_threshold)
            for k, t in sorted(tallies.items())
            if k.startswith('split:')
        },
        'by_status': {
            s: {'items': status_items[s], 'with_box': status_boxes[s]} for s in sorted(status_items)
        },
        'missed_but_boxed_by_status': dict(sorted(rejected_hits.items())),
        'miss_reasons': dict(sorted(Counter(m['reason'] for m in misses).items())),
    }
    return EvalResult(summary=summary, misses=misses, false_positives=fps)


async def run_eval(
    opensearch: Any,
    cohort: list[CohortImage],
    *,
    config: CurationConfig,
    fields: RegionFields,
    accepted: Iterable[str] = DEFAULT_ACCEPTED_STATUSES,
    iou_threshold: float = 0.5,
    dedup_iou: float = 0.7,
    wait_pending_s: float = 0.0,
    poll_interval_s: float = 30.0,
    on_poll: Callable[[EvalResult, float], None] | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> EvalResult:
    """Resolve, fetch and score; with ``wait_pending_s`` re-poll until nothing is pending.

    Returns the last evaluation — if the wait timed out it still carries the
    pending counts, so the caller can tell a timeout from a finished cascade.
    """
    accepted = tuple(accepted)
    deadline = clock() + max(0.0, wait_pending_s)
    while True:
        await resolve_image_ids(opensearch, cohort, config=config)
        regions = await fetch_regions(
            opensearch, [c.image_id for c in cohort if c.image_id], config=config, fields=fields
        )
        result = evaluate(
            cohort, regions, accepted=accepted, iou_threshold=iou_threshold, dedup_iou=dedup_iou
        )
        remaining = deadline - clock()
        if on_poll is not None:
            on_poll(result, remaining)
        if result.pending_images == 0 or remaining <= 0:
            result.summary['wait_timed_out'] = result.pending_images > 0 and wait_pending_s > 0
            return result
        await sleep(min(poll_interval_s, remaining))


__all__ = [
    'CROP_FRAMES',
    'DEFAULT_ACCEPTED_STATUSES',
    'NO_STATUS',
    'CohortImage',
    'EvalResult',
    'RegionFrameError',
    'RegionRecord',
    'evaluate',
    'fetch_regions',
    'greedy_match',
    'iou',
    'parse_yolo_labels',
    'resolve_image_ids',
    'run_eval',
    'to_source_frame',
    'yolo_to_xyxy',
]
