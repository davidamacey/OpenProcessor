"""Evaluate the region-detection cascade against a ground-truth dataset.

The region cascade (``OP_REGION_PROFILE``) writes one region box per item
into the items index (:class:`~src.config.region_fields.RegionFields`).
This module checks those boxes against YOLO ground truth drawn on the
*whole source frame* (e.g. a single-class region-of-interest dataset with
background images) and reports recall / precision / F1 / mean IoU, the
false-positive gate on background images, and why each missed box was
missed.

Coordinate frames. Boxes come from the item's
``region_boxes`` list (:mod:`src.domain.region_boxes`), whose
element ``bbox_norm`` is ALWAYS stored in source-frame coordinates --
every writer (the cascade, the human PUT-box routes) projects into source
frame before a box is appended to the list, so there is no longer a
per-box (or per-item) stored frame marker to read. ``to_source_frame`` /
:class:`RegionFrameError` / ``CROP_FRAMES`` stay as pure, tested geometry
helpers (a future crop-relative ingestion path would need them again) but
:func:`region_records` no longer calls them -- an item with no boxes at
all (pending, or a terminal box-less status) falls back to the item-level
``RegionFields.status`` field, which the worker still writes as a
same-request denormalization of the box list.

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
from src.services.curation.region_eval_data import (
    CohortImage,
    RegionRecord,
    fetch_regions,
    normalize_status,
    resolve_image_ids,
)
from src.services.curation.region_eval_geometry import greedy_match, iou


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from src.config.curation import CurationConfig
    from src.config.region_fields import RegionFields


DEFAULT_ACCEPTED_STATUSES: tuple[str, ...] = (RegionStatus.DETECTED.value,)
PENDING_VALUES: frozenset[str] = frozenset(s.value for s in PENDING_STATUSES)


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
