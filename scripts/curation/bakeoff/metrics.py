"""Scoring: one shared pycocotools metric for every backend.

Using a single COCOeval implementation for all models is the whole point
of the bake-off — it removes per-framework metric differences so accuracy
numbers are directly comparable. We also report precision/recall/F1 at the
deployed operating point (conf=0.25, IoU=0.45), which is what operators
actually feel, since mAP integrates over thresholds they never use.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from .backends.base import Detection
    from .dataset import GtImage


@dataclass(slots=True)
class CocoMetrics:
    """COCO-style detection metrics (single class).

    ``ap_small/medium/large`` use COCO's area bins (small <32^2, medium
    32^2-96^2, large >96^2 px). For a small-object target in full frames
    most boxes are small, so ``ap_small`` is the
    discriminating localization number; ``map_50_95`` rewards tighter
    boxes (averaged over IoU).
    """

    map_50_95: float
    map_50: float
    map_75: float
    ap_small: float
    ap_medium: float
    ap_large: float
    recall_100: float


@dataclass(slots=True)
class OperatingPoint:
    """Precision/recall/F1 at a fixed conf + IoU (what deployment uses).

    ``mean_iou`` is the average IoU of the matched (true-positive) boxes —
    a direct localization-tightness score on top of the count-based P/R/F1.
    """

    conf: float
    iou: float
    precision: float
    recall: float
    f1: float
    mean_iou: float
    tp: int
    fp: int
    fn: int


def coco_eval(coco_gt: dict[str, Any], results: list[dict[str, Any]]) -> CocoMetrics:
    """Run pycocotools COCOeval over detection results.

    Args:
        coco_gt: Ground-truth dict from :meth:`YoloTestSet.coco_gt`.
        results: COCO detection records ``{image_id, category_id, bbox, score}``
            with ``bbox`` in absolute ``[x, y, w, h]``.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # pycocotools is chatty; swallow its index/eval prints.
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = coco_gt
        gt.createIndex()
        if results:
            dt = gt.loadRes(results)
        else:
            # No detections at all -> zero metrics without crashing COCOeval.
            return CocoMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ev = COCOeval(gt, dt, iouType='bbox')
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats  # standard 12-entry COCO stats vector
    return CocoMetrics(
        map_50_95=float(s[0]),
        map_50=float(s[1]),
        map_75=float(s[2]),
        ap_small=float(s[3]),
        ap_medium=float(s[4]),
        ap_large=float(s[5]),
        recall_100=float(s[8]),
    )


def operating_point(
    gt_images: list[GtImage],
    dets_by_image: dict[int, list[Detection]],
    *,
    conf: float = 0.25,
    iou: float = 0.45,
) -> OperatingPoint:
    """Greedy TP/FP/FN matching at a fixed conf + IoU, single class.

    Per image: keep detections with ``score >= conf``, sort by score desc,
    and greedily match each to the highest-IoU unmatched GT box above
    ``iou``. Unmatched detections are FP; unmatched GT are FN.
    """
    tp = fp = fn = 0
    iou_sum = 0.0  # sum of IoU over true-positive matches (localization tightness)
    for img in gt_images:
        gts = list(img.boxes)
        matched = [False] * len(gts)
        dets = sorted(
            (d for d in dets_by_image.get(img.image_id, []) if d.score >= conf),
            key=lambda d: d.score,
            reverse=True,
        )
        for d in dets:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if matched[j]:
                    continue
                i = _iou((d.x1, d.y1, d.x2, d.y2), g)
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_j >= 0 and best_iou >= iou:
                matched[best_j] = True
                tp += 1
                iou_sum += best_iou
            else:
                fp += 1
        fn += matched.count(False)
    mean_iou = iou_sum / tp if tp else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return OperatingPoint(conf, iou, precision, recall, f1, mean_iou, tp, fp, fn)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detections_to_coco(dets_by_image: dict[int, list[Detection]]) -> list[dict[str, Any]]:
    """Flatten per-image detections into COCO result records."""
    out: list[dict[str, Any]] = []
    for image_id, dets in dets_by_image.items():
        out.extend(
            {
                'image_id': image_id,
                'category_id': 1,
                'bbox': d.coco_bbox,
                'score': float(d.score),
            }
            for d in dets
        )
    return out


def percentiles(values: list[float]) -> dict[str, float]:
    """p50/p90/p99 + mean for a latency sample (ms)."""
    if not values:
        return {'mean': 0.0, 'p50': 0.0, 'p90': 0.0, 'p99': 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        'mean': float(arr.mean()),
        'p50': float(np.percentile(arr, 50)),
        'p90': float(np.percentile(arr, 90)),
        'p99': float(np.percentile(arr, 99)),
    }
