"""Scoring: one shared pycocotools metric for every backend, per class.

Using a single COCOeval implementation for all models is the whole point
of the bake-off -- it removes per-framework metric differences so accuracy
numbers are directly comparable. Every metric is per eval class; "overall"
blocks are the unweighted mean of per-class AP (equal to COCOeval's own
``stats`` for the same ``catIds``) and micro-averaged P/R/F1 (summed
TP/FP/FN). We also report precision/recall/F1 at the deployed operating
point (conf=0.25, IoU=0.45), which is what operators actually feel, since
mAP integrates over thresholds they never use.

Detections passed to these functions carry EVAL class ids unless a
``class_map`` (model id -> eval id) is given to :func:`detections_to_coco`.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .dataset import category_id


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from .backends.base import Detection
    from .dataset import GtImage


@dataclass(slots=True)
class CocoMetrics:
    """COCO-style detection metrics over a set of classes.

    ``map_*`` are the unweighted mean of the per-class APs. ``ap_small/medium/
    large`` use COCO's area bins (small <32^2, medium 32^2-96^2, large >96^2
    px) and ``recall_100`` is AR@100, both straight from COCOeval ``stats``.
    ``None`` everywhere when no class is in scope.
    """

    map_50_95: float | None
    map_50: float | None
    map_75: float | None
    ap_small: float | None
    ap_medium: float | None
    ap_large: float | None
    recall_100: float | None


@dataclass(slots=True)
class ClassAp:
    """One class's AP at IoU .50, .50:.95 and .75 (area all, maxDets 100)."""

    ap50: float
    ap50_95: float
    ap75: float


@dataclass(slots=True)
class CocoResult:
    overall: CocoMetrics
    per_class: dict[int, ClassAp]


@dataclass(slots=True)
class OperatingPoint:
    """Precision/recall/F1 at a fixed conf + IoU (what deployment uses).

    ``mean_iou`` is the average IoU of the matched (true-positive) boxes --
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


@dataclass(slots=True)
class OperatingPointResult:
    """Per-class operating points plus their micro average (summed TP/FP/FN)."""

    per_class: dict[int, OperatingPoint]
    micro: OperatingPoint


_EMPTY = CocoMetrics(None, None, None, None, None, None, None)


def _mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else 0.0


def coco_eval(
    coco_gt: dict[str, Any], results: list[dict[str, Any]], class_ids: Iterable[int]
) -> CocoResult:
    """Run one COCOeval over ``class_ids`` (eval class ids) and split it per class.

    Args:
        coco_gt: Ground-truth dict from :meth:`YoloTestSet.coco_gt`.
        results: COCO detection records ``{image_id, category_id, bbox, score}``
            with ``bbox`` in absolute ``[x, y, w, h]``.
        class_ids: Eval classes in scope (the model's covered scored classes).
            GT of any other class is ignored.

    Per-class AP comes from ``ev.eval['precision']`` (``[T, R, K, A, M]``;
    area index 0 = all, maxDets index 2 = 100). No detections at all ->
    zero APs, not a crash.
    """
    ids = sorted(set(class_ids))
    if not ids:
        return CocoResult(_EMPTY, {})
    if not results:
        zero = ClassAp(0.0, 0.0, 0.0)
        return CocoResult(CocoMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dict.fromkeys(ids, zero))

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # pycocotools is chatty; swallow its index/eval prints.
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = coco_gt
        gt.createIndex()
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, iouType='bbox')
        ev.params.catIds = [category_id(c) for c in ids]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    precision = ev.eval['precision']  # [T, R, K, A, M], K ordered like params.catIds
    per_class = {
        cid: ClassAp(
            ap50=_mean_valid(precision[0, :, k, 0, 2]),
            ap50_95=_mean_valid(precision[:, :, k, 0, 2]),
            ap75=_mean_valid(precision[5, :, k, 0, 2]),
        )
        for k, cid in enumerate(ids)
    }
    s = ev.stats  # standard 12-entry COCO stats vector
    n = len(per_class)
    overall = CocoMetrics(
        map_50_95=sum(a.ap50_95 for a in per_class.values()) / n,
        map_50=sum(a.ap50 for a in per_class.values()) / n,
        map_75=sum(a.ap75 for a in per_class.values()) / n,
        ap_small=float(s[3]),
        ap_medium=float(s[4]),
        ap_large=float(s[5]),
        recall_100=float(s[8]),
    )
    return CocoResult(overall, per_class)


def _op(conf: float, iou: float, tp: int, fp: int, fn: int, iou_sum: float) -> OperatingPoint:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = iou_sum / tp if tp else 0.0
    return OperatingPoint(conf, iou, precision, recall, f1, mean_iou, tp, fp, fn)


def operating_point(
    gt_images: Sequence[GtImage],
    dets_by_image: Mapping[int, list[Detection]],
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    class_ids: Iterable[int],
) -> OperatingPointResult:
    """Greedy TP/FP/FN matching at a fixed conf + IoU, per eval class.

    Per image and class: keep that class's detections with ``score >= conf``,
    sort by score desc, and greedily match each to the highest-IoU unmatched
    GT box of the same class above ``iou``. Unmatched detections are FP;
    unmatched GT are FN. Detections of classes outside ``class_ids`` are
    ignored. ``micro`` sums TP/FP/FN (and matched IoU) over the classes.
    """
    ids = sorted(set(class_ids))
    counts = {c: [0, 0, 0, 0.0] for c in ids}  # tp, fp, fn, iou_sum
    for img in gt_images:
        dets = dets_by_image.get(img.image_id, [])
        for cid in ids:
            gts = [b.xyxy for b in img.boxes if b.class_id == cid]
            matched = [False] * len(gts)
            mine = sorted(
                (d for d in dets if d.class_id == cid and d.score >= conf),
                key=lambda d: d.score,
                reverse=True,
            )
            c = counts[cid]
            for d in mine:
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gts):
                    if matched[j]:
                        continue
                    i = _iou((d.x1, d.y1, d.x2, d.y2), g)
                    if i > best_iou:
                        best_iou, best_j = i, j
                if best_j >= 0 and best_iou >= iou:
                    matched[best_j] = True
                    c[0] += 1
                    c[3] += best_iou
                else:
                    c[1] += 1
            c[2] += matched.count(False)
    per_class = {
        cid: _op(conf, iou, int(c[0]), int(c[1]), int(c[2]), float(c[3]))
        for cid, c in counts.items()
    }
    micro = _op(
        conf,
        iou,
        sum(int(c[0]) for c in counts.values()),
        sum(int(c[1]) for c in counts.values()),
        sum(int(c[2]) for c in counts.values()),
        sum(float(c[3]) for c in counts.values()),
    )
    return OperatingPointResult(per_class, micro)


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


def detections_to_coco(
    dets_by_image: Mapping[int, list[Detection]], class_map: Mapping[int, int] | None
) -> list[dict[str, Any]]:
    """Flatten per-image detections into COCO result records.

    ``class_map`` translates model class ids to eval class ids (detections
    of unmapped model classes are dropped); ``None`` means the detections
    already carry eval class ids.
    """
    out: list[dict[str, Any]] = []
    for image_id, dets in dets_by_image.items():
        for d in dets:
            eid = d.class_id if class_map is None else class_map.get(d.class_id)
            if eid is None:
                continue
            out.append(
                {
                    'image_id': image_id,
                    'category_id': category_id(eid),
                    'bbox': d.coco_bbox,
                    'score': float(d.score),
                }
            )
    return out


def subset_scope(per_class: Iterable[Mapping[str, Any]], ids: Iterable[int]) -> dict[str, Any]:
    """A metric block restricted to eval classes ``ids`` (the "common" scope).

    ``per_class`` are report rows (``eval_class_id``, ``ap50``, ``ap50_95``,
    ``tp``/``fp``/``fn``). AP = unweighted mean over the classes; P/R/F1 are
    micro (summed TP/FP/FN). No class in scope -> ``n_classes 0`` and null
    metrics.
    """
    wanted = set(ids)
    rows = [r for r in per_class if r['eval_class_id'] in wanted and r.get('covered', True)]
    tp = sum(int(r['tp'] or 0) for r in rows)
    fp = sum(int(r['fp'] or 0) for r in rows)
    fn = sum(int(r['fn'] or 0) for r in rows)
    if not rows:
        return {
            'n_classes': 0,
            'map_50': None,
            'map_50_95': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'tp': 0,
            'fp': 0,
            'fn': 0,
        }
    op = _op(0.0, 0.0, tp, fp, fn, 0.0)
    return {
        'n_classes': len(rows),
        'map_50': sum(float(r['ap50']) for r in rows) / len(rows),
        'map_50_95': sum(float(r['ap50_95']) for r in rows) / len(rows),
        'precision': op.precision,
        'recall': op.recall,
        'f1': op.f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


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
