#!/usr/bin/env python3
"""Detector quality against ground truth, and how two engines differ on it.

Parity (``parity.py``) says whether a rebuilt engine agrees with a stored
reference; it cannot say whether a disagreement is a loss. This module scores
raw detections against COCO ground truth (AP, recall and precision at the
product confidence floor) and, for two engines, scores every box that moved
by its IoU to the ground truth: better, worse or equal.

Detections are ``{image_id: [(x1, y1, x2, y2, score, class_index), ...]}`` in
pixels of the original image; ``class_index`` is the 0-based detector class,
mapped to a COCO category id by sorted category order (the standard 80-class
order). ``detector_capture.py`` produces them from a running Triton.

    .venv/bin/python scripts/bench/detector_quality.py \\
        --annotations instances_train2017.json \\
        --manifest scripts/datasets/manifests/coco_bench_2000.json \\
        --dets old=old.json --dets rebuilt=rebuilt.json --conf 0.4 --out report.json

The first ``--dets`` is the baseline of every comparison. Read-only.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Det = tuple[float, float, float, float, float, int]
Dets = dict[int, list[Det]]
Box = tuple[float, float, float, float]

# Boxes whose IoU with the baseline engine's box is below this "moved".
MOVED_IOU = 0.99
# A moved box is better or worse only if its IoU to the truth changed by more than this.
GT_IOU_EPS = 0.002
# A detection and a box of the other engine are the same object above this IoU.
PAIR_IOU = 0.5


@dataclass(frozen=True, slots=True)
class GtBox:
    xyxy: Box
    category: int
    crowd: bool


@dataclass(slots=True)
class GroundTruth:
    boxes: dict[int, list[GtBox]]
    sizes: dict[int, tuple[int, int]]


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def class_to_category(categories: list[dict[str, Any]]) -> list[int]:
    """Detector class index -> COCO category id (sorted category order)."""
    return sorted(int(c['id']) for c in categories)


def merge_annotations(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """One annotation document from several COCO splits (image ids are unique across splits)."""
    return {
        'images': [im for d in docs for im in d['images']],
        'annotations': [a for d in docs for a in d['annotations']],
        'categories': docs[0]['categories'],
    }


def build_ground_truth(annotations: dict[str, Any], image_ids: list[int]) -> GroundTruth:
    wanted = set(image_ids)
    boxes: dict[int, list[GtBox]] = {i: [] for i in image_ids}
    sizes = {
        int(im['id']): (int(im['width']), int(im['height']))
        for im in annotations['images']
        if int(im['id']) in wanted
    }
    for ann in annotations['annotations']:
        img = int(ann['image_id'])
        if img not in wanted:
            continue
        x, y, w, h = (float(v) for v in ann['bbox'])
        boxes[img].append(
            GtBox((x, y, x + w, y + h), int(ann['category_id']), bool(ann.get('iscrowd', 0)))
        )
    return GroundTruth(boxes, sizes)


def to_categories(dets: dict[int, list[Det]], cat_ids: list[int]) -> dict[int, list[Det]]:
    """Replace each detection's class index with its COCO category id."""
    return {
        img: [(*d[:5], cat_ids[d[5]]) for d in items if 0 <= d[5] < len(cat_ids)]
        for img, items in dets.items()
    }


def coco_ap(gt: GroundTruth, dets: dict[int, list[Det]], cat_ids: list[int]) -> dict[str, Any]:
    """AP (.50:.95), AP50, AP75, AR@100 and per-class AP; ``dets`` carry category ids."""
    results = [
        {
            'image_id': img,
            'category_id': d[5],
            'bbox': [d[0], d[1], d[2] - d[0], d[3] - d[1]],
            'score': d[4],
        }
        for img, items in dets.items()
        for d in items
    ]
    if not results:
        zero = {'ap': 0.0, 'ap50': 0.0}
        return {
            'ap': 0.0,
            'ap50': 0.0,
            'ap75': 0.0,
            'ar100': 0.0,
            'per_class': dict.fromkeys(cat_ids, zero),
        }

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    anns, n = [], 0
    for img, items in gt.boxes.items():
        for b in items:
            n += 1
            w, h = b.xyxy[2] - b.xyxy[0], b.xyxy[3] - b.xyxy[1]
            anns.append(
                {
                    'id': n,
                    'image_id': img,
                    'category_id': b.category,
                    'bbox': [b.xyxy[0], b.xyxy[1], w, h],
                    'area': w * h,
                    'iscrowd': int(b.crowd),
                }
            )
    doc = {
        'images': [{'id': i, 'width': s[0], 'height': s[1]} for i, s in gt.sizes.items()],
        'annotations': anns,
        'categories': [{'id': c} for c in cat_ids],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = doc
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.params.catIds = cat_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    prec = ev.eval['precision']  # [T, R, K, A, M]; area 0 = all, maxDets index 2 = 100

    def _mean(a: Any) -> float:
        valid = a[a > -1]
        return float(valid.mean()) if valid.size else 0.0

    per_class = {
        cid: {'ap': _mean(prec[:, :, k, 0, 2]), 'ap50': _mean(prec[0, :, k, 0, 2])}
        for k, cid in enumerate(cat_ids)
    }
    return {
        'ap': float(ev.stats[0]),
        'ap50': float(ev.stats[1]),
        'ap75': float(ev.stats[2]),
        'ar100': float(ev.stats[8]),
        'per_class': per_class,
    }


def _crowd_covers(det: Det, crowds: list[GtBox]) -> bool:
    area = (det[2] - det[0]) * (det[3] - det[1])
    if area <= 0:
        return False
    for c in crowds:
        if c.category != det[5]:
            continue
        ix = max(0.0, min(det[2], c.xyxy[2]) - max(det[0], c.xyxy[0]))
        iy = max(0.0, min(det[3], c.xyxy[3]) - max(det[1], c.xyxy[1]))
        if ix * iy / area >= PAIR_IOU:
            return True
    return False


def operating_point(
    gt: GroundTruth,
    dets: dict[int, list[Det]],
    *,
    conf: float,
    iou_thr: float,
    cat_ids: list[int],
) -> dict[str, Any]:
    """Greedy TP/FP/FN at ``conf`` and ``iou_thr``; detections on crowd regions are ignored."""
    counts = {c: [0, 0, 0] for c in cat_ids}
    iou_sum = 0.0
    for img, items in gt.boxes.items():
        solid = [b for b in items if not b.crowd]
        crowds = [b for b in items if b.crowd]
        used = [False] * len(solid)
        for d in sorted((x for x in dets.get(img, []) if x[4] >= conf), key=lambda x: -x[4]):
            if d[5] not in counts:
                continue
            best, best_j = 0.0, -1
            for j, g in enumerate(solid):
                if used[j] or g.category != d[5]:
                    continue
                v = iou(d[:4], g.xyxy)
                if v > best:
                    best, best_j = v, j
            if best_j >= 0 and best >= iou_thr:
                used[best_j] = True
                counts[d[5]][0] += 1
                iou_sum += best
            elif not _crowd_covers(d, crowds):
                counts[d[5]][1] += 1
        for j, g in enumerate(solid):
            if not used[j] and g.category in counts:
                counts[g.category][2] += 1
    tp = sum(c[0] for c in counts.values())
    fp = sum(c[1] for c in counts.values())
    fn = sum(c[2] for c in counts.values())
    return {
        'conf': conf,
        'iou': iou_thr,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': tp / (tp + fp) if tp + fp else 0.0,
        'recall': tp / (tp + fn) if tp + fn else 0.0,
        'mean_tp_iou': iou_sum / tp if tp else 0.0,
        'per_class_recall': {
            c: (v[0] / (v[0] + v[2]) if v[0] + v[2] else None) for c, v in counts.items()
        },
    }


def _best_gt_iou(det: Det, solid: list[GtBox]) -> float:
    return max((iou(det[:4], g.xyxy) for g in solid if g.category == det[5]), default=0.0)


def _pair(a: list[Det], b: list[Det]) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Same-class greedy pairing by descending IoU at PAIR_IOU and above."""
    cand = sorted(
        (
            (iou(x[:4], y[:4]), i, j)
            for i, x in enumerate(a)
            for j, y in enumerate(b)
            if x[5] == y[5]
        ),
        reverse=True,
    )
    ua, ub = set(range(len(a))), set(range(len(b)))
    pairs: list[tuple[int, int, float]] = []
    for v, i, j in cand:
        if v < PAIR_IOU:
            break
        if i in ua and j in ub:
            ua.discard(i)
            ub.discard(j)
            pairs.append((i, j, v))
    return pairs, ua, ub


def compare(
    gt: GroundTruth, base: dict[int, list[Det]], other: dict[int, list[Det]], *, conf: float
) -> dict[str, Any]:
    """Score every box of ``other`` that moved relative to ``base`` against the truth."""
    matched = moved = better = worse = equal = 0
    deltas: list[float] = []
    only = {'only_a': [0, 0], 'only_b': [0, 0]}  # on truth, off truth
    for img in gt.boxes:
        solid = [g for g in gt.boxes[img] if not g.crowd]
        a = [d for d in base.get(img, []) if d[4] >= conf]
        b = [d for d in other.get(img, []) if d[4] >= conf]
        pairs, left_a, left_b = _pair(a, b)
        for i, j, v in pairs:
            matched += 1
            if v >= MOVED_IOU:
                continue
            moved += 1
            delta = _best_gt_iou(b[j], solid) - _best_gt_iou(a[i], solid)
            deltas.append(delta)
            if delta > GT_IOU_EPS:
                better += 1
            elif delta < -GT_IOU_EPS:
                worse += 1
            else:
                equal += 1
        for key, left, items in (('only_a', left_a, a), ('only_b', left_b, b)):
            for k in left:
                only[key][0 if _best_gt_iou(items[k], solid) >= PAIR_IOU else 1] += 1

    def _split(v: list[int]) -> dict[str, int]:
        return {'total': v[0] + v[1], 'on_truth': v[0], 'off_truth': v[1]}

    return {
        'conf': conf,
        'matched': matched,
        'moved': moved,
        'moved_better': better,
        'moved_worse': worse,
        'moved_equal': equal,
        'mean_gt_iou_delta_moved': sum(deltas) / len(deltas) if deltas else 0.0,
        'only_a': _split(only['only_a']),
        'only_b': _split(only['only_b']),
    }


def _load_dets(path: str) -> dict[int, list[Det]]:
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    return {int(k): [tuple(d) for d in v] for k, v in raw.items()}


def evaluate(
    gt: GroundTruth,
    engines: dict[str, dict[int, list[Det]]],
    cat_ids: list[int],
    conf: float,
) -> dict[str, Any]:
    """Per-engine quality plus every later engine compared with the first."""
    converted = {name: to_categories(d, cat_ids) for name, d in engines.items()}
    report: dict[str, Any] = {'conf': conf, 'images': len(gt.boxes), 'engines': {}, 'compare': {}}
    for name, d in converted.items():
        ap = coco_ap(gt, d, cat_ids)
        report['engines'][name] = {
            'ap': ap,
            'at_conf': operating_point(gt, d, conf=conf, iou_thr=0.5, cat_ids=cat_ids),
            'at_conf_iou75': operating_point(gt, d, conf=conf, iou_thr=0.75, cat_ids=cat_ids),
            'at_025': operating_point(gt, d, conf=0.25, iou_thr=0.5, cat_ids=cat_ids),
            'detections_at_conf': sum(1 for v in d.values() for x in v if x[4] >= conf),
        }
    names = list(converted)
    for name in names[1:]:
        report['compare'][f'{names[0]}->{name}'] = compare(
            gt, converted[names[0]], converted[name], conf=conf
        )
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        '--annotations',
        action='append',
        required=True,
        help='COCO instances json; repeat to cover every split the manifest draws from',
    )
    p.add_argument('--manifest', required=True, help='pinned manifest json (image_id list)')
    p.add_argument(
        '--split',
        help='score only manifest images of this split (val2017: unseen by COCO-trained models)',
    )
    p.add_argument('--dets', action='append', required=True, metavar='NAME=PATH')
    p.add_argument('--conf', type=float, default=0.4, help='product confidence floor')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)

    annotations = merge_annotations(
        [json.loads(Path(a).read_text(encoding='utf-8')) for a in args.annotations]
    )
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    image_ids = [
        int(r['image_id']) for r in manifest['images'] if not args.split or r['split'] == args.split
    ]
    cat_ids = class_to_category(annotations['categories'])
    gt = build_ground_truth(annotations, image_ids)
    engines = {}
    for spec in args.dets:
        name, _, path = spec.partition('=')
        engines[name] = {i: d for i, d in _load_dets(path).items() if i in gt.boxes}
    report = evaluate(gt, engines, cat_ids, args.conf)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    for name, e in report['engines'].items():
        o = e['at_conf']
        sys.stdout.write(
            f'{name}: AP {e["ap"]["ap"]:.4f} AP50 {e["ap"]["ap50"]:.4f} '
            f'P {o["precision"]:.4f} R {o["recall"]:.4f}\n'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
