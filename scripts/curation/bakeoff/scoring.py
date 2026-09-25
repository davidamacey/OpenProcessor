"""Score one model's detections on one dataset, through its class mapping.

Turns raw detections (model class ids) plus a :class:`ClassMapping` into the
report blocks every comparison row carries: ``overall`` (the model's own
covered classes), ``per_class`` (one row per scored eval class, covered or
not), ``coverage`` (what could not be scored and why) and ``per_stratum``.
Nothing is dropped silently: uncovered eval classes get null-metric rows,
predictions of unmapped model classes are counted per class, predictions
mapped to an eval class outside the scored set are counted too.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .metrics import coco_eval, detections_to_coco, operating_point


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from .backends.base import Detection
    from .class_map import ClassMapping
    from .dataset import YoloTestSet


_NULL_CLASS_METRICS = ('ap50', 'ap50_95', 'ap75', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn')
_NULL_OVERALL = (
    'map_50',
    'map_50_95',
    'map_75',
    'ap_small',
    'ap_medium',
    'ap_large',
    'precision',
    'recall',
    'f1',
    'mean_iou',
)


def map_detections(
    dets_by_image: Mapping[int, list[Detection]],
    model_to_eval: Mapping[int, int],
    scored: set[int],
) -> tuple[dict[int, list[Detection]], Counter[int], int]:
    """Model-id detections -> eval-id detections of scored classes.

    Returns ``(mapped, unmapped_counts_by_model_class, n_outside_scored)``.
    """
    mapped: dict[int, list[Detection]] = {}
    unmapped: Counter[int] = Counter()
    outside = 0
    for image_id, dets in dets_by_image.items():
        keep: list[Detection] = []
        for d in dets:
            eid = model_to_eval.get(d.class_id)
            if eid is None:
                unmapped[d.class_id] += 1
            elif eid not in scored:
                outside += 1
            else:
                keep.append(replace(d, class_id=eid))
        mapped[image_id] = keep
    return mapped, unmapped, outside


def score(
    ds: YoloTestSet,
    dets_by_image: Mapping[int, list[Detection]],
    mapping: ClassMapping,
    *,
    scored_class_ids: Iterable[int],
    conf: float = 0.25,
    iou: float = 0.45,
) -> dict[str, Any]:
    """The ``overall`` / ``per_class`` / ``coverage`` / ``per_stratum`` report blocks."""
    scored = set(scored_class_ids)
    m2e = mapping.model_to_eval or {}
    mapped, unmapped_counts, outside = map_detections(dets_by_image, m2e, scored)
    covered = sorted(scored & set(m2e.values()))
    coco = coco_eval(ds.coco_gt(sorted(scored)), detections_to_coco(mapped, None), covered)
    op = operating_point(ds.images, mapped, conf=conf, iou=iou, class_ids=covered)

    n_gt = ds.n_gt_by_class
    model_ids_by_eval: dict[int, list[int]] = defaultdict(list)
    for mid, eid in sorted(m2e.items()):
        model_ids_by_eval[eid].append(mid)
    per_class: list[dict[str, Any]] = []
    for cid in sorted(scored):
        row: dict[str, Any] = {
            'eval_class_id': cid,
            'name': ds.class_names.get(cid, str(cid)),
            'n_gt': n_gt.get(cid, 0),
            'covered': cid in coco.per_class,
            'model_class_ids': model_ids_by_eval.get(cid, []) if cid in coco.per_class else [],
        }
        if cid in coco.per_class:
            ap, p = coco.per_class[cid], op.per_class[cid]
            row.update(
                ap50=ap.ap50,
                ap50_95=ap.ap50_95,
                ap75=ap.ap75,
                precision=p.precision,
                recall=p.recall,
                f1=p.f1,
                tp=p.tp,
                fp=p.fp,
                fn=p.fn,
            )
        else:
            row.update(dict.fromkeys(_NULL_CLASS_METRICS))
        per_class.append(row)

    overall: dict[str, Any] = {'n_classes': len(covered)}
    if covered:
        o, m = coco.overall, op.micro
        overall.update(
            map_50=o.map_50,
            map_50_95=o.map_50_95,
            map_75=o.map_75,
            ap_small=o.ap_small,
            ap_medium=o.ap_medium,
            ap_large=o.ap_large,
            precision=m.precision,
            recall=m.recall,
            f1=m.f1,
            mean_iou=m.mean_iou,
            tp=m.tp,
            fp=m.fp,
            fn=m.fn,
        )
    else:
        overall.update(dict.fromkeys(_NULL_OVERALL), tp=0, fp=0, fn=0)

    unmapped: dict[int, dict[str, Any]] = {
        int(u['model_class_id']): {**u, 'n_predictions': 0} for u in mapping.unmapped_model_classes
    }
    for mid, n in unmapped_counts.items():
        unmapped.setdefault(mid, {'model_class_id': mid, 'name': None, 'n_predictions': 0})
        unmapped[mid]['n_predictions'] = n
    coverage = {
        'n_eval_classes': len(scored),
        'n_covered': len(covered),
        'not_covered': [
            {'eval_class_id': r['eval_class_id'], 'name': r['name']}
            for r in per_class
            if not r['covered']
        ],
        'unmapped_model_classes': [unmapped[k] for k in sorted(unmapped)],
        'predictions_outside_scored_classes': outside,
    }
    return {
        'overall': overall,
        'per_class': per_class,
        'coverage': coverage,
        'per_stratum': _per_stratum(ds, mapped, sorted(scored), covered, conf=conf, iou=iou)
        if ds.stratum_map
        else {},
    }


def _per_stratum(
    ds: YoloTestSet,
    mapped: Mapping[int, list[Detection]],
    scored: list[int],
    covered: list[int],
    *,
    conf: float,
    iou: float,
) -> dict[str, dict[str, Any]]:
    """mAP@0.5 + micro operating-point P/R per stratum (cluster), covered classes."""
    groups: dict[str, list[Any]] = defaultdict(list)
    for img in ds.images:
        groups[ds.stratum_for(img)].append(img)
    out: dict[str, dict[str, Any]] = {}
    for stratum, imgs in sorted(groups.items()):
        sub = {i.image_id: mapped.get(i.image_id, []) for i in imgs}
        m = coco_eval(ds.coco_gt(scored, images=imgs), detections_to_coco(sub, None), covered)
        op = operating_point(imgs, sub, conf=conf, iou=iou, class_ids=covered)
        out[stratum] = {
            'n_images': len(imgs),
            'map_50': m.overall.map_50,
            'recall': op.micro.recall,
            'precision': op.micro.precision,
        }
    return out
