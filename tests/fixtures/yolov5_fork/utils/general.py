"""Minimal, self-contained YOLOv5-style NMS for test purposes only.

src.services.detection.ensemble_nms imports ``non_max_suppression`` from
a vendored external fork (location configurable via
``DETECTION_YOLOV5_FORK``) rather than reimplementing NMS, to guarantee
bit-exact behavior with a specific trained model. No such fork is
vendored in this repo (see the allowlist note in
tests/test_dependency_manifest.py), so the module raises at import time
without one.

This fixture stands in for that fork in tests: a plain, textbook
class-aware NMS implementation (confidence = objectness * class score,
per-class box offset trick + torchvision.ops.nms), matching the
signature ``apply_ensemble_nms`` calls. It is not a copy of any
reference/proprietary fork — just enough correct NMS math to exercise
the caller.
"""

from __future__ import annotations

import torch
import torchvision


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: list[int] | None = None,
    agnostic: bool = False,
    max_det: int = 300,
) -> list[torch.Tensor]:
    """Return a list (len = batch) of ``(n, 6)`` tensors: x1,y1,x2,y2,conf,cls."""
    bs = prediction.shape[0]
    nc = prediction.shape[2] - 5
    output = [torch.zeros((0, 6)) for _ in range(bs)]

    for xi in range(bs):
        x = prediction[xi]
        obj_conf = x[:, 4]
        cls_scores = x[:, 5 : 5 + nc]
        cls_conf, cls_idx = cls_scores.max(1)
        conf = obj_conf * cls_conf
        mask = conf > conf_thres
        x = x[mask]
        conf = conf[mask]
        cls_idx = cls_idx[mask]
        if x.shape[0] == 0:
            continue

        cx, cy, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        boxes = torch.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            dim=1,
        )

        if classes is not None:
            keep_cls = torch.zeros(len(cls_idx), dtype=torch.bool)
            for c in classes:
                keep_cls |= cls_idx == c
            boxes, conf, cls_idx = boxes[keep_cls], conf[keep_cls], cls_idx[keep_cls]
        if boxes.shape[0] == 0:
            continue

        if agnostic:
            keep = torchvision.ops.nms(boxes, conf, iou_thres)
        else:
            offset = cls_idx.float().unsqueeze(1) * (boxes.max() + 1.0)
            keep = torchvision.ops.nms(boxes + offset, conf, iou_thres)
        keep = keep[:max_det]

        det = torch.cat(
            [boxes[keep], conf[keep].unsqueeze(1), cls_idx[keep].float().unsqueeze(1)],
            dim=1,
        )
        output[xi] = det
    return output
