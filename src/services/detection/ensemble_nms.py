"""CPU NMS for a YOLOv5-style raw-output ensemble/backbone detector.

Some backbone detectors are served by Triton with NMS applied
client-side rather than baked into the exported graph — Triton returns
the raw output ``[batch, N, 5 + nc]`` (4 cx/cy/w/h + 1 objectness + nc
class scores, in pixel coordinates relative to the network input) and
this module applies NMS.

The algorithm is a native implementation that follows the documented
semantics of YOLOv5's ``non_max_suppression`` (single-label mode, the
YOLOv5 default); no YOLOv5 code is copied or imported. Per image:

1. Objectness gate: keep candidates with ``obj_conf > conf_thres``.
2. Class confidence is ``cls_score * obj_conf``; take the best class
   per candidate and keep it if that confidence ``> conf_thres``.
3. Convert ``xywh`` to ``xyxy``; apply the optional ``classes`` filter.
4. Sort by confidence descending and cap at ``max_nms`` candidates.
5. Class-aware NMS by offsetting each box by ``class_id * max_wh``
   (skipped when ``agnostic``), then ``torchvision.ops.nms``.
6. Cap at ``max_det`` detections.

Default thresholds follow common YOLOv5 defaults: ``conf_thres=0.25``,
``iou_thres=0.45``, ``max_det=300``.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import torch
import torchvision


_DEFAULT_CONF: Final[float] = 0.25
_DEFAULT_IOU: Final[float] = 0.45
_DEFAULT_MAX_DET: Final[int] = 300
# Larger than any plausible network-input side, so per-class offset boxes
# can never overlap across classes.
_MAX_WH: Final[int] = 7680
_MAX_NMS: Final[int] = 30000


def _xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = xywh.unbind(1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = _DEFAULT_CONF,
    iou_thres: float = _DEFAULT_IOU,
    classes: list[int] | None = None,
    agnostic: bool = False,
    max_det: int = _DEFAULT_MAX_DET,
    max_nms: int = _MAX_NMS,
) -> list[torch.Tensor]:
    """Single-label, class-aware NMS over a ``[B, N, 5 + nc]`` raw output.

    Returns a list (len = batch) of ``(n, 6)`` float32 tensors with rows
    ``x1, y1, x2, y2, conf, class_id``, sorted by ``conf`` descending.
    """
    if not 0.0 <= conf_thres <= 1.0:
        raise ValueError(f'conf_thres must be in [0, 1], got {conf_thres}')
    if not 0.0 <= iou_thres <= 1.0:
        raise ValueError(f'iou_thres must be in [0, 1], got {iou_thres}')
    if prediction.ndim != 3 or prediction.shape[2] < 6:
        raise ValueError(f'expected raw output [B, N, 5 + nc], got {tuple(prediction.shape)}')

    # torchvision's CPU NMS kernel has no half-precision path (FP16 Triton outputs).
    prediction = prediction.float()
    output: list[torch.Tensor] = []
    for image in prediction:
        x = image[image[:, 4] > conf_thres]
        if x.shape[0] == 0:
            output.append(torch.zeros((0, 6), device=prediction.device))
            continue

        cls_conf = x[:, 5:] * x[:, 4:5]
        conf, cls_idx = cls_conf.max(1)
        boxes = _xywh_to_xyxy(x[:, :4])
        det = torch.cat([boxes, conf[:, None], cls_idx[:, None].float()], dim=1)
        det = det[conf > conf_thres]

        if classes is not None:
            wanted = torch.tensor(classes, device=det.device, dtype=det.dtype)
            det = det[(det[:, 5:6] == wanted).any(1)]
        if det.shape[0] == 0:
            output.append(torch.zeros((0, 6), device=prediction.device))
            continue

        det = det[det[:, 4].argsort(descending=True)[:max_nms]]
        offsets = det[:, 5:6] * (0 if agnostic else _MAX_WH)
        keep = torchvision.ops.nms(det[:, :4] + offsets, det[:, 4], iou_thres)
        output.append(det[keep[:max_det]])
    return output


def apply_ensemble_nms(
    raw_output: np.ndarray | torch.Tensor,
    conf_thres: float = _DEFAULT_CONF,
    iou_thres: float = _DEFAULT_IOU,
    max_det: int = _DEFAULT_MAX_DET,
    classes: list[int] | None = None,
    agnostic: bool = False,
) -> list[list[dict]]:
    """Apply YOLOv5-semantics NMS to a raw ensemble-detector output.

    Args:
        raw_output: shape `[B, N, 5 + nc]` (cx, cy, w, h, obj_conf, +nc class
            scores). PIXEL space coordinates relative to the network input
            (e.g. 1280x1280). NumPy or torch.
        conf_thres, iou_thres, max_det, classes, agnostic: passed through to
            :func:`non_max_suppression`.

    Returns:
        List of per-image detection lists. Each detection is a dict with
        keys `box` ([x1, y1, x2, y2] in pixel space), `score`, `class_id`.
        Caller normalizes coords vs. input size for storage.
    """
    if isinstance(raw_output, np.ndarray):
        prediction = torch.from_numpy(raw_output)
    else:
        prediction = raw_output

    nms_out = non_max_suppression(
        prediction,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=classes,
        agnostic=agnostic,
        max_det=max_det,
    )

    results: list[list[dict]] = []
    for det in nms_out:
        per_image: list[dict] = []
        for row in det.tolist():
            x1, y1, x2, y2, score, cls = row
            per_image.append(
                {
                    'box': [x1, y1, x2, y2],
                    'score': float(score),
                    'class_id': int(cls),
                }
            )
        results.append(per_image)
    return results


__all__ = ['apply_ensemble_nms', 'non_max_suppression']
