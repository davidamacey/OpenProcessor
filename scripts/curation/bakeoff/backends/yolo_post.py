"""Shared YOLO pre/post-processing for raw-tensor backends (Triton/ONNX).

Ultralytics-loaded models handle their own letterbox + decode; this module
is only for backends that receive raw tensors (the deployed Triton TRT
engine), so the decode must be explicit and match how the model was
exported. Kept deliberately small and dependency-light (numpy + cv2).
"""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    image_rgb: np.ndarray, size: int = 640, *, color: int = 114
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize-with-pad to a square ``size`` keeping aspect ratio.

    Returns the letterboxed image plus the scale and (pad_x, pad_y) needed
    to map model-space boxes back to the original frame.
    """
    h, w = image_rgb.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, (pad_x, pad_y)


def to_input_tensor(letterboxed_rgb: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> NCHW float32 [0,1], contiguous, batch dim added."""
    arr = letterboxed_rgb.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None]
    return np.ascontiguousarray(arr)


def decode_yolo_v11(
    output: np.ndarray,
    *,
    scale: float,
    pad: tuple[int, int],
    input_size: int,
    conf_thresh: float,
    coords_normalized: bool,
    num_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a YOLOv8/v11 head ``[1, 4 + nc, N]`` -> boxes + scores + classes.

    Rows are ``(cx, cy, w, h, cls_0, ..., cls_{nc-1})``; the transposed
    ``[1, N, 4 + nc]`` layout is accepted too. The channel axis is the one of
    length ``4 + num_classes`` when given, else the shorter axis (anchor
    counts are in the thousands). Each anchor's score is its max class
    score and its class the argmax. ``coords_normalized`` selects whether
    the box rows are in [0,1] of the letterboxed input (the training export
    convention) or in input pixels; either way boxes are mapped back to the
    ORIGINAL frame's pixel xyxy. NMS is applied by the caller.

    Returns ``(boxes_xyxy[K,4], scores[K], class_ids[K])`` before NMS.
    """
    pred = output[0]
    if num_classes is not None:
        if pred.shape[0] != 4 + num_classes and pred.shape[1] == 4 + num_classes:
            pred = pred.T
    elif pred.shape[0] > pred.shape[1]:
        pred = pred.T
    cls_scores = pred[4:]
    conf = cls_scores.max(axis=0)
    class_ids = cls_scores.argmax(axis=0).astype(np.int64)
    keep = conf >= conf_thresh
    cx, cy, w, h = pred[0][keep], pred[1][keep], pred[2][keep], pred[3][keep]
    conf, class_ids = conf[keep], class_ids[keep]
    if coords_normalized:
        cx, cy, w, h = cx * input_size, cy * input_size, w * input_size, h * input_size
    pad_x, pad_y = pad
    x1 = (cx - w / 2 - pad_x) / scale
    y1 = (cy - h / 2 - pad_y) / scale
    x2 = (cx + w / 2 - pad_x) / scale
    y2 = (cy + h / 2 - pad_y) / scale
    return np.stack([x1, y1, x2, y2], axis=1), conf, class_ids


def decode_yolo26_e2e(
    output: np.ndarray, *, scale: float, pad: tuple[int, int], conf_thresh: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a YOLO26 NMS-free head ``[1, max_det, 6]`` -> boxes + scores + classes.

    YOLO26 is NMS-free: the exported graph emits up to ``max_det`` already-decoded
    detections ``(x1, y1, x2, y2, score, class)`` in the letterboxed input's pixel
    space (zero-padded rows for the unused slots). We filter by score and map the
    boxes back to the ORIGINAL frame's pixel xyxy. No external NMS is needed.

    Returns ``(boxes_xyxy[K,4], scores[K], class_ids[K])``.
    """
    pred = output[0] if output.ndim == 3 else output  # [max_det, 6]
    boxes = pred[:, :4].astype(np.float32, copy=True)
    scores = pred[:, 4].astype(np.float32)
    class_ids = np.rint(pred[:, 5]).astype(np.int64)
    keep = scores >= conf_thresh
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    return boxes, scores, class_ids


def nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float,
    class_ids: np.ndarray | None = None,
) -> list[int]:
    """Greedy NMS; returns indices to keep (sorted by score desc).

    With ``class_ids`` the suppression is per class (boxes of different
    classes never suppress each other), via the usual coordinate-offset
    trick: each class is shifted into its own disjoint region.
    """
    if len(boxes_xyxy) == 0:
        return []
    if class_ids is not None:
        span = float(np.abs(boxes_xyxy).max()) * 2 + 1
        offset = np.asarray(class_ids, dtype=np.float64)[:, None] * span
        boxes_xyxy = boxes_xyxy.astype(np.float64) + offset
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou < iou_thresh]
    return keep
