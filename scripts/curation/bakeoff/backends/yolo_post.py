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
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a single-class YOLOv11 head ``[1, 5, N]`` -> boxes + scores.

    Output rows are ``(cx, cy, w, h, conf)``. ``coords_normalized`` selects
    whether those are in [0,1] of the letterboxed input (the training export
    convention) or in input pixels; either way boxes are mapped back to the
    ORIGINAL frame's pixel xyxy. NMS is applied by the caller.

    Returns ``(boxes_xyxy[K,4], scores[K])`` before NMS.
    """
    pred = output[0]  # [5, N]
    if pred.shape[0] != 5:
        pred = pred.T if pred.shape[1] == 5 else pred
    cx, cy, w, h, conf = pred[0], pred[1], pred[2], pred[3], pred[4]
    keep = conf >= conf_thresh
    cx, cy, w, h, conf = cx[keep], cy[keep], w[keep], h[keep], conf[keep]
    if coords_normalized:
        cx, cy, w, h = cx * input_size, cy * input_size, w * input_size, h * input_size
    pad_x, pad_y = pad
    x1 = (cx - w / 2 - pad_x) / scale
    y1 = (cy - h / 2 - pad_y) / scale
    x2 = (cx + w / 2 - pad_x) / scale
    y2 = (cy + h / 2 - pad_y) / scale
    return np.stack([x1, y1, x2, y2], axis=1), conf


def decode_yolo26_e2e(
    output: np.ndarray, *, scale: float, pad: tuple[int, int], conf_thresh: float
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a YOLO26 NMS-free head ``[1, max_det, 6]`` -> boxes + scores.

    YOLO26 is NMS-free: the exported graph emits up to ``max_det`` already-decoded
    detections ``(x1, y1, x2, y2, score, class)`` in the letterboxed input's pixel
    space (zero-padded rows for the unused slots). We filter by score and map the
    boxes back to the ORIGINAL frame's pixel xyxy. No external NMS is needed.

    Returns ``(boxes_xyxy[K,4], scores[K])``.
    """
    pred = output[0] if output.ndim == 3 else output  # [max_det, 6]
    boxes = pred[:, :4].astype(np.float32, copy=True)
    scores = pred[:, 4].astype(np.float32)
    keep = scores >= conf_thresh
    boxes, scores = boxes[keep], scores[keep]
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    return boxes, scores


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS; returns indices to keep (sorted by score desc)."""
    if len(boxes_xyxy) == 0:
        return []
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
