"""Text boxes from the detector's probability map: PaddleOCR's DB post-processing.

A port of ``DBPostProcess.boxes_from_bitmap`` (``box_type="quad"``) of PaddleX release/3.3
(``paddlex/inference/models/text_detection/processors.py``, the code PaddleOCR 3.x runs for
PP-OCRv6): binarise the map at ``thresh`` (optionally dilated with a 2 x 2 kernel), take OpenCV's
``findContours`` (``RETR_LIST``, ``CHAIN_APPROX_SIMPLE``), score each contour's minimum-area
rectangle with the mean map value inside it (``box_score_fast``), unclip the kept ones by
``unclip_ratio`` with pyclipper (``JT_ROUND``), take the minimum-area rectangle of that again, and scale it to
the source frame with rounding. The map is small (a side at most the detector size), so this runs
on the host.

Differences from the Triton ``ocr_pipeline`` post-process (``models/ocr_pipeline/1/ocr_postprocess.py``),
measured in ``docs/OCR.md``: the score is the mean over the box rather than the centre pixel, there is
the size gate before the unclip (3) and after it (5), the corners are ordered by ``get_mini_boxes`` and
the scaled corners are rounded and clipped to the frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import pyclipper


MIN_SIZE = 3


@dataclass(frozen=True)
class DbParams:
    """PP-OCRv6 small ``inference.yml`` ``PostProcess``: thresh 0.2, box_thresh 0.45,
    unclip_ratio 1.4, max_candidates 3000; ``use_dilation`` is the class default (off)."""

    thresh: float = 0.2
    box_thresh: float = 0.45
    unclip_ratio: float = 1.4
    max_candidates: int = 3000
    use_dilation: bool = False


def _mini_box(contour: np.ndarray) -> tuple[list[np.ndarray], float]:
    """The four corners of the minimum-area rectangle, top-left, top-right, bottom-right,
    bottom-left (``get_mini_boxes``), and its shorter side."""
    rect = cv2.minAreaRect(contour)
    pts = sorted(cv2.boxPoints(rect), key=lambda p: p[0])
    i1, i4 = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    i2, i3 = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
    return [pts[i1], pts[i2], pts[i3], pts[i4]], min(rect[1])


def _score(pred: np.ndarray, box: np.ndarray) -> float:
    """Mean of ``pred`` inside the polygon (``box_score_fast``)."""
    h, w = pred.shape[:2]
    box = box.copy()
    xmin = max(0, min(math.floor(box[:, 0].min()), w - 1))
    xmax = max(0, min(math.ceil(box[:, 0].max()), w - 1))
    ymin = max(0, min(math.floor(box[:, 1].min()), h - 1))
    ymax = max(0, min(math.ceil(box[:, 1].max()), h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
    box[:, 0] -= xmin
    box[:, 1] -= ymin
    cv2.fillPoly(mask, [box.astype(np.int32)], 1)
    return float(cv2.mean(pred[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def _unclip(box: np.ndarray, ratio: float) -> np.ndarray | None:
    area = cv2.contourArea(box)
    length = cv2.arcLength(box, True)
    if length == 0:
        return None
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    grown = offset.Execute(area * ratio / length)
    if not grown:
        return None
    return np.array(grown[0])


def boxes_from_map(
    pred: np.ndarray, dest_w: int, dest_h: int, params: DbParams
) -> tuple[np.ndarray, np.ndarray]:
    """``(boxes [n, 4, 2] int32 in a ``dest_w`` x ``dest_h`` frame, scores [n] float32)`` from the
    map ``pred`` ``[H, W]`` float, in the order OpenCV found the contours."""
    h, w = pred.shape
    mask = (pred > params.thresh).astype(np.uint8)
    if params.use_dilation:
        mask = np.asarray(cv2.dilate(mask, np.array([[1, 1], [1, 1]], np.uint8)), np.uint8)
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sx, sy = dest_w / w, dest_h / h
    boxes: list[np.ndarray] = []
    scores: list[float] = []
    for contour in contours[: params.max_candidates]:
        points, side = _mini_box(contour)
        if side < MIN_SIZE:
            continue
        pts = np.array(points)
        score = _score(pred, pts)
        if params.box_thresh > score:
            continue
        grown = _unclip(pts, params.unclip_ratio)
        if grown is None:
            continue
        quad, side = _mini_box(grown.reshape(-1, 1, 2))
        if side < MIN_SIZE + 2:
            continue
        out = np.array(quad)
        out[:, 0] = np.clip(np.round(out[:, 0] * sx), 0, dest_w)
        out[:, 1] = np.clip(np.round(out[:, 1] * sy), 0, dest_h)
        boxes.append(out.astype(np.int32))
        scores.append(score)
    if not boxes:
        return np.empty((0, 4, 2), np.int32), np.empty(0, np.float32)
    return np.stack(boxes), np.array(scores, np.float32)


def sort_reading_order(boxes: np.ndarray) -> np.ndarray:
    """Indices that order ``boxes`` top to bottom, left to right (``SortQuadBoxes``: by the first
    corner's ``(y, x)``, then boxes within 10 rows are swapped into x order)."""
    n = len(boxes)
    order = sorted(range(n), key=lambda i: (boxes[i][0][1], boxes[i][0][0]))
    for i in range(n - 1):
        for j in range(i, -1, -1):
            a, b = boxes[order[j]], boxes[order[j + 1]]
            if abs(b[0][1] - a[0][1]) < 10 and b[0][0] < a[0][0]:
                order[j], order[j + 1] = order[j + 1], order[j]
            else:
                break
    return np.array(order, np.int64)
