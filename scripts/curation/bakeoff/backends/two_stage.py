"""Generic two-stage pipeline: coarse detect -> crop -> fine detect -> map back.

A fine-grained model often runs on a coarse-detector's crop, where its
target is large in-frame (e.g. a plate detector on a vehicle crop). To
score it on the SAME full-frame benchmark as single-pass detectors, this
wraps the real pipeline: detect coarse regions, crop each (with padding),
run the fine-grained detector on the crop, and project its boxes back to
full-frame coordinates, then NMS. Originated for a vehicle->plate
cascade; the two detectors and their classes are entirely caller-supplied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import Detection
from .yolo_post import nms


if TYPE_CHECKING:
    from .base import Detector


class TwoStageDetector:
    """Compose a coarse-region detector with a crop-based fine detector."""

    runtime = 'two-stage'

    def __init__(
        self,
        primary: Detector,
        secondary: Detector,
        *,
        name: str = 'two-stage (coarse->crop->fine)',
        pad_frac: float = 0.1,
        primary_conf: float = 0.25,
        nms_iou: float = 0.45,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.name = name
        self.pad_frac = pad_frac
        self.primary_conf = primary_conf
        self.nms_iou = nms_iou

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        h, w = image_rgb.shape[:2]
        dets: list[Detection] = []
        for region in self.primary.detect(image_rgb):
            if region.score < self.primary_conf:
                continue
            cx1, cy1, cx2, cy2 = self._padded_crop(region, w, h)
            if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                continue
            crop = image_rgb[cy1:cy2, cx1:cx2]
            dets.extend(
                Detection(p.x1 + cx1, p.y1 + cy1, p.x2 + cx1, p.y2 + cy1, p.score)
                for p in self.secondary.detect(crop)
            )
        return self._nms(dets)

    def _padded_crop(self, region: Detection, w: int, h: int) -> tuple[int, int, int, int]:
        bw, bh = region.x2 - region.x1, region.y2 - region.y1
        px, py = bw * self.pad_frac, bh * self.pad_frac
        x1 = max(0, int(region.x1 - px))
        y1 = max(0, int(region.y1 - py))
        x2 = min(w, int(region.x2 + px))
        y2 = min(h, int(region.y2 + py))
        return x1, y1, x2, y2

    def _nms(self, dets: list[Detection]) -> list[Detection]:
        if not dets:
            return []
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=float)
        scores = np.array([d.score for d in dets], dtype=float)
        return [dets[i] for i in nms(boxes, scores, self.nms_iou)]
