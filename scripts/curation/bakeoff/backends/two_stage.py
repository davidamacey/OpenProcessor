"""Two-stage in-house pipeline: vehicle detect -> crop -> LPR -> map back.

The deployed LPR model runs on vehicle crops at 640, where plates are
large. To score it on the SAME full-frame benchmark as the single-pass
public detectors, we wrap the real pipeline: detect vehicles, crop each
(with padding), run the crop-based LPR detector, and project plate boxes
back to full-frame coordinates, then NMS. Documented in the paper as a
pipeline (two models) vs the single-pass contenders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import Detection
from .yolo_post import nms


if TYPE_CHECKING:
    from .base import Detector


class TwoStageDetector:
    """Compose a vehicle-region detector with a crop-based LPR detector."""

    runtime = 'two-stage'

    def __init__(
        self,
        vehicle: Detector,
        lpr: Detector,
        *,
        name: str = 'in-house 2-stage (vehicle->crop->LPR)',
        pad_frac: float = 0.1,
        vehicle_conf: float = 0.25,
        nms_iou: float = 0.45,
    ) -> None:
        self.vehicle = vehicle
        self.lpr = lpr
        self.name = name
        self.pad_frac = pad_frac
        self.vehicle_conf = vehicle_conf
        self.nms_iou = nms_iou

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        h, w = image_rgb.shape[:2]
        plate_dets: list[Detection] = []
        for veh in self.vehicle.detect(image_rgb):
            if veh.score < self.vehicle_conf:
                continue
            cx1, cy1, cx2, cy2 = self._padded_crop(veh, w, h)
            if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                continue
            crop = image_rgb[cy1:cy2, cx1:cx2]
            plate_dets.extend(
                Detection(p.x1 + cx1, p.y1 + cy1, p.x2 + cx1, p.y2 + cy1, p.score)
                for p in self.lpr.detect(crop)
            )
        return self._nms(plate_dets)

    def _padded_crop(self, veh: Detection, w: int, h: int) -> tuple[int, int, int, int]:
        bw, bh = veh.x2 - veh.x1, veh.y2 - veh.y1
        px, py = bw * self.pad_frac, bh * self.pad_frac
        x1 = max(0, int(veh.x1 - px))
        y1 = max(0, int(veh.y1 - py))
        x2 = min(w, int(veh.x2 + px))
        y2 = min(h, int(veh.y2 + py))
        return x1, y1, x2, y2

    def _nms(self, dets: list[Detection]) -> list[Detection]:
        if not dets:
            return []
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=float)
        scores = np.array([d.score for d in dets], dtype=float)
        return [dets[i] for i in nms(boxes, scores, self.nms_iou)]
