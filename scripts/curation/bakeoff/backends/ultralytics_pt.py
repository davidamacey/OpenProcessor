"""Ultralytics backend — runs any ``.pt``/``.onnx`` Ultralytics model.

Covers the current ``best.pt``, the new full-frame YOLO26 contender, and
public Ultralytics-format detectors (DeepPlate YOLOv11, YOLOv8 ALPR). For
multi-class public models, pass ``plate_class_id`` to keep only the plate
class; single-class plate models keep everything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Detection


if TYPE_CHECKING:
    import numpy as np


class UltralyticsDetector:
    """Wrap an Ultralytics model as a bake-off :class:`Detector`."""

    runtime = 'ultralytics'

    def __init__(
        self,
        weights: str,
        *,
        name: str | None = None,
        imgsz: int = 1280,
        device: str | int = 0,
        conf: float = 0.001,
        iou: float = 0.7,
        plate_class_id: int | None = None,
        keep_classes: set[int] | None = None,
        half: bool = True,
    ) -> None:
        from ultralytics import YOLO

        self.name = name or weights
        self.imgsz = imgsz
        self.device = device
        self.conf = conf
        self.iou = iou
        self.plate_class_id = plate_class_id
        # When set, keep only these class ids (e.g. COCO vehicle classes
        # {2,3,5,7} for the crop-mode vehicle stage). Takes precedence over
        # plate_class_id.
        self.keep_classes = keep_classes
        self.half = half
        self._model = YOLO(weights)

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        # Ultralytics accepts an RGB ndarray directly and returns boxes in
        # the ORIGINAL frame's pixel coords (it undoes its own letterbox).
        res = self._model.predict(
            image_rgb,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            half=self.half,
            verbose=False,
        )
        if not res:
            return []
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        out: list[Detection] = []
        for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, classes, strict=True):
            if self.keep_classes is not None:
                if cls not in self.keep_classes:
                    continue
            elif self.plate_class_id is not None and cls != self.plate_class_id:
                continue
            out.append(Detection(float(x1), float(y1), float(x2), float(y2), float(score)))
        return out
