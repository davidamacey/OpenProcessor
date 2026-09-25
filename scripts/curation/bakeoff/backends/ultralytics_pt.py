"""Ultralytics backend — runs any ``.pt``/``.onnx`` Ultralytics model.

Returns every class with its model class id; the harness maps model
classes to the eval dataset's classes. ``keep_classes`` restricts output to
a set of model class ids (the crop-mode coarse stage's context classes).
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
        keep_classes: set[int] | None = None,
        half: bool = True,
    ) -> None:
        from ultralytics import YOLO

        self.name = name or weights
        self.imgsz = imgsz
        self.device = device
        self.conf = conf
        self.iou = iou
        # When set, keep only these class ids (a profile's context_class_ids
        # for a crop-mode coarse stage).
        self.keep_classes = keep_classes
        self.half = half
        self._model = YOLO(weights)
        names = getattr(self._model, 'names', None)
        self.class_names: dict[int, str] | None = (
            {int(k): str(v) for k, v in dict(names).items()} if names else None
        )

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
            if self.keep_classes is not None and cls not in self.keep_classes:
                continue
            out.append(
                Detection(float(x1), float(y1), float(x2), float(y2), float(score), int(cls))
            )
        return out
