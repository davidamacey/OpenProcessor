"""open-image-models backend (ankandrew) — YOLOv9-tiny license-plate detector.

A license-plate-only public model, kept as a baseline backend for the
``license_plate`` example profile.

Runs the model through its own package API (it returns boxes directly), so
we compare it exactly as a user of that library would. Install:
``.venv/bin/pip install open-image-models``. The package ships pretrained
ONNX weights trained on open plate datasets — record that provenance in
any write-up (it's the "their training data vs ours" comparison).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Detection


if TYPE_CHECKING:
    import numpy as np


class OpenImageModelsDetector:
    """Wrap ankandrew/open-image-models LicensePlateDetector."""

    runtime = 'open-image-models'
    class_names: dict[int, str] | None = None

    def __init__(
        self,
        *,
        model_name: str = 'yolo-v9-t-640-license-plate-end2end',
        name: str | None = None,
        conf: float = 0.001,
        device: str = 'cuda',
    ) -> None:
        from open_image_models import LicensePlateDetector

        self.name = name or model_name
        self.conf = conf
        # The library selects compute via onnxruntime providers (no `device`
        # arg). CUDA needs onnxruntime-gpu; otherwise it falls back to CPU.
        providers = (
            ['CPUExecutionProvider']
            if device == 'cpu'
            else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self._det = LicensePlateDetector(
            detection_model=model_name, conf_thresh=conf, providers=providers
        )

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        results = self._det.predict(image_rgb)
        out: list[Detection] = []
        for r in results:
            bb = r.bounding_box
            score = float(getattr(r, 'confidence', 1.0))
            if score < self.conf:
                continue
            out.append(Detection(float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2), score))
        return out
