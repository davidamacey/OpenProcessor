"""CoreML backend — scores the Apple ``.mlpackage`` deliverables (macOS only).

Runs the FP16 / INT8 CoreML packages produced on the Mac Studio so they get the
SAME COCO metrics + latency as every other backend, on the same frozen test set.

Two CoreML-specific differences from the ONNX/Triton path:
  * Ultralytics' CoreML export takes an IMAGE input with the ``1/255`` scale baked
    in, so we feed a letterboxed PIL image directly and must NOT divide by 255
    ourselves (no ``to_input_tensor``) -- doing both would double-normalize.
  * We export with ``nms=False`` and decode externally with the shared
    ``yolo_post`` decode, sidestepping the CoreML pipeline's FP32 class-index
    handling and keeping the metric identical to the other backends.

CoreML inference runs only on macOS (Apple Neural Engine / Metal), so this module
imports ``coremltools`` lazily and is only constructed on the Mac.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Detection
from .yolo_post import decode_yolo26_e2e, decode_yolo_v11, letterbox, nms


class CoreMLDetector:
    """Run a single-class YOLO plate ``.mlpackage`` (no embedded NMS) via CoreML."""

    runtime = 'coreml'

    def __init__(
        self,
        weights: str,
        *,
        name: str | None = None,
        imgsz: int = 640,
        conf: float = 0.001,
        iou: float = 0.45,
        compute_units: str = 'ALL',
        coords_normalized: bool = False,
    ) -> None:
        import coremltools as ct

        self._ct = ct
        units = ct.ComputeUnit[compute_units]
        self._model = ct.models.MLModel(weights, compute_units=units)
        spec = self._model.get_spec()
        self._input_name = spec.description.input[0].name
        self._output_name = spec.description.output[0].name

        self.name = name or Path(weights).stem
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.coords_normalized = coords_normalized

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        from PIL import Image

        lb, scale, pad = letterbox(image_rgb, self.imgsz)
        # Image input: feed the uint8 RGB letterboxed frame; the model applies
        # its own /255 scale, so we deliberately do NOT normalize here.
        pil = Image.fromarray(lb)
        out = self._model.predict({self._input_name: pil})
        output = np.asarray(out[self._output_name], dtype=np.float32)
        # YOLO26 NMS-free head [1, max_det, 6]; older YOLO raw [1, 5, N] grid.
        if output.ndim == 3 and output.shape[-1] == 6:
            boxes, scores = decode_yolo26_e2e(output, scale=scale, pad=pad, conf_thresh=self.conf)
            keep = list(range(len(scores)))
        else:
            boxes, scores = decode_yolo_v11(
                output,
                scale=scale,
                pad=pad,
                input_size=self.imgsz,
                conf_thresh=self.conf,
                coords_normalized=self.coords_normalized,
            )
            keep = nms(boxes, scores, self.iou)
        return [
            Detection(
                float(boxes[i, 0]),
                float(boxes[i, 1]),
                float(boxes[i, 2]),
                float(boxes[i, 3]),
                float(scores[i]),
            )
            for i in keep
        ]
