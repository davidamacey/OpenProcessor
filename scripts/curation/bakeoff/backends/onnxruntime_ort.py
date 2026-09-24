"""ONNX Runtime backend — scores the PORTABLE ONNX artifacts we ship.

This measures the deliverables deployed NVIDIA users actually run: the FP16 ONNX
(primary, on CUDAExecutionProvider, no engine build) and the INT8 QDQ ONNX
(edge/CPU, or GPU via the TensorRT EP). Decode is the shared ``yolo_post`` path,
identical to the Triton backend, so accuracy is directly comparable.

Coordinate space: our Ultralytics-exported ONNX emits PIXEL coords in the
letterboxed input space, so ``coords_normalized`` defaults to False here (the
Triton plan uses the custom training export with normalized coords -> True). Flip
it if the FP32-ONNX-vs-.pt parity check disagrees.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .base import Detection
from .yolo_post import decode_yolo26_e2e, decode_yolo_v11, letterbox, nms, to_input_tensor


if TYPE_CHECKING:
    import numpy as np


# Display tag per active execution provider, so latency is read in context.
_EP_RUNTIME = {
    'TensorrtExecutionProvider': 'ort-trt',
    'CUDAExecutionProvider': 'ort-cuda',
    'CoreMLExecutionProvider': 'ort-coreml',  # Apple Neural Engine via ONNX Runtime
    'CPUExecutionProvider': 'ort-cpu',
}


class OnnxRuntimeDetector:
    """Run a single-class YOLO ONNX (no embedded NMS) via ONNX Runtime."""

    def __init__(
        self,
        weights: str,
        *,
        name: str | None = None,
        imgsz: int = 640,
        conf: float = 0.001,
        iou: float = 0.45,
        providers: str | None = None,
        coords_normalized: bool = False,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> None:
        import numpy as np
        import onnxruntime as ort

        self._np = np
        requested = [p.strip() for p in (providers or '').split(',') if p.strip()]
        available = ort.get_available_providers()
        chosen = [p for p in requested if p in available] or [
            p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in available
        ]
        self._sess = ort.InferenceSession(weights, providers=chosen)
        active = self._sess.get_providers()[0]

        self.name = name or Path(weights).stem
        self.runtime = _EP_RUNTIME.get(active, active)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.coords_normalized = coords_normalized

        sess_in = self._sess.get_inputs()[0]
        self.input_name = input_name or sess_in.name
        self.output_name = output_name or self._sess.get_outputs()[0].name
        # FP16-weight exports take an FP16 input tensor; cast to match.
        self._in_dtype = np.float16 if 'float16' in sess_in.type else np.float32

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        lb, scale, pad = letterbox(image_rgb, self.imgsz)
        tensor = to_input_tensor(lb).astype(self._in_dtype, copy=False)
        out = self._np.asarray(
            self._sess.run([self.output_name], {self.input_name: tensor})[0],
            dtype=self._np.float32,
        )
        # YOLO26 exports an NMS-free head [1, max_det, 6] = (x1,y1,x2,y2,score,cls);
        # older YOLO (v8/v11) export the raw [1, 5, N] grid needing decode + NMS.
        if out.ndim == 3 and out.shape[-1] == 6:
            boxes, scores = decode_yolo26_e2e(out, scale=scale, pad=pad, conf_thresh=self.conf)
            keep = list(range(len(scores)))  # already NMS-free
        else:
            boxes, scores = decode_yolo_v11(
                out,
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
