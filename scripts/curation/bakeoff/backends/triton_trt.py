"""Triton gRPC backend — scores the actually-deployed TRT engine.

This measures the production path (lpr_nanov11_640 TensorRT FP16 served by
Triton), so its accuracy vs the same weights run through Ultralytics is a
real "deployment parity" datapoint, and its latency reflects real serving.
Decode follows the the training export's coordinate convention (normalized [0,1] coords);
flip ``coords_normalized`` if a parity check vs Ultralytics disagrees.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Detection
from .yolo_post import decode_yolo_v11, letterbox, nms, to_input_tensor


if TYPE_CHECKING:
    import numpy as np


class TritonLprDetector:
    """Run a YOLOv11-format plate model served by Triton over gRPC."""

    runtime = 'triton-trt'

    def __init__(
        self,
        *,
        url: str = 'localhost:4601',
        model: str = 'lpr_nanov11_640',
        name: str | None = None,
        input_name: str = 'images',
        output_name: str = 'output0',
        input_size: int = 640,
        conf: float = 0.001,
        iou: float = 0.45,
        coords_normalized: bool = True,
    ) -> None:
        import tritonclient.grpc as grpcclient

        self._grpc = grpcclient
        self._client = grpcclient.InferenceServerClient(url=url)
        self.model = model
        self.name = name or model
        self.input_name = input_name
        self.output_name = output_name
        self.input_size = input_size
        self.conf = conf
        self.iou = iou
        self.coords_normalized = coords_normalized

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        lb, scale, pad = letterbox(image_rgb, self.input_size)
        tensor = to_input_tensor(lb)
        infer_input = self._grpc.InferInput(self.input_name, list(tensor.shape), 'FP32')
        infer_input.set_data_from_numpy(tensor)
        requested = self._grpc.InferRequestedOutput(self.output_name)
        resp = self._client.infer(self.model, [infer_input], outputs=[requested])
        output = resp.as_numpy(self.output_name)
        boxes, scores = decode_yolo_v11(
            output,
            scale=scale,
            pad=pad,
            input_size=self.input_size,
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
