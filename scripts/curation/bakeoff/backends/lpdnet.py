"""NVIDIA TAO LPDNet backend (DetectNet_v2) via onnxruntime.

LPDNet is a DetectNet_v2 grid detector, not a YOLO head: it outputs a
per-cell coverage map (``output_cov``) + per-cell bbox regressors
(``output_bbox``) on a stride-16 grid. We decode with the standard TAO
gridbox formula (cell-center +/- regressor*bbox_norm), threshold on
coverage, and NMS-cluster. The USA variant (input 640x480) matches our
US-plate domain; the CCPD variant (720x1168) is the China model.

Preprocessing follows the LPDNet nvinfer config: plain resize to the
model's WxH, RGB, scale 1/255 (net-scale-factor), NCHW.

Model files (unpacked, on nvm):
    /mnt/nvm/datasets/models/lpdnet_pruned_v2.2.1/lpdnet_pruned_v2.2.1/
        LPDNet_usa_pruned_tao5.onnx     (input 3x480x640)
        LPDNet_CCPD_pruned_tao5.onnx    (input 3x1168x720)
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import Detection
from .yolo_post import nms


# Per-variant model input (width, height). Grid stride is 16.
_VARIANTS = {
    'usa': (640, 480),
    'ccpd': (720, 1168),
}
_STRIDE = 16
_BBOX_NORM = 35.0  # TAO DetectNet_v2 gridbox default


class LpdnetDetector:
    """Run NVIDIA LPDNet (DetectNet_v2 ONNX) as a bake-off Detector."""

    runtime = 'onnxruntime'

    def __init__(
        self,
        weights: str,
        *,
        name: str | None = None,
        variant: str = 'usa',
        conf: float = 0.001,
        iou: float = 0.5,
        device: str = 'cuda',
    ) -> None:
        import onnxruntime as ort

        if variant not in _VARIANTS:
            raise ValueError(f'lpdnet variant must be one of {sorted(_VARIANTS)}')
        self.name = name or f'lpdnet-{variant}'
        self.variant = variant
        self.in_w, self.in_h = _VARIANTS[variant]
        self.conf = conf
        self.iou = iou
        providers = (
            ['CPUExecutionProvider']
            if device == 'cpu'
            else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self._sess = ort.InferenceSession(weights, providers=providers)
        self._in = self._sess.get_inputs()[0].name
        # outputs: cov (n,1,gh,gw), bbox (n,4,gh,gw) — order by name to be safe.
        outs = {o.name: o.name for o in self._sess.get_outputs()}
        self._cov = next(n for n in outs if 'cov' in n.lower())
        self._bbox = next(n for n in outs if 'bbox' in n.lower())

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        h0, w0 = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        tensor = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        tensor = np.ascontiguousarray(tensor)
        cov, bbox = self._sess.run([self._cov, self._bbox], {self._in: tensor})
        boxes, scores = self._decode(cov[0, 0], bbox[0], w0, h0)
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

    def _decode(
        self, cov: np.ndarray, bbox: np.ndarray, w0: int, h0: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """DetectNet_v2 gridbox decode -> abs xyxy in the ORIGINAL frame."""
        _gh, _gw = cov.shape
        ys, xs = np.where(cov >= self.conf)
        if xs.size == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        # grid-cell centers in model-input pixels
        gcx = xs * _STRIDE + _STRIDE / 2.0
        gcy = ys * _STRIDE + _STRIDE / 2.0
        b0, b1, b2, b3 = bbox[0, ys, xs], bbox[1, ys, xs], bbox[2, ys, xs], bbox[3, ys, xs]
        x1 = gcx - b0 * _BBOX_NORM
        y1 = gcy - b1 * _BBOX_NORM
        x2 = gcx + b2 * _BBOX_NORM
        y2 = gcy + b3 * _BBOX_NORM
        # scale model-input coords back to the original frame (plain resize)
        sx, sy = w0 / self.in_w, h0 / self.in_h
        out = np.stack([x1 * sx, y1 * sy, x2 * sx, y2 * sy], axis=1).astype(np.float32)
        return out, cov[ys, xs].astype(np.float32)
