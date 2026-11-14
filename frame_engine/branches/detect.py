"""``detect``: the detector letterbox and call (round 1), its boxes (settle), the blur sums (finish).

The letterbox is uint8 with Pillow ``BILINEAR`` (antialiased) resampling, the model's fill around
it. The detector is a fused end-to-end model: one tensor ``[batch, 300, 6]``, rows ``x1, y1, x2, y2,
score, class`` in letterboxed input pixels, zero-padded past the last detection. Each detection's
rectangle (``crop_rect`` of the box mapped back to the frame) is what the crop branches cut. The
blur sums are the frame's and every detection's clamped rectangle (``blur_rect``): pixels, sum and
sum of squares of OpenCV's Laplacian on the grey plane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from frame_engine.branches.base import Branch, Call, Photo
from frame_engine.prep_geometry import blur_rect
from frame_engine.transport import ModelInput
from src.domain import frame_engine_wire as wire
from src.domain.image_geometry import crop_rect, letterbox_geometry, undo_letterbox


if TYPE_CHECKING:
    from src.domain.image_geometry import Letterbox


DET_INPUT = 'images'
DET_OUTPUT = 'output0'


class Detect(Branch):
    status = 'detect'

    def __init__(self, spec: wire.DetectorSpec) -> None:
        self.spec = spec
        self.call: Call | None = None
        self.letterbox: Letterbox | None = None
        self.boxes: np.ndarray | None = None  # letterboxed input pixels, one row per detection

    def round1(self, photo: Photo) -> None:
        w, h = photo.size
        net = self.spec.input_size
        lb = self.letterbox = letterbox_geometry(w, h, net)
        tensor = ModelInput(DET_INPUT, 'UINT8', (1, 3, net, net))
        staged = photo.calls.stage(tensor.nbytes, 'letterbox')
        with photo.clock.stage('letterbox'):
            canvas = staged.device.reshape(tensor.shape)
            canvas.fill(self.spec.letterbox_fill)
            dst = canvas[0, :, lb.top : lb.top + lb.new_h, lb.left : lb.left + lb.new_w]
            photo.kernels.resample_into(photo.frame, (0, 0, w, h), (lb.new_w, lb.new_h), dst)
        self.call = photo.calls.submit(
            self, 'detect', staged, self.spec.model, tensor, (DET_OUTPUT,)
        )

    def settle(self, photo: Photo) -> None:
        if self.call is None or self.letterbox is None:
            raise RuntimeError('detect settled before round 1')
        fused = photo.calls.result(self.call)[DET_OUTPUT]
        lb = self.letterbox
        w, h = photo.size
        with photo.clock.stage('detect'):
            rows = fused[0][fused[0][:, 4] > 0.0]
            boxes = self.boxes = rows[:, :4].astype(np.float32)
            photo.classes = rows[:, 5].astype(np.int32)
            photo.rects = [
                crop_rect(undo_letterbox(tuple(b), lb.scale, (lb.pad_w, lb.pad_h)), w, h)
                for b in boxes
            ]
            photo.out[wire.LETTERBOX] = np.array([lb.scale, lb.pad_w, lb.pad_h], np.float64)
            photo.out[wire.DET_BOXES] = boxes
            photo.out[wire.DET_SCORES] = rows[:, 4].astype(np.float32)
            photo.out[wire.DET_CLASSES] = photo.classes

    def finish(self, photo: Photo) -> None:
        with photo.clock.stage('blur'):
            photo.out[wire.BLUR_SUMS] = self._blur_sums(photo)

    def _blur_sums(self, photo: Photo) -> np.ndarray:
        """``[1 + detections, 3]`` (pixels, sum, sum of squares): the frame, then each box."""
        if self.boxes is None or self.letterbox is None:
            raise RuntimeError('blur sums before the detector answered')
        w, h = photo.size
        lb = self.letterbox
        clamped = [
            blur_rect(undo_letterbox(tuple(b), lb.scale, (lb.pad_w, lb.pad_h)), w, h)
            for b in self.boxes
        ]
        rects = [(0, 0, w, h)] + [r for r in clamped if r is not None]
        sums = iter(cp.asnumpy(photo.kernels.lap_sums(photo.frame, rects)))
        rows = [(w * h, *next(sums))]
        rows.extend(
            (0, 0, 0) if r is None else ((r[2] - r[0]) * (r[3] - r[1]), *next(sums))
            for r in clamped
        )
        return np.array(rows, np.int64).reshape(-1, 3)
