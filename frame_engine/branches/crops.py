"""``crops: inline``: each rectangle's pixels as a JPEG (round 2 starts them, ``finish`` waits).

The rectangle's pixels are copied to the host (BGR) and encoded on host threads with OpenCV at
``OP_FRAME_ENGINE_JPEG_QUALITY``, the long side first capped at ``OP_FRAME_ENGINE_CROP_MAX_SIDE``
with the area filter (never upscaled). Started after the round-2 model calls are submitted, so
the copies and the encodes overlap the calls in flight. Not a ``branch_status`` entry: a photo
with no rectangles (the detector failed) answers no ``crop_jpegs``; one with no detections an
empty list.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cupy as cp
import cv2

from frame_engine.branches.base import Branch, Photo
from src.domain import frame_engine_wire as wire


if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor

    import numpy as np


class CropJpegs(Branch):
    def __init__(self, pool: ThreadPoolExecutor, quality: int, max_side: int) -> None:
        self.pool = pool
        self.quality = quality
        self.max_side = max_side
        self.jobs: list[Future[tuple[bytes, float]]] | None = None

    def round2(self, photo: Photo) -> None:
        if photo.rects is None:
            return
        with photo.clock.stage('crop_copy'):
            pixels = [
                cp.ascontiguousarray(photo.frame[y1:y2, x1:x2, ::-1]).get()
                for x1, y1, x2, y2 in photo.rects
            ]
        self.jobs = [self.pool.submit(self._encode, px) for px in pixels]

    def finish(self, photo: Photo) -> None:
        if self.jobs is None:
            return
        done = [f.result() for f in self.jobs]
        photo.out[wire.CROP_JPEGS] = [data for data, _ in done]  # empty when nothing was found
        if done:
            photo.clock.add('crop_jpeg', sum(secs for _, secs in done))

    def close(self, _photo: Photo) -> None:
        for f in self.jobs or ():
            f.cancel()  # not started yet; a running encode ends on its own (host pixels only)

    def _encode(self, pixels: np.ndarray) -> tuple[bytes, float]:
        """The JPEG and the seconds the encode took on its host thread."""
        start = time.perf_counter()
        side = max(pixels.shape[:2])
        if 0 < self.max_side < side:  # cap before encode: area filter, never upscaled
            scale = self.max_side / side
            h, w = pixels.shape[:2]
            pixels = cv2.resize(
                pixels,
                (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise RuntimeError('crop JPEG encode failed')
        return buf.tobytes(), time.perf_counter() - start
