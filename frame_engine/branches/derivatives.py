"""Stored renditions (thumbnails) of the resident, already oriented frame: ``plan.derivatives``.

The frame is already in sRGB (``frame_engine/colour.py`` converts a photo tagged with a wide-gamut
profile before any branch reads it), so every rendition is an sRGB picture and is written as a
JPEG without a profile, which every viewer reads as sRGB.

The geometry is ``PIL.Image.thumbnail``'s (the long side fits ``side``, the aspect is kept, a
photo smaller than ``side`` is not enlarged), so a rendition lands on the pixels the CPU thumbnail
route serves. The pixels are one GPU resample per rendition from the full-resolution frame with
the Lanczos weights of ``prep_geometry.filter_weights`` (Pillow's ``LANCZOS``); the small result
goes to the host and is JPEG-encoded on the engine's encode pool (OpenCV, as the crop JPEGs are),
started after the round-2 model calls are submitted so the copy and the encode overlap them.
``finish`` writes ``derivative_jpegs``, one JPEG per rendition in plan order. The answer is the
``derivatives`` ``branch_status`` entry; one failed encode fails the photo, not a model call.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cupy as cp
import cv2
import numpy as np

from frame_engine.branches.base import Branch, Photo
from frame_engine.prep_geometry import fit_size
from src.domain import frame_engine_wire as wire


if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor

    from src.domain.frame_engine_wire import RenditionSpec


class Derivatives(Branch):
    status = wire.DERIVATIVES

    def __init__(self, pool: ThreadPoolExecutor) -> None:
        self.pool = pool
        self.jobs: list[Future[tuple[bytes, float]]] | None = None

    def round2(self, photo: Photo) -> None:
        with photo.clock.stage('derivative_render'):
            pixels = [self._render(photo, spec) for spec in photo.plan.derivatives]
        self.jobs = [
            self.pool.submit(_encode, px, spec.quality)
            for px, spec in zip(pixels, photo.plan.derivatives, strict=True)
        ]

    def finish(self, photo: Photo) -> None:
        done = [f.result() for f in self.jobs or ()]
        photo.out[wire.DERIVATIVE_JPEGS] = [data for data, _ in done]
        photo.clock.add('derivative_jpeg', sum(secs for _, secs in done))

    def close(self, _photo: Photo) -> None:
        for f in self.jobs or ():
            f.cancel()  # not started yet; a running encode ends on its own (host pixels only)

    @staticmethod
    def _render(photo: Photo, spec: RenditionSpec) -> np.ndarray:
        """The rendition's pixels, HWC BGR uint8 on the host (what ``cv2.imencode`` takes)."""
        w, h = photo.size
        ow, oh = fit_size(w, h, spec.side)
        if (ow, oh) == (w, h):
            return np.asarray(cp.ascontiguousarray(photo.frame[:, :, ::-1]).get())
        chw = cp.empty((3, oh, ow), cp.uint8)
        photo.kernels.resample_into(photo.frame, (0, 0, w, h), (ow, oh), chw, lanczos=True)
        return np.asarray(cp.ascontiguousarray(chw[::-1].transpose(1, 2, 0)).get())


def _encode(bgr: np.ndarray, quality: int) -> tuple[bytes, float]:
    """The JPEG and the seconds the encode took on its host thread."""
    start = time.perf_counter()
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError('rendition JPEG encode failed')
    return buf.tobytes(), time.perf_counter() - start
