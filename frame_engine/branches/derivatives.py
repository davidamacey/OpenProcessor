"""Stored renditions (thumbnails) of the resident, already oriented frame: ``plan.derivatives``.

The geometry is ``PIL.Image.thumbnail``'s (the long side fits ``side``, the aspect is kept, a
photo smaller than ``side`` is not enlarged), so a rendition lands on the pixels the CPU thumbnail
route serves. The pixels are one GPU resample per rendition from the full-resolution frame with
the Lanczos weights of ``prep_geometry.filter_weights`` (Pillow's ``LANCZOS``); the small result
is encoded to JPEG by nvJPEG on the GPU (``nvjpeg_encode``: only the bitstream crosses to the host,
about 0.3 ms of host CPU for a 256 plus a 512 pixel rendition against 2.3 to 2.8 ms for OpenCV),
on the engine's encode pool, started after the round-2 model calls are submitted so the encode overlaps them.
``finish`` writes ``derivative_jpegs``, one JPEG per rendition in plan order. The answer is the
``derivatives`` ``branch_status`` entry; one failed encode fails the photo, not a model call.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import cupy as cp

from frame_engine.branches.base import Branch, Photo
from frame_engine.prep_geometry import fit_size
from src.domain import frame_engine_wire as wire


if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor

    from frame_engine.nvjpeg_encode import NvJpegEncoders
    from src.domain.frame_engine_wire import RenditionSpec


class Derivatives(Branch):
    status = wire.DERIVATIVES

    def __init__(self, pool: ThreadPoolExecutor, encoders: NvJpegEncoders) -> None:
        self.pool = pool
        self.encoders = encoders
        self.jobs: list[Future[tuple[bytes, float]]] | None = None

    def round2(self, photo: Photo) -> None:
        with photo.clock.stage('derivative_render'):
            pixels = [self._render(photo, spec) for spec in photo.plan.derivatives]
            ready = self.encoders.mark()
        self.jobs = [
            self.pool.submit(self._encode, px, spec.quality, ready)
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
    def _render(photo: Photo, spec: RenditionSpec) -> Any:
        """The rendition's pixels, HWC RGB uint8, on the GPU."""
        w, h = photo.size
        ow, oh = fit_size(w, h, spec.side)
        if (ow, oh) == (w, h):
            return photo.frame
        chw = cp.empty((3, oh, ow), cp.uint8)
        photo.kernels.resample_into(photo.frame, (0, 0, w, h), (ow, oh), chw, lanczos=True)
        return cp.ascontiguousarray(chw.transpose(1, 2, 0))

    def _encode(self, rgb: Any, quality: int, ready: Any) -> tuple[bytes, float]:
        """The JPEG and the seconds the encode took on its host thread."""
        start = time.perf_counter()
        return self.encoders.encode(rgb, quality, ready), time.perf_counter() - start
