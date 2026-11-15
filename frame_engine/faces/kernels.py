"""The face warp kernel (CuPy ``RawKernel``): aligned 112 x 112 crops sampled from the full-resolution
frame straight into the staged recogniser batch.

One launch per batch. Each thread is one output pixel of one face: it maps ``samples x samples``
points spread over the pixel's footprint through the face's inverse similarity
(``alignment.warp_params``), reads each with bilinear interpolation (pixels outside the frame are
0, ``cv2.warpAffine``'s ``BORDER_CONSTANT``), averages them and writes the rounded value as uint8
CHW, the frame's RGB order. With one tap it is ``cv2.warpAffine``'s ``INTER_LINEAR`` up to
OpenCV's 1/32-pixel fixed-point coordinates.
"""

from __future__ import annotations

from typing import Any

import cupy as cp
import numpy as np

from frame_engine.kernel_sources import kernel_source


_SRC = kernel_source('faces_warp.cu')


class WarpKernel:
    def __init__(self) -> None:
        self.k_warp = cp.RawModule(code=_SRC).get_function('warp_faces')

    def warp_into(self, frame: Any, inv: np.ndarray, taps: np.ndarray, dst: Any) -> None:
        """Write one aligned crop per row of ``inv`` (``[m, 6]`` float32 output-to-source maps,
        ``taps`` ``[m]`` int32) into ``dst``, a contiguous ``[m, 3, S, S]`` uint8 device view.

        The maps are uploaded per call on the caller's current stream and read only by this
        launch on it; nothing on the device is cached across workers (CuPy's pool reuses a freed
        block only on the stream it was allocated on)."""
        m = len(inv)
        if m == 0:
            return
        size = int(dst.shape[-1])
        if dst.shape != (m, 3, size, size) or not dst.flags.c_contiguous:
            raise ValueError(f'dst must be a contiguous [{m}, 3, S, S] view, got {dst.shape}')
        h_img, w_img = int(frame.shape[0]), int(frame.shape[1])
        d_inv = cp.asarray(np.ascontiguousarray(inv, np.float32))
        d_taps = cp.asarray(np.ascontiguousarray(taps, np.int32))
        self.k_warp(
            ((size * size + 127) // 128, m),
            (128,),
            (frame, np.int32(h_img), np.int32(w_img), d_inv, d_taps, np.int32(m), np.int32(size),
             dst, np.int64(3 * size * size)),
        )  # fmt: skip
