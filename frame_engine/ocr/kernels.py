"""The text-line warp kernel (CuPy ``RawKernel``): recogniser inputs sampled from the full-resolution
frame straight into the staged batch.

One launch per batch. Each thread is one output pixel ``(u, v)`` of one line: it maps ``nx x ny``
points spread over the pixel's footprint through the line's 3 x 3 projective map
(``crop_geometry.rec_map``: perspective cut, quarter turn and resize composed), reads each with
bilinear interpolation (a coordinate outside the frame reads the nearest edge pixel, the
reference's ``BORDER_REPLICATE``), averages them and writes the rounded value as uint8 CHW in BGR
order (the frame is RGB; PP-OCR reads BGR). A pixel right of the line's resized width is the batch
padding. With one tap it is ``cv2.warpPerspective``'s ``INTER_LINEAR`` followed by ``cv2.resize``
up to OpenCV's fixed-point coordinates.
"""

from __future__ import annotations

from typing import Any

import cupy as cp
import numpy as np

from frame_engine.kernel_sources import kernel_source
from frame_engine.ocr.batching import REC_HEIGHT


#: Written right of a line's resized width. The recogniser graph turns it into 0.0039 where the
#: reference pads with exactly 0 (a 1 / 255 step); the gate scripts measured the effect.
PAD_VALUE = 128

_SRC = kernel_source('ocr_warp.cu')


class LineWarp:
    def __init__(self) -> None:
        self.k_warp = cp.RawModule(code=_SRC).get_function('warp_lines')

    def warp_into(
        self, frame: Any, maps: np.ndarray, widths: np.ndarray, taps: np.ndarray, dst: Any
    ) -> None:
        """Write one line per row of ``maps`` (``[m, 9]`` float32, ``rec_map`` row-major),
        resized to ``widths`` ``[m]`` (int32) with ``taps`` ``[m, 2]`` (int32) samples per axis,
        into ``dst``, a contiguous ``[m, 3, 48, W]`` uint8 device view (padding right of each
        line's width)."""
        m = len(maps)
        if m == 0:
            return
        batch_w = int(dst.shape[-1])
        if dst.shape != (m, 3, REC_HEIGHT, batch_w) or not dst.flags.c_contiguous:
            raise ValueError(
                f'dst must be a contiguous [{m}, 3, {REC_HEIGHT}, W] view, got {dst.shape}'
            )
        if int(np.max(widths)) > batch_w:
            raise ValueError('a line is wider than its batch')
        h_img, w_img = int(frame.shape[0]), int(frame.shape[1])
        d_maps = cp.asarray(np.ascontiguousarray(maps, np.float32))
        d_widths = cp.asarray(np.ascontiguousarray(widths, np.int32))
        d_taps = cp.asarray(np.ascontiguousarray(taps, np.int32))
        self.k_warp(
            ((batch_w * REC_HEIGHT + 127) // 128, m),
            (128,),
            (frame, np.int32(h_img), np.int32(w_img), d_maps, d_widths, d_taps, np.int32(m),
             np.int32(batch_w), np.int32(REC_HEIGHT), np.uint8(PAD_VALUE), dst),
        )  # fmt: skip
