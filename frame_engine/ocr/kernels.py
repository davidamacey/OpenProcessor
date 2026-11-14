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

from frame_engine.ocr.batching import REC_HEIGHT


#: Written right of a line's resized width. The recogniser graph turns it into 0.0039 where the
#: reference pads with exactly 0 (a 1 / 255 step); the gate scripts measured the effect.
PAD_VALUE = 128

_SRC = r"""
__device__ __forceinline__ void tap(
    const unsigned char* src, int H, int W, float sx, float sy, float* acc)
{
    float fx = floorf(sx), fy = floorf(sy);
    float ax = sx - fx, ay = sy - fy;
    int x0 = min(max((int)fx, 0), W - 1), x1 = min(max((int)fx + 1, 0), W - 1);
    int y0 = min(max((int)fy, 0), H - 1), y1 = min(max((int)fy + 1, 0), H - 1);
    const unsigned char* p00 = src + ((size_t)y0 * W + x0) * 3;
    const unsigned char* p01 = src + ((size_t)y0 * W + x1) * 3;
    const unsigned char* p10 = src + ((size_t)y1 * W + x0) * 3;
    const unsigned char* p11 = src + ((size_t)y1 * W + x1) * 3;
    float w00 = (1.f - ax) * (1.f - ay), w01 = ax * (1.f - ay);
    float w10 = (1.f - ax) * ay, w11 = ax * ay;
    for (int c = 0; c < 3; ++c)
        acc[c] += w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c];
}

extern "C" __global__ void warp_lines(
    const unsigned char* src, int H, int W,
    const float* maps, const int* widths, const int* taps, int n_lines, int batch_w, int height,
    unsigned char pad, unsigned char* dst)
{
    int line = blockIdx.y;
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= n_lines || o >= batch_w * height) return;
    int u = o % batch_w, v = o / batch_w;
    unsigned char* d = dst + (size_t)line * 3 * height * batch_w;
    size_t at = (size_t)v * batch_w + u, plane = (size_t)height * batch_w;
    if (u >= widths[line]) {
        d[at] = pad; d[at + plane] = pad; d[at + 2 * plane] = pad;
        return;
    }
    const float* m = maps + line * 9;
    int nx = taps[line * 2], ny = taps[line * 2 + 1];
    float acc[3] = {0.f, 0.f, 0.f};
    for (int j = 0; j < ny; ++j) {
        float y = (float)v + ((float)j + 0.5f) / (float)ny - 0.5f;
        for (int i = 0; i < nx; ++i) {
            float x = (float)u + ((float)i + 0.5f) / (float)nx - 0.5f;
            float den = m[6] * x + m[7] * y + m[8];
            tap(src, H, W, (m[0] * x + m[1] * y + m[2]) / den, (m[3] * x + m[4] * y + m[5]) / den, acc);
        }
    }
    float norm = 1.f / (float)(nx * ny);
    // RGB frame to the BGR planes PP-OCR reads
    d[at] = (unsigned char)fminf(fmaxf(rintf(acc[2] * norm), 0.f), 255.f);
    d[at + plane] = (unsigned char)fminf(fmaxf(rintf(acc[1] * norm), 0.f), 255.f);
    d[at + 2 * plane] = (unsigned char)fminf(fmaxf(rintf(acc[0] * norm), 0.f), 255.f);
}
"""


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
