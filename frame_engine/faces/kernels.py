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


_SRC = r"""
__device__ __forceinline__ void tap(
    const unsigned char* src, int H, int W, float sx, float sy, float* acc)
{
    float fx = floorf(sx), fy = floorf(sy);
    int x0 = (int)fx, y0 = (int)fy;
    float ax = sx - fx, ay = sy - fy;
    float w00 = (1.f - ax) * (1.f - ay), w01 = ax * (1.f - ay);
    float w10 = (1.f - ax) * ay, w11 = ax * ay;
    int xs[2] = {x0, x0 + 1};
    int ys[2] = {y0, y0 + 1};
    float ws[4] = {w00, w01, w10, w11};
    for (int j = 0; j < 2; ++j) {
        int y = ys[j];
        if (y < 0 || y >= H) continue;
        for (int i = 0; i < 2; ++i) {
            int x = xs[i];
            if (x < 0 || x >= W) continue;
            const unsigned char* p = src + ((size_t)y * W + x) * 3;
            float w = ws[j * 2 + i];
            acc[0] += w * p[0]; acc[1] += w * p[1]; acc[2] += w * p[2];
        }
    }
}

extern "C" __global__ void warp_faces(
    const unsigned char* src, int H, int W,
    const float* inv, const int* taps, int n_faces, int S,
    unsigned char* dst, long long slot)
{
    int f = blockIdx.y;
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces || o >= S * S) return;
    int ox = o % S, oy = o / S;
    const float* m = inv + f * 6;
    int n = taps[f];
    float acc[3] = {0.f, 0.f, 0.f};
    float inv_n = 1.f / (float)n;
    for (int j = 0; j < n; ++j) {
        float y = (float)oy + ((float)j + 0.5f) * inv_n - 0.5f;
        for (int i = 0; i < n; ++i) {
            float x = (float)ox + ((float)i + 0.5f) * inv_n - 0.5f;
            tap(src, H, W, m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5], acc);
        }
    }
    float norm = inv_n * inv_n;
    unsigned char* d = dst + (size_t)f * slot;
    size_t plane = (size_t)S * S;
    size_t at = (size_t)oy * S + ox;
    d[at] = (unsigned char)fminf(fmaxf(rintf(acc[0] * norm), 0.f), 255.f);
    d[at + plane] = (unsigned char)fminf(fmaxf(rintf(acc[1] * norm), 0.f), 255.f);
    d[at + 2 * plane] = (unsigned char)fminf(fmaxf(rintf(acc[2] * norm), 0.f), 255.f);
}
"""


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
