"""Fused uint8 CUDA kernels of the GPU decode, through CuPy ``RawKernel``.

Every model input is resampled from the full-resolution decoded frame (HWC uint8 on the GPU) by
one separable two-pass resampler: a horizontal pass into a float32 scratch holding only the rows
the output needs, then a vertical pass that writes uint8 CHW straight into its slot of a batch
tensor (the detector letterbox canvas, a PE square). Weights come from
``prep_geometry.filter_weights`` (a few KB per size, cached). A region may extend past the
frame: outside pixels read as 0, as ``PIL.Image.crop`` fills them.

The blur kernel computes OpenCV's grey plane (15-bit fixed-point weights) and the 3 x 3 Laplacian
with ``BORDER_REFLECT_101`` inside each rectangle and accumulates the integer sum and sum of
squares (int64 atomics): what ``crop_quality.laplacian_var`` reduces, bit for bit.
"""

from __future__ import annotations

from typing import Any

import cupy as cp
import numpy as np
from prep_geometry import filter_weights


_SRC = r"""
extern "C" __global__ void resample_h(
    const unsigned char* src, int H, int W,
    int x0, int y0, int rows,
    const int* xmin, const int* xcnt, const float* wx, int kmax, int out_w,
    float* tmp)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (ox >= out_w || r >= rows) return;
    int sy = y0 + r;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f;
    if (sy >= 0 && sy < H) {
        const unsigned char* row = src + (size_t)sy * W * 3;
        int b = xmin[ox], n = xcnt[ox];
        const float* w = wx + (size_t)ox * kmax;
        for (int k = 0; k < n; ++k) {
            int sx = x0 + b + k;
            if (sx < 0 || sx >= W) continue;
            float wk = w[k];
            const unsigned char* p = row + sx * 3;
            a0 += wk * p[0]; a1 += wk * p[1]; a2 += wk * p[2];
        }
    }
    float* t = tmp + ((size_t)r * out_w + ox) * 3;
    t[0] = a0; t[1] = a1; t[2] = a2;
}

extern "C" __global__ void resample_v(
    const float* tmp, int out_w,
    const int* ymin, const int* ycnt, const float* wy, int kmax, int out_h,
    unsigned char* dst, long long plane, int row_stride)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y;
    if (ox >= out_w || oy >= out_h) return;
    int b = ymin[oy], n = ycnt[oy];
    const float* w = wy + (size_t)oy * kmax;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f;
    for (int k = 0; k < n; ++k) {
        const float* t = tmp + ((size_t)(b + k) * out_w + ox) * 3;
        float wk = w[k];
        a0 += wk * t[0]; a1 += wk * t[1]; a2 += wk * t[2];
    }
    size_t o = (size_t)oy * row_stride + ox;
    dst[o] = (unsigned char)fminf(fmaxf(rintf(a0), 0.f), 255.f);
    dst[o + plane] = (unsigned char)fminf(fmaxf(rintf(a1), 0.f), 255.f);
    dst[o + 2 * plane] = (unsigned char)fminf(fmaxf(rintf(a2), 0.f), 255.f);
}

__device__ __forceinline__ int refl(int i, int n) {
    if (n == 1) return 0;
    if (i < 0) return -i;
    if (i >= n) return 2 * n - 2 - i;
    return i;
}

__device__ __forceinline__ int grey(const unsigned char* src, int W, int x, int y) {
    const unsigned char* p = src + ((size_t)y * W + x) * 3;
    return (p[0] * 9798 + p[1] * 19235 + p[2] * 3735 + 16384) >> 15;
}

extern "C" __global__ void lap_sums(
    const unsigned char* src, int H, int W,
    const int* rects, int nrects, long long* out)
{
    int rid = blockIdx.y;
    if (rid >= nrects) return;
    int x1 = rects[rid * 4], y1 = rects[rid * 4 + 1], x2 = rects[rid * 4 + 2], y2 = rects[rid * 4 + 3];
    int w = x2 - x1, h = y2 - y1;
    long long n = (long long)w * h;
    long long s = 0, q = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        int x = (int)(i % w), y = (int)(i / w);
        int c = grey(src, W, x1 + x, y1 + y);
        int l = grey(src, W, x1 + refl(x - 1, w), y1 + y) + grey(src, W, x1 + refl(x + 1, w), y1 + y)
              + grey(src, W, x1 + x, y1 + refl(y - 1, h)) + grey(src, W, x1 + x, y1 + refl(y + 1, h))
              - 4 * c;
        s += l; q += (long long)l * l;
    }
    __shared__ long long ss[256];
    __shared__ long long sq[256];
    ss[threadIdx.x] = s; sq[threadIdx.x] = q;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) { ss[threadIdx.x] += ss[threadIdx.x + k]; sq[threadIdx.x] += sq[threadIdx.x + k]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd((unsigned long long*)&out[rid * 2], (unsigned long long)ss[0]);
        atomicAdd((unsigned long long*)&out[rid * 2 + 1], (unsigned long long)sq[0]);
    }
}
"""


class PrepKernels:
    def __init__(self) -> None:
        mod = cp.RawModule(code=_SRC)
        self.k_h = mod.get_function('resample_h')
        self.k_v = mod.get_function('resample_v')
        self.k_lap = mod.get_function('lap_sums')
        self._cache: dict[tuple[int, int, int, int, bool], tuple[Any, ...]] = {}

    def _weights(
        self, n_in: int, n_out: int, left: int, take: int, antialias: bool
    ) -> tuple[Any, ...]:
        key = (n_in, n_out, left, take, antialias)
        hit = self._cache.get(key)
        if hit is None:
            f, c, w = filter_weights(n_in, n_out, antialias=antialias)
            f, c, w = f[left : left + take], c[left : left + take], w[left : left + take]
            hit = (
                cp.asarray(f),
                cp.asarray(c),
                cp.asarray(w),
                w.shape[1],
                int(f[0]),
                int((f + c).max()),
            )
            if len(self._cache) > 4096:
                self._cache.clear()
            self._cache[key] = hit
        return hit

    def resample_into(
        self,
        frame: Any,
        roi: tuple[int, int, int, int],
        size: tuple[int, int],
        dst: Any,
        window: tuple[int, int, int, int] | None = None,
        *,
        antialias: bool = True,
    ) -> None:
        """Resample ``frame[roi]`` (``x0, y0, w, h``) to ``size`` (``w, h``) and write its
        ``window`` (``left, top, w, h``; default all of it) into ``dst``, a CHW uint8 view."""
        h_img, w_img = int(frame.shape[0]), int(frame.shape[1])
        x0, y0, rw, rh = roi
        ow, oh = size
        left, top, ww, wh = window or (0, 0, ow, oh)
        fx, cx, wx, kx, _, _ = self._weights(rw, ow, left, ww, antialias)
        fy, cy, wy, ky, y_lo, y_hi = self._weights(rh, oh, top, wh, antialias)
        rows = max(1, y_hi - y_lo)
        tmp = cp.empty((rows, ww, 3), dtype=cp.float32)
        self.k_h(
            ((ww + 127) // 128, rows),
            (128,),
            (frame, np.int32(h_img), np.int32(w_img), np.int32(x0), np.int32(y0 + y_lo), np.int32(rows),
             fx, cx, wx, np.int32(kx), np.int32(ww), tmp),
        )  # fmt: skip
        plane = dst.strides[0] // dst.itemsize
        row_stride = dst.strides[1] // dst.itemsize
        self.k_v(
            ((ww + 127) // 128, wh),
            (128,),
            (tmp, np.int32(ww), fy - y_lo, cy, wy, np.int32(ky), np.int32(wh), dst, np.int64(plane),
             np.int32(row_stride)),
        )  # fmt: skip

    def lap_sums(self, frame: Any, rects: list[tuple[int, int, int, int]]) -> Any:
        """Device ``int64 [len(rects), 2]`` (sum, sum of squares) per clamped, non-empty
        rectangle (``x1, y1, x2, y2``); the caller downloads it once per batch."""
        out = cp.zeros((max(1, len(rects)), 2), dtype=cp.int64)
        if not rects:
            return out[:0]
        r = cp.asarray(np.asarray(rects, np.int32).reshape(-1, 4))
        h_img, w_img = int(frame.shape[0]), int(frame.shape[1])
        self.k_lap(
            (64, len(rects)),
            (256,),
            (frame, np.int32(h_img), np.int32(w_img), r, np.int32(len(rects)), out),
        )
        return out
