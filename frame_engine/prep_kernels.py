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

import functools
import threading
from typing import Any, NamedTuple

import cupy as cp
import numpy as np

from frame_engine.prep_geometry import filter_weights


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
    const int* ymin, const int* ycnt, const float* wy, int kmax, int out_h, int ybase,
    unsigned char* dst, long long plane, int row_stride)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y;
    if (ox >= out_w || oy >= out_h) return;
    int b = ymin[oy] - ybase, n = ycnt[oy];
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


#: Host-side weight sets remembered (``functools.lru_cache``: its documentation promises thread
#: safety). A set is a few KB (a crop square) to about 200 KB (a Lanczos full-frame rendition).
WEIGHT_SETS = 2048
#: Bytes of one worker's staging area for the weights of the resamples it has queued.
ARENA_BYTES = 2 << 20
_ALIGN = 16


class _Weights(NamedTuple):
    """One axis's filter, packed as ``first int32[n] | count int32[n] | weights float32[n * kmax]``
    (the layout the kernels read), plus what the caller needs to size its scratch."""

    blob: np.ndarray
    n: int
    kmax: int
    #: First and one past the last source index any output of this slice reads.
    lo: int
    hi: int


@functools.lru_cache(maxsize=WEIGHT_SETS)
def _weights(
    n_in: int, n_out: int, left: int, take: int, antialias: bool, lanczos: bool
) -> _Weights:
    f, c, w = filter_weights(n_in, n_out, antialias=antialias, lanczos=lanczos)
    f, c, w = f[left : left + take], c[left : left + take], w[left : left + take]
    blob = np.concatenate([f.view(np.uint8).ravel(), c.view(np.uint8).ravel(),
                           np.ascontiguousarray(w).view(np.uint8).ravel()])  # fmt: skip
    return _Weights(blob, int(f.shape[0]), int(w.shape[1]), int(f[0]), int((f + c).max()))


class _Arena:
    """One worker's staging for weights: a pinned host buffer and its device twin, filled front to
    back. A set is written to the host buffer, copied to the device on the worker's stream
    (``cudaMemcpyAsync``, https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.runtime.memcpyAsync.html,
    which CuPy calls without the interpreter lock) and read there by the kernels queued behind the
    copy on the same stream. Neither buffer is written again until the stream was waited for: when
    the buffer is full the worker waits for its stream and starts over at the front."""

    def __init__(self) -> None:
        self.pinned = cp.cuda.alloc_pinned_memory(ARENA_BYTES)
        self.host = np.frombuffer(self.pinned, np.uint8, ARENA_BYTES)
        self.device = cp.empty(ARENA_BYTES, cp.uint8)
        self.used = 0

    def put(self, blob: np.ndarray) -> int:
        """The device address of a copy of ``blob`` (valid for kernels queued on this stream until
        the stream is next waited for by this arena)."""
        n = blob.nbytes
        if n > ARENA_BYTES:
            raise ValueError(f'a weight set of {n} bytes does not fit the {ARENA_BYTES} byte arena')
        at = -(-self.used // _ALIGN) * _ALIGN
        stream = cp.cuda.get_current_stream()
        if at + n > ARENA_BYTES:
            stream.synchronize()  # queued kernels read the device bytes, queued copies the host ones
            at = 0
        self.host[at : at + n] = blob
        base = int(self.device.data.ptr) + at
        cp.cuda.runtime.memcpyAsync(
            base, int(self.pinned.ptr) + at, n, cp.cuda.runtime.memcpyHostToDevice, stream.ptr
        )
        self.used = at + n
        return base


class PrepKernels:
    """One instance serves every worker thread; each worker stages its weights in its own arena
    (an array uploaded on one worker's stream is read only by kernels queued on that stream)."""

    def __init__(self) -> None:
        mod = cp.RawModule(code=_SRC)
        self.k_h = mod.get_function('resample_h')
        self.k_v = mod.get_function('resample_v')
        self.k_lap = mod.get_function('lap_sums')
        self._local = threading.local()

    def _arena(self) -> _Arena:
        arena: _Arena | None = getattr(self._local, 'arena', None)
        if arena is None:
            arena = self._local.arena = _Arena()
        return arena

    def resample_into(
        self,
        frame: Any,
        roi: tuple[int, int, int, int],
        size: tuple[int, int],
        dst: Any,
        window: tuple[int, int, int, int] | None = None,
        *,
        antialias: bool = True,
        lanczos: bool = False,
    ) -> None:
        """Resample ``frame[roi]`` (``x0, y0, w, h``) to ``size`` (``w, h``) and write its
        ``window`` (``left, top, w, h``; default all of it) into ``dst``, a CHW uint8 view."""
        h_img, w_img = int(frame.shape[0]), int(frame.shape[1])
        x0, y0, rw, rh = roi
        ow, oh = size
        left, top, ww, wh = window or (0, 0, ow, oh)
        wx = _weights(rw, ow, left, ww, antialias, lanczos)
        wy = _weights(rh, oh, top, wh, antialias, lanczos)
        arena = self._arena()
        px = arena.put(wx.blob)
        py = arena.put(wy.blob)
        rows = max(1, wy.hi - wy.lo)
        tmp = cp.empty((rows, ww, 3), dtype=cp.float32)
        self.k_h(
            ((ww + 127) // 128, rows),
            (128,),
            (frame, np.int32(h_img), np.int32(w_img), np.int32(x0), np.int32(y0 + wy.lo), np.int32(rows),
             np.int64(px), np.int64(px + 4 * wx.n), np.int64(px + 8 * wx.n), np.int32(wx.kmax), np.int32(ww), tmp),
        )  # fmt: skip
        plane = dst.strides[0] // dst.itemsize
        row_stride = dst.strides[1] // dst.itemsize
        # the device copy holds source row indices; the scratch starts at row ``wy.lo``
        self.k_v(
            ((ww + 127) // 128, wh),
            (128,),
            (tmp, np.int32(ww), np.int64(py), np.int64(py + 4 * wy.n), np.int64(py + 8 * wy.n),
             np.int32(wy.kmax), np.int32(wh), np.int32(wy.lo), dst, np.int64(plane), np.int32(row_stride)),
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
