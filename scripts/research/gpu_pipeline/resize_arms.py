"""Arms of gpu:PR-2: the four image operations of the shipped model on a decoded GPU frame.

Each arm implements ``letterbox``, ``whole_square``, ``crop_squares`` and ``blur_sums``. K-a is the
shipped CuPy kernel set (``prep_kernels.py``). K-b is CV-CUDA (``pillowresize``, ``resize``,
``cvtcolor``, ``laplacian``; github.com/CVCUDA/CV-CUDA). K-c is DALI ``fn.resize`` in a
one-image pipeline (docs.nvidia.com/deeplearning/dali, "nvidia.dali.fn.resize").
Outputs are CuPy arrays: letterbox ``[1, 3, 640, 640]``, squares ``[k, 3, 336, 336]`` uint8.
"""

from __future__ import annotations

from typing import Any

import harness_common as common


# ruff: noqa: E402
common.use_shipped_prep_modules()

import cupy as cp
import nvtx
from prep_geometry import letterbox_geometry, whole_square_geometry
from prep_kernels import PrepKernels


NET, PE, FILL = 640, 336, 114


class ShippedKernels:
    name = 'K-a cupy-kernels'

    def __init__(self) -> None:
        self.k = PrepKernels()

    def letterbox(self, frame: Any) -> Any:
        h, w = frame.shape[:2]
        lb = letterbox_geometry(w, h, NET)
        canvas = cp.full((1, 3, NET, NET), FILL, cp.uint8)
        dst = canvas[0, :, lb.top : lb.top + lb.new_h, lb.left : lb.left + lb.new_w]
        self.k.resample_into(frame, (0, 0, w, h), (lb.new_w, lb.new_h), dst)
        return canvas

    def whole_square(self, frame: Any) -> Any:
        h, w = frame.shape[:2]
        g = whole_square_geometry(w, h, PE)
        out = cp.empty((1, 3, PE, PE), cp.uint8)
        self.k.resample_into(
            frame, (0, 0, w, h), (g.new_w, g.new_h), out[0], (g.left, g.top, PE, PE)
        )
        return out

    def crop_squares(self, frame: Any, rects: list[tuple[int, int, int, int]]) -> Any:
        out = cp.empty((len(rects), 3, PE, PE), cp.uint8)
        for k, (x1, y1, x2, y2) in enumerate(rects):
            g = whole_square_geometry(x2 - x1, y2 - y1, PE)
            self.k.resample_into(frame, (x1, y1, x2 - x1, y2 - y1), (g.new_w, g.new_h), out[k],
                                 (g.left, g.top, PE, PE), antialias=False)  # fmt: skip
        return out

    def blur_sums(self, frame: Any, rects: list[tuple[int, int, int, int]]) -> Any:
        h, w = frame.shape[:2]
        return cp.asnumpy(self.k.lap_sums(frame, [(0, 0, w, h), *rects]))


class CvCuda:
    name = 'K-b cvcuda'

    def __init__(self) -> None:
        import cvcuda

        self.cv = cvcuda

    def _tensor(self, hwc: Any) -> Any:
        return self.cv.as_tensor(cp.ascontiguousarray(hwc)[None], 'NHWC')

    def _resized(self, hwc: Any, size: tuple[int, int], *, pillow: bool) -> Any:
        w, h = size
        src = self._tensor(hwc)
        if pillow:
            out = self.cv.pillowresize(
                src, (1, h, w, 3), self.cv.Format.RGB8, self.cv.Interp.LINEAR
            )
        else:
            out = self.cv.resize(src, (1, h, w, 3), self.cv.Interp.LINEAR)
        return cp.asarray(out.cuda())[0]

    def letterbox(self, frame: Any) -> Any:
        h, w = frame.shape[:2]
        lb = letterbox_geometry(w, h, NET)
        canvas = cp.full((1, 3, NET, NET), FILL, cp.uint8)
        small = self._resized(frame, (lb.new_w, lb.new_h), pillow=True)
        canvas[0, :, lb.top : lb.top + lb.new_h, lb.left : lb.left + lb.new_w] = small.transpose(
            2, 0, 1
        )
        return canvas

    def whole_square(self, frame: Any) -> Any:
        h, w = frame.shape[:2]
        g = whole_square_geometry(w, h, PE)
        big = self._resized(frame, (g.new_w, g.new_h), pillow=True)
        return cp.ascontiguousarray(
            big[g.top : g.top + PE, g.left : g.left + PE].transpose(2, 0, 1)
        )[None]

    def crop_squares(self, frame: Any, rects: list[tuple[int, int, int, int]]) -> Any:
        out = cp.empty((len(rects), 3, PE, PE), cp.uint8)
        for k, (x1, y1, x2, y2) in enumerate(rects):
            g = whole_square_geometry(x2 - x1, y2 - y1, PE)
            big = self._resized(frame[y1:y2, x1:x2], (g.new_w, g.new_h), pillow=False)
            out[k] = big[g.top : g.top + PE, g.left : g.left + PE].transpose(2, 0, 1)
        return out

    def blur_sums(self, frame: Any, rects: list[tuple[int, int, int, int]]) -> Any:
        """(sum, sum of squares) of the 3 x 3 Laplacian of the grey plane per rectangle, as float
        sums: grey by ``cvtcolor``, ``convertto`` float32, ``laplacian`` ksize 1 (reflect-101)."""
        cv = self.cv
        h, w = frame.shape[:2]
        rows = []
        for x1, y1, x2, y2 in [(0, 0, w, h), *rects]:
            grey = cv.cvtcolor(self._tensor(frame[y1:y2, x1:x2]), cv.ColorConversion.RGB2GRAY)
            lap = cv.laplacian(cv.convertto(grey, cv.Type.F32), 1, 1.0, cv.Border.REFLECT101)
            a = cp.asarray(lap.cuda()).astype(cp.float64)
            rows.append((float(a.sum()), float((a * a).sum())))
        return rows


class CvCudaFused(CvCuda):
    """K-b variant: the whole-frame square in one ``resize_crop_convert_reformat`` call (output
    NCHW uint8). Everything else is K-b's."""

    name = 'K-b-fused cvcuda'

    def whole_square(self, frame: Any) -> Any:
        h, w = frame.shape[:2]
        g = whole_square_geometry(w, h, PE)
        cv = self.cv
        out = cv.resize_crop_convert_reformat(
            self._tensor(frame),
            (g.new_w, g.new_h),
            cv.Interp.LINEAR,
            cv.RectI(g.left, g.top, PE, PE),
            layout='NCHW',
        )
        return cp.asarray(out.cuda())


class DaliResize:
    name = 'K-c dali-fn.resize'

    def __init__(self) -> None:
        self.pipes = {aa: self._build(aa) for aa in (True, False)}

    @staticmethod
    def _build(antialias: bool) -> Any:
        from nvidia.dali import fn, pipeline_def, types

        @pipeline_def(batch_size=1, num_threads=2, device_id=0, prefetch_queue_depth=1)
        def graph() -> Any:
            img = fn.external_source(name='img', device='gpu', layout='HWC')
            size = fn.external_source(name='size', device='cpu')
            rs = fn.external_source(name='rs', device='cpu')
            re_ = fn.external_source(name='re', device='cpu')
            return fn.resize(img, resize_x=size[0], resize_y=size[1], roi_start=rs, roi_end=re_,
                             interp_type=types.INTERP_LINEAR, antialias=antialias, device='gpu')  # fmt: skip

        pipe = graph()
        pipe.build()
        return pipe

    def _resize(
        self,
        frame: Any,
        roi: tuple[int, int, int, int],
        size: tuple[int, int],
        *,
        antialias: bool = True,
    ) -> Any:
        import numpy as np

        x1, y1, x2, y2 = roi
        pipe = self.pipes[antialias]
        pipe.feed_input('img', [frame])
        pipe.feed_input('size', [np.array(size, np.float32)])
        pipe.feed_input('rs', [np.array([y1, x1], np.float32)])
        pipe.feed_input('re', [np.array([y2, x2], np.float32)])
        (out,) = pipe.run()
        return cp.asarray(out[0])

    def letterbox(self, frame: Any) -> Any:
        h, w = frame.shape[:2]
        lb = letterbox_geometry(w, h, NET)
        canvas = cp.full((1, 3, NET, NET), FILL, cp.uint8)
        small = self._resize(frame, (0, 0, w, h), (lb.new_w, lb.new_h))
        canvas[0, :, lb.top : lb.top + lb.new_h, lb.left : lb.left + lb.new_w] = small.transpose(
            2, 0, 1
        )
        return canvas

    def whole_square(self, frame: Any) -> Any:
        h, w = frame.shape[:2]
        g = whole_square_geometry(w, h, PE)
        big = self._resize(frame, (0, 0, w, h), (g.new_w, g.new_h))
        return cp.ascontiguousarray(
            big[g.top : g.top + PE, g.left : g.left + PE].transpose(2, 0, 1)
        )[None]

    def crop_squares(self, frame: Any, rects: list[tuple[int, int, int, int]]) -> Any:
        out = cp.empty((len(rects), 3, PE, PE), cp.uint8)
        for k, rect in enumerate(rects):
            g = whole_square_geometry(rect[2] - rect[0], rect[3] - rect[1], PE)
            big = self._resize(frame, rect, (g.new_w, g.new_h), antialias=False)
            out[k] = big[g.top : g.top + PE, g.left : g.left + PE].transpose(2, 0, 1)
        return out

    def blur_sums(self, frame: Any, rects: list[tuple[int, int, int, int]]) -> Any:
        raise NotImplementedError('K-c has no Laplacian operator; the blur sums stay on K-a')


def build(name: str) -> Any:
    return {'K-a': ShippedKernels, 'K-b': CvCuda, 'K-b-fused': CvCudaFused, 'K-c': DaliResize}[
        name
    ]()


def traced(label: str) -> Any:
    return nvtx.annotate(label)
