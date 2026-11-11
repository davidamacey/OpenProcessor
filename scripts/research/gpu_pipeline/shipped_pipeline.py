"""The shipped ``gpu_image_prep`` algorithm outside Triton, stage by stage.

Same decode lanes, kernels and geometry as ``models/gpu_image_prep/1`` (imported by path, not
copied); the two BLS calls become direct TensorRT calls (``trt_runner``). Every stage is wrapped in
an NVTX range so an ``nsys`` trace attributes GPU kernel time to it, and timed on the wall clock.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import harness_common as common


# ruff: noqa: E402
common.use_shipped_prep_modules()

import cupy as cp
import cv2
import numpy as np
import nvtx
from nvjpeg_lanes import NvJpegLanes
from prep_geometry import (
    blur_rect,
    crop_rect,
    letterbox_geometry,
    oriented_size,
    route,
    undo_letterbox,
    whole_square_geometry,
)
from prep_kernels import PrepKernels


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from trt_runner import TrtModel

STAGES = (
    'read',
    'decode',
    'letterbox',
    'detect',
    'embed_frame',
    'embed_crops',
    'embed_pe',
    'blur',
    'crop_copy',
    'crop_jpeg',
)
NET, PE_SIZE, PE_BATCH, FILL, JPEG_QUALITY = 640, 336, 32, 114, 90


@dataclass
class Result:
    status: str = 'ok'
    stages: dict[str, float] = field(default_factory=dict)
    finished: float = 0.0
    detections: int = 0
    # kept for parity checks: frame size, letterbox canvas, squares, crop rects, vectors, blur sums
    size: tuple[int, int] = (0, 0)
    rects: list[tuple[int, int, int, int]] = field(default_factory=list)
    boxes: np.ndarray | None = None
    vectors: np.ndarray | None = None
    sums: np.ndarray | None = None
    canvas: np.ndarray | None = None
    squares: np.ndarray | None = None


@contextmanager
def stage(res: Result, name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    with nvtx.annotate(name):
        yield
    res.stages[name] = res.stages.get(name, 0.0) + (time.perf_counter() - t0)


def orient_gpu(frame: Any, orientation: int) -> Any:
    """EXIF re-orientation on the GPU (the logic of ``model._orient``)."""
    t = frame
    if orientation in (5, 6, 7, 8):
        t = t.transpose(1, 0, 2)
    if orientation in (2, 3, 6, 7):
        t = t[:, ::-1]
    if orientation in (3, 4, 7, 8):
        t = t[::-1]
    if t is frame:
        return frame
    out = cp.empty(t.shape, cp.uint8)
    out[...] = t
    return out


class ShippedPipeline:
    def __init__(
        self, detector: TrtModel, encoder: TrtModel, *, lanes: int = 16, workers: int = 16
    ) -> None:
        self.detector, self.encoder = detector, encoder
        self.lanes = NvJpegLanes(lanes=lanes)
        self.kernels = PrepKernels()
        self.alloc_stream = cp.cuda.Stream(non_blocking=True)
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix='pr-img')
        self.encode_pool = ThreadPoolExecutor(8, thread_name_prefix='pr-jpeg')
        self.local = threading.local()
        cp.get_default_memory_pool().set_limit(size=4096 * 1024 * 1024)

    def run_batch(self, paths: list[Path], *, keep: bool = False) -> list[Result]:
        return list(self.pool.map(lambda p: self.process(p, keep=keep), paths))

    def process(self, path: Path, *, keep: bool = False) -> Result:
        res = Result()
        stream = getattr(self.local, 'stream', None)
        if stream is None:
            stream = self.local.stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            self._process(path, res, keep)
            stream.synchronize()
        res.finished = time.perf_counter()
        return res

    def _process(self, path: Path, res: Result, keep: bool) -> None:
        with stage(res, 'read'):
            data = path.read_bytes()
        reason, header = route(data, 0.0, 100.0)
        if reason != 'gpu':
            res.status = reason
            return
        with stage(res, 'decode'):
            frame, _backend = self.lanes.decode(data, self.alloc_stream)
            frame = orient_gpu(frame, header.orientation)
        w, h = oriented_size(header)
        res.size = (w, h)
        lb = letterbox_geometry(w, h, NET)
        with stage(res, 'letterbox'):
            canvas = cp.full((1, 3, NET, NET), FILL, cp.uint8)
            dst = canvas[0, :, lb.top : lb.top + lb.new_h, lb.left : lb.left + lb.new_w]
            self.kernels.resample_into(frame, (0, 0, w, h), (lb.new_w, lb.new_h), dst)
            canvas_copy = canvas.get() if keep else None
        with stage(res, 'detect'):
            out = self.detector.run(canvas)
            n = int(out['num_dets'][0][0])
            boxes = out['det_boxes'][0][:n].astype(np.float32).reshape(-1, 4)
            rects = [crop_rect(undo_letterbox(tuple(b), NET, lb), w, h) for b in boxes]
        res.detections, res.rects, res.boxes = n, rects, boxes
        with stage(res, 'crop_copy'):
            pixels = [
                cp.ascontiguousarray(frame[y1:y2, x1:x2, ::-1]).get() for x1, y1, x2, y2 in rects
            ]
        jpegs = [self.encode_pool.submit(_encode, px) for px in pixels]
        square = cp.empty((1 + len(rects), 3, PE_SIZE, PE_SIZE), cp.uint8)
        with stage(res, 'embed_frame'):
            g = whole_square_geometry(w, h, PE_SIZE)
            self.kernels.resample_into(
                frame,
                (0, 0, w, h),
                (g.new_w, g.new_h),
                square[0],
                (g.left, g.top, PE_SIZE, PE_SIZE),
            )
        with stage(res, 'embed_crops'):
            for k, (x1, y1, x2, y2) in enumerate(rects, start=1):
                g = whole_square_geometry(x2 - x1, y2 - y1, PE_SIZE)
                self.kernels.resample_into(
                    frame, (x1, y1, x2 - x1, y2 - y1), (g.new_w, g.new_h), square[k],
                    (g.left, g.top, PE_SIZE, PE_SIZE), antialias=False,
                )  # fmt: skip
        with stage(res, 'embed_pe'):
            rows = [
                self.encoder.run(cp.ascontiguousarray(square[s : s + PE_BATCH]))['image_embeddings']
                for s in range(0, int(square.shape[0]), PE_BATCH)
            ]
            vectors = np.concatenate(rows, axis=0).astype(np.float32)
        with stage(res, 'blur'):
            clamped = [blur_rect(undo_letterbox(tuple(b), NET, lb), w, h) for b in boxes]
            lap_rects = [(0, 0, w, h)] + [r for r in clamped if r is not None]
            sums = cp.asnumpy(self.kernels.lap_sums(frame, lap_rects))
        with stage(res, 'crop_jpeg'):
            for f in jpegs:
                f.result()
        if keep:
            res.vectors, res.sums, res.canvas, res.squares = (
                vectors,
                sums,
                canvas_copy,
                square.get(),
            )
        del frame


def _encode(pixels: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError('crop JPEG encode failed')
    return buf.tobytes()
