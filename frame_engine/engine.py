"""The per-photo pipeline: decode once on the GPU, cut every model input from that frame.

One request per image (protocol version 1, :mod:`src.domain.frame_engine_wire`). Requests run on a
pool of worker threads, each with its own CuPy stream, so decoding, kernels and the model calls of
different images overlap; the detector and encoder calls of concurrent images meet in those
models' own dynamic batchers in Triton. Per image, and only for the branches the plan names:

1. Route by header (JPEG, 8-bit, 1 or 3 components, baseline, end-of-image marker present, size
   gate); anything else answers with a status (the API decodes that image on the CPU, counted).
2. Reserve the frame's bytes (``prep_budget``, a bounded wait), nvJPEG decode (GPU Huffman lanes),
   EXIF orientation applied on the GPU.
3. ``detect``: the detector letterbox (uint8, Pillow ``BILINEAR``) and the detector call; or the
   caller's stored ``boxes`` (the embed pass), cut by the rectangle the CPU pass cuts.
4. ``crops: inline``: each rectangle's pixels copied to the host and encoded as a JPEG on host
   threads (quality 90), started before the encoder call so the two overlap.
5. ``embed_frame`` / ``embed_crops``: the whole-frame square (Pillow ``BILINEAR``, shorter edge to
   the encoder size, centre crop) and one square per wanted rectangle (``crop_pil``'s rectangle,
   plain bilinear as ``resize_crop_rgb``), through the encoder in chunks of the plan's batch.
6. ``detect``: the blur sums of the frame and of every detection's clamped rectangle.

The kernels, their order and the JPEG settings are those of the Triton prep model this engine
replaces, so the outputs are bit-identical for the photos it decoded. Model inputs travel through
:mod:`frame_engine.transport`.

Device memory: a request reserves its frame's bytes from the budget before it decodes; frames and
orientation copies are allocated on one dedicated stream (``alloc_stream``), so a block freed by
one worker serves any other. That is only safe if no kernel of the freeing worker still reads the
block: every path that drops a frame waits for its worker stream first. An allocation that fails
anyway answers ``gpu_memory``; a photo larger than the whole budget answers ``over_budget``; a
budget wait that times out raises :class:`~frame_engine.transport.OverloadError`.

Time: every model call ends by the request's deadline (:mod:`frame_engine.transport`), and the
request records the stage it is in (``EngineRequest.stage``) for the front's deadline log.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cupy as cp
import cv2
import numpy as np

from frame_engine.messages import EngineRequest, EngineResult
from frame_engine.nvjpeg_lanes import NvJpegError, NvJpegLanes
from frame_engine.prep_budget import ByteBudget, is_memory_error, need_bytes
from frame_engine.prep_geometry import blur_rect, oriented_size, route, stored_box_rects
from frame_engine.prep_kernels import PrepKernels
from frame_engine.slab import SlabTooLargeError
from frame_engine.stage_metrics import StageClock
from frame_engine.transport import ModelCallError, OverloadError
from src.domain import frame_engine_wire as wire
from src.domain.image_geometry import (
    crop_rect,
    letterbox_geometry,
    undo_letterbox,
    whole_square_geometry,
)


if TYPE_CHECKING:
    from frame_engine.prep_geometry import JpegHeader
    from frame_engine.settings import EngineSettings
    from frame_engine.stage_metrics import EngineMetrics
    from frame_engine.transport import BodyTransport, CudaShmTransport


#: The detector's single fused tensor ``[batch, 300, 6]``: ``x1, y1, x2, y2, score, class`` in
#: letterboxed input pixels, zero-padded past the last detection.
DET_OUTPUT = 'output0'
EMBED_OUTPUT = 'image_embeddings'
MIB = 1024 * 1024
logger = logging.getLogger('frame_engine')


def _orient(frame: Any, orientation: int, alloc_stream: Any) -> Any:
    """Re-orient an HWC frame by its EXIF tag as ``ImageOps.exif_transpose`` does.

    The copy lives in the shared arena, is written on the current (worker) stream, and the source
    block returns to the arena on the caller's next line, so the copy is waited for here.
    """
    t = frame
    if orientation in (5, 6, 7, 8):
        t = t.transpose(1, 0, 2)
    if orientation in (2, 3, 6, 7):
        t = t[:, ::-1]
    if orientation in (3, 4, 7, 8):
        t = t[::-1]
    if t is frame:
        return frame
    with alloc_stream:
        out = cp.empty(t.shape, cp.uint8)
    out[...] = t
    cp.cuda.get_current_stream().synchronize()
    return out


def _ordered(
    plan: wire.EnginePlan, states: dict[str, tuple[wire.BranchState, str]]
) -> dict[str, tuple[wire.BranchState, str]]:
    """One entry per reported name, in the plan's order; a missing entry is a bug, not ``ok``."""
    missing = [name for name in plan.reported() if name not in states]
    if missing:
        raise RuntimeError(f'no branch status for {missing}')
    return {name: states[name] for name in plan.reported()}


def _sync_stream() -> None:
    cp.cuda.get_current_stream().synchronize()


@dataclass
class _Image:
    frame: Any
    size: tuple[int, int]
    rects: list[tuple[int, int, int, int]] = field(default_factory=list)
    boxes: np.ndarray | None = None
    scores: np.ndarray | None = None
    classes: np.ndarray | None = None
    lb: Any = None


class FrameEngine:
    def __init__(
        self,
        settings: EngineSettings,
        transport: BodyTransport | CudaShmTransport,
        metrics: EngineMetrics,
        *,
        device: int = 0,
    ) -> None:
        self.s = settings
        self.transport = transport
        self.metrics = metrics
        self.device = device
        cp.cuda.Device(device).use()
        # CuPy keeps freed blocks for reuse; photo frames differ in size, so without a cap the pool
        # grows past 15 GB. At the cap it frees cached blocks before allocating.
        cp.get_default_memory_pool().set_limit(size=settings.pool_mib * MIB)
        self.lanes = NvJpegLanes(lanes=settings.lanes)
        self.alloc_stream = cp.cuda.Stream(non_blocking=True)
        self.budget = self._reserve_budget(settings.budget_mib)
        metrics.watch_budget(self.budget)
        self.kernels = PrepKernels()
        self.encode_pool = ThreadPoolExecutor(settings.encode_threads, thread_name_prefix='fe-jpeg')
        self.workers = ThreadPoolExecutor(settings.workers, thread_name_prefix='fe-image')
        self.local = threading.local()
        self.memory_events = 0
        self.region_events = 0

    def _reserve_budget(self, budget_mib: float) -> ByteBudget:
        """Hold the whole budget in the shared arena once, then hand it back to the pool (which
        keeps it cached for the frames). A card that cannot spare it fails the start here."""
        total = int(budget_mib * MIB)
        with self.alloc_stream:
            block = cp.empty(total, cp.uint8)
        del block
        return ByteBudget(total)

    def submit(self, request: EngineRequest) -> Future[EngineResult]:
        return self.workers.submit(self.run, request)

    def close(self) -> None:
        """Queued photos are dropped (their callers are gone); running ones finish, every wait
        inside one being bounded."""
        self.workers.shutdown(wait=True, cancel_futures=True)
        self.encode_pool.shutdown(wait=True)

    # ------------------------------------------------------------------ one request

    def run(self, request: EngineRequest) -> EngineResult:
        clock = StageClock(_sync_stream if self.s.stage_sync else None, request.enter)
        clock.add('queue', time.perf_counter() - request.arrived)
        try:
            stream = getattr(self.local, 'stream', None)
            if stream is None:
                cp.cuda.Device(self.device).use()
                stream = self.local.stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                result = self._process(request, clock)
        finally:
            self.metrics.publish(clock)
        self.metrics.outcomes.labels(result.status, result.decode_route).inc()
        return result

    def _failed(self, plan: wire.EnginePlan, status: str) -> EngineResult:
        branches = self._initial_states(plan)
        for name in plan.reported():
            branches.setdefault(name, ('error', wire.IMAGE_FAILED))
        return EngineResult(status, wire.DECODED_NONE, _ordered(plan, branches))

    @staticmethod
    def _initial_states(plan: wire.EnginePlan) -> dict[str, tuple[wire.BranchState, str]]:
        return dict.fromkeys(plan.unsupported(), ('error', wire.UNSUPPORTED_BRANCH))

    def _note_memory_failure(self, where: str, exc: BaseException) -> None:
        """Count-worthy but rare once the budget holds: log the first and every hundredth."""
        self.memory_events += 1
        if self.memory_events % 100 == 1:
            logger.warning(
                'device memory failure in %s (%s: %s); event %d, budget in use %d of %d bytes',
                where, type(exc).__name__, exc, self.memory_events,
                self.budget.in_use, self.budget.total,
            )  # fmt: skip

    def _note_region_too_small(self, exc: SlabTooLargeError) -> None:
        """A model input larger than the whole region: a size setting, so one line, no trace."""
        self.region_events += 1
        if self.region_events % 100 == 1:
            logger.warning(
                '%s: raise OP_FRAME_ENGINE_REGION_MIB (photos answer over_budget); event %d',
                exc, self.region_events,
            )  # fmt: skip

    def _process(self, request: EngineRequest, clock: StageClock) -> EngineResult:
        plan = request.plan
        with clock.stage('decode'):
            reason, header = route(request.data, self.s.min_megapixels, self.s.max_megapixels)
        if reason != 'gpu':
            return self._failed(plan, reason)
        data = request.data[: header.end]
        need = need_bytes(header)
        if need > self.budget.total:
            return self._failed(plan, 'over_budget')
        with clock.stage('budget'):
            admitted = self.budget.acquire(need, self.s.budget_wait_seconds)
        self.metrics.budget_waits.labels('admitted' if admitted else 'timeout').inc()
        if not admitted:
            raise OverloadError(
                'budget', f'{need} bytes not free within {self.s.budget_wait_seconds:g} s'
            )
        im: _Image | None = None
        try:
            with clock.stage('decode'):
                im_or_reason = self._decode(data, header)
            if isinstance(im_or_reason, str):
                return self._failed(plan, im_or_reason)
            im = im_or_reason
            return self._branches(request, im, clock)
        except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError) as exc:
            if not is_memory_error(exc):
                raise
            self._note_memory_failure('process', exc)
            return self._failed(plan, 'gpu_memory')
        except SlabTooLargeError as exc:
            self._note_region_too_small(exc)
            return self._failed(plan, 'over_budget')
        finally:
            # every path that drops the frame has waited for this stream, a failed one included
            if im is not None and im.frame is not None:
                cp.cuda.get_current_stream().synchronize()
                im.frame = None
            self.budget.release(need)

    def _decode(self, data: bytes, header: JpegHeader) -> _Image | str:
        try:
            frame, _backend = self.lanes.decode(data, self.alloc_stream)
        except (
            NvJpegError,
            cp.cuda.memory.OutOfMemoryError,
            cp.cuda.runtime.CUDARuntimeError,
        ) as exc:
            if is_memory_error(exc):
                self._note_memory_failure('decode', exc)
                return 'gpu_memory'
            return 'decoder'
        frame = _orient(frame, header.orientation, self.alloc_stream)
        w, h = oriented_size(header)
        if (int(frame.shape[1]), int(frame.shape[0])) != (w, h):
            return 'decoder'
        return _Image(frame=frame, size=(w, h))

    def _branches(self, request: EngineRequest, im: _Image, clock: StageClock) -> EngineResult:
        plan = request.plan
        states = self._initial_states(plan)
        w, h = im.size
        out: dict[str, Any] = {wire.SIZE: np.array([w, h], np.int32)}
        have_rects = False
        if plan.runs('detect') and plan.detector is not None:
            try:
                self._detect(im, plan.detector, clock, request.deadline)
                states['detect'] = ('ok', '')
                have_rects = True
                out[wire.LETTERBOX] = np.array([im.lb.scale, im.lb.pad_w, im.lb.pad_h], np.float64)
                out[wire.DET_BOXES] = im.boxes
                out[wire.DET_SCORES] = im.scores
                out[wire.DET_CLASSES] = im.classes
            except ModelCallError as exc:
                logger.warning('detector call failed: %s', exc)
                states['detect'] = ('error', wire.MODEL_ERROR)
        elif request.boxes is not None:
            im.rects = stored_box_rects(request.boxes, w, h)
            have_rects = True
        jpegs: list[Future[tuple[bytes, float]]] = []
        if plan.crops == 'inline' and have_rects:
            with clock.stage('crop_copy'):
                pixels = self._crop_pixels(im)
            jpegs = [self.encode_pool.submit(self._encode, px) for px in pixels]
        try:
            self._embed_branches(request, im, have_rects, states, out, clock)
            if states.get('detect') == ('ok', ''):
                with clock.stage('blur'):
                    out[wire.BLUR_SUMS] = self._blur_sums(im)
            cp.cuda.get_current_stream().synchronize()
            im.frame = None
        finally:
            done = [f.result() for f in jpegs]
        if plan.crops == 'inline' and have_rects:
            out[wire.CROP_JPEGS] = [data for data, _ in done]  # empty when nothing was found
        if jpegs:
            clock.add('crop_jpeg', sum(secs for _, secs in done))
        return EngineResult(wire.STATUS_OK, wire.DECODED_GPU, _ordered(plan, states), out)

    # ------------------------------------------------------------------ stages

    def _detect(
        self, im: _Image, det: wire.DetectorSpec, clock: StageClock, deadline: float
    ) -> None:
        w, h = im.size
        net = det.input_size
        lb = letterbox_geometry(w, h, net)
        im.lb = lb
        staged = None
        try:
            with clock.stage('letterbox'):
                staged = self.transport.stage(3 * net * net)
                canvas = staged.device.reshape((1, 3, net, net))
                canvas.fill(det.letterbox_fill)
                dst = canvas[0, :, lb.top : lb.top + lb.new_h, lb.left : lb.left + lb.new_w]
                self.kernels.resample_into(im.frame, (0, 0, w, h), (lb.new_w, lb.new_h), dst)
            with clock.stage('detect'):
                self.transport.commit(staged)
                fused = self.transport.infer(
                    det.model, staged, 0, [1, 3, net, net], DET_OUTPUT, deadline=deadline
                )
        finally:
            if staged is not None:
                self.transport.release(staged, _sync_stream)
        with clock.stage('detect'):
            rows = fused[0][fused[0][:, 4] > 0.0]
            im.boxes = rows[:, :4].astype(np.float32)
            im.scores = rows[:, 4].astype(np.float32)
            im.classes = rows[:, 5].astype(np.int32)
            im.rects = [
                crop_rect(undo_letterbox(tuple(b), lb.scale, (lb.pad_w, lb.pad_h)), w, h)
                for b in im.boxes
            ]

    def _embed_branches(
        self,
        request: EngineRequest,
        im: _Image,
        have_rects: bool,
        states: dict[str, tuple[wire.BranchState, str]],
        out: dict[str, Any],
        clock: StageClock,
    ) -> None:
        plan = request.plan
        frame = plan.runs('embed_frame')
        crops = plan.runs('embed_crops')
        if crops and not have_rects:
            states['embed_crops'] = ('error', wire.DEPENDENCY_FAILED)
            crops = False
        if not (frame or crops) or plan.embed is None:
            return
        mask = np.zeros(len(im.rects), np.bool_)
        if crops:
            if plan.embed_classes is None or im.classes is None:
                mask[:] = True
            else:
                wanted = np.array(sorted(plan.embed_classes), np.int32)
                mask = np.isin(im.classes, wanted)
        picked = [r for r, keep in zip(im.rects, mask, strict=True) if keep]
        try:
            vectors = self._embed(im, plan.embed, frame, picked, clock, request.deadline)
        except ModelCallError as exc:
            logger.warning('encoder call failed: %s', exc)
            for name in ('embed_frame', 'embed_crops'):
                if plan.runs(name) and name not in states:
                    states[name] = ('error', wire.MODEL_ERROR)
            return
        if frame:
            out[wire.WHOLE_EMBEDDING] = vectors[0]
            states['embed_frame'] = ('ok', '')
        if crops:
            rows = vectors[1:] if frame else vectors
            out[wire.CROP_EMBEDDINGS] = rows  # [len(picked), dim]; reshape(0, -1) cannot infer dim
            out[wire.CROP_EMBEDDED] = mask
            states['embed_crops'] = ('ok', '')

    def _embed(
        self,
        im: _Image,
        spec: wire.EmbedSpec,
        frame: bool,
        rects: list[tuple[int, int, int, int]],
        clock: StageClock,
        deadline: float,
    ) -> np.ndarray:
        """Encoder vectors of the whole-frame square (when ``frame``), then one per rectangle.

        The squares are staged one encoder call (``spec.batch`` squares) at a time, so the slice of
        the region a photo holds is bounded by the batch, not by its number of detections.
        """
        t = spec.size
        n = int(frame) + len(rects)
        if n == 0:
            return np.zeros((0, 0), np.float32)
        sqb = 3 * t * t
        w, h = im.size
        rows = []
        for s in range(0, n, spec.batch):
            m = min(spec.batch, n - s)
            staged = self.transport.stage(m * sqb)
            try:
                sq = staged.device.reshape((m, 3, t, t))
                if frame and s == 0:
                    with clock.stage('embed_frame'):
                        g = whole_square_geometry(w, h, t)
                        self.kernels.resample_into(
                            im.frame, (0, 0, w, h), (g.new_w, g.new_h), sq[0], (g.left, g.top, t, t)
                        )
                with clock.stage('embed_crops'):
                    for k in range(max(s, int(frame)), s + m):
                        x1, y1, x2, y2 = rects[k - int(frame)]
                        g = whole_square_geometry(x2 - x1, y2 - y1, t)
                        self.kernels.resample_into(
                            im.frame,
                            (x1, y1, x2 - x1, y2 - y1),
                            (g.new_w, g.new_h),
                            sq[k - s],
                            (g.left, g.top, t, t),
                            antialias=False,
                        )
                with clock.stage('embed_pe'):
                    self.transport.commit(staged)
                    rows.append(
                        self.transport.infer(
                            spec.model, staged, 0, [m, 3, t, t], EMBED_OUTPUT, deadline=deadline
                        )
                    )
            finally:
                self.transport.release(staged, _sync_stream)
        return np.concatenate(rows, axis=0).astype(np.float32)

    def _blur_sums(self, im: _Image) -> np.ndarray:
        """``[1 + detections, 3]`` (pixels, sum, sum of squares): the frame, then each box."""
        w, h = im.size
        boxes = im.boxes if im.boxes is not None else np.zeros((0, 4), np.float32)
        clamped = [
            blur_rect(undo_letterbox(tuple(b), im.lb.scale, (im.lb.pad_w, im.lb.pad_h)), w, h)
            for b in boxes
        ]
        rects = [(0, 0, w, h)] + [r for r in clamped if r is not None]
        sums = iter(cp.asnumpy(self.kernels.lap_sums(im.frame, rects)))
        rows = [(w * h, *next(sums))]
        rows.extend(
            (0, 0, 0) if r is None else ((r[2] - r[0]) * (r[3] - r[1]), *next(sums))
            for r in clamped
        )
        return np.array(rows, np.int64).reshape(-1, 3)

    def _crop_pixels(self, im: _Image) -> list[np.ndarray]:
        """Each rectangle (inside the frame), BGR (host)."""
        return [
            cp.ascontiguousarray(im.frame[y1:y2, x1:x2, ::-1]).get() for x1, y1, x2, y2 in im.rects
        ]

    def _encode(self, pixels: np.ndarray) -> tuple[bytes, float]:
        """The JPEG and the seconds the encode took on its host thread."""
        start = time.perf_counter()
        side = max(pixels.shape[:2])
        cap = self.s.crop_max_side
        if 0 < cap < side:  # cap before encode: area filter, never upscaled
            scale = cap / side
            h, w = pixels.shape[:2]
            pixels = cv2.resize(
                pixels,
                (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, self.s.jpeg_quality])
        if not ok:
            raise RuntimeError('crop JPEG encode failed')
        return buf.tobytes(), time.perf_counter() - start
