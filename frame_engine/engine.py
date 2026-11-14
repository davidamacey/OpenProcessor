"""The per-photo pipeline: decode once on the GPU, cut every model input from that frame.

One request per image (protocol version 1, :mod:`src.domain.frame_engine_wire`). Requests run on a
pool of worker threads, each with its own CuPy stream, so decoding, kernels and the model calls of
different images overlap; the calls of concurrent images meet in the models' own dynamic batchers
in Triton. This module is the orchestrator: routing, the device-memory budget, the decode, the two
rounds of model calls and the mapping of failures. The branches themselves live in
:mod:`frame_engine.branches` (contract in its docstring). Per image:

1. Route by header (JPEG, 8-bit, 1 or 3 components, baseline, end-of-image marker present, size
   gate); anything else answers with a status (the API decodes that image on the CPU, counted).
2. Reserve the frame's bytes (``prep_budget``, a bounded wait), nvJPEG decode (GPU Huffman lanes),
   EXIF orientation applied on the GPU.
3. Round 1: every requested branch that needs only the frame stages its input and its call (the
   detector letterbox, the whole-frame encoder square); one wait for the stream, then all those
   calls go out at once. The caller's stored ``boxes`` (the embed pass) stand in for the detector.
4. Settle: the answers later branches read (the detector's boxes as frame rectangles).
5. Round 2: the branches cut from those answers stage and submit every call before collecting any
   (the encoder squares of the rectangles, a call per ``spec.batch``), within the photo's window
   of the region; then host work overlaps the calls in flight (crop JPEGs).
6. Finish: collect, blur sums of the frame and of every detection, outputs.

The kernels, their order and the JPEG settings are those of the Triton prep model this engine
replaces, so the outputs are bit-identical for the photos it decoded. Model inputs travel through
:mod:`frame_engine.transport`.

Device memory: a request reserves its frame's bytes from the budget before it decodes; frames and
orientation copies are allocated on one dedicated stream (``alloc_stream``), so a block freed by
one worker serves any other. That is only safe if no kernel of the freeing worker still reads the
block: every path that drops a frame waits for its worker stream first. An allocation that fails
anyway answers ``gpu_memory``; a photo larger than the whole budget answers ``over_budget``; a
budget wait that times out raises :class:`~frame_engine.transport.OverloadError`. The region
bytes one photo holds with calls in flight are bounded by its fair share of the region
(``OP_FRAME_ENGINE_REGION_MIB`` over ``OP_FRAME_ENGINE_WORKERS``, at least one call), so every
worker can hold its share at once.

Time: every model call ends by the request's deadline (:mod:`frame_engine.transport`), and the
request records the stage it is in (``EngineRequest.stage``) for the front's deadline log.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import cupy as cp
import numpy as np

from frame_engine.branches.base import Branch, DependencyFailedError, Photo, PhotoCalls
from frame_engine.branches.crops import CropJpegs
from frame_engine.branches.derivatives import Derivatives
from frame_engine.branches.detect import Detect
from frame_engine.branches.embed import EmbedCrops, EmbedFrame
from frame_engine.branches.ocr import Ocr
from frame_engine.messages import EngineRequest, EngineResult
from frame_engine.nvjpeg_lanes import NvJpegError, NvJpegLanes
from frame_engine.ocr.ctc import CtcDecoder
from frame_engine.ocr.kernels import LineWarp
from frame_engine.prep_budget import ByteBudget, is_memory_error, need_bytes
from frame_engine.prep_geometry import oriented_size, route, stored_box_rects
from frame_engine.prep_kernels import PrepKernels
from frame_engine.slab import SlabTooLargeError
from frame_engine.stage_metrics import StageClock
from frame_engine.transport import ModelCallError, OverloadError
from src.domain import frame_engine_wire as wire


if TYPE_CHECKING:
    from collections.abc import Callable

    from frame_engine.prep_geometry import JpegHeader
    from frame_engine.settings import EngineSettings
    from frame_engine.stage_metrics import EngineMetrics
    from frame_engine.transport import BodyTransport, CudaShmTransport


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


def _ordered_outputs(plan: wire.EnginePlan, out: dict[str, Any]) -> dict[str, Any]:
    """The outputs in the wire module's order, whatever order the branches finished in."""
    names = (wire.SIZE, *wire.expected_outputs(plan))
    return {name: out[name] for name in names if name in out}


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
        self.line_warp = LineWarp()
        self._ocr_decoder: CtcDecoder | None = None
        self._ocr_decoder_lock = threading.Lock()
        self.encode_pool = ThreadPoolExecutor(settings.encode_threads, thread_name_prefix='fe-jpeg')
        self.workers = ThreadPoolExecutor(settings.workers, thread_name_prefix='fe-image')
        #: Region bytes one photo may hold with calls in flight: its worker's fair share.
        self.window = settings.region_mib * MIB // settings.workers
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
        need = need_bytes(header, tuple(r.side for r in plan.derivatives))
        if need > self.budget.total:
            return self._failed(plan, 'over_budget')
        with clock.stage('budget'):
            admitted = self.budget.acquire(need, self.s.budget_wait_seconds)
        self.metrics.budget_waits.labels('admitted' if admitted else 'timeout').inc()
        if not admitted:
            raise OverloadError(
                'budget', f'{need} bytes not free within {self.s.budget_wait_seconds:g} s'
            )
        photo: Photo | None = None
        try:
            with clock.stage('decode'):
                decoded = self._decode(data, header)
            if isinstance(decoded, str):
                return self._failed(plan, decoded)
            # the photo is the frame's one holder: cleared below, never pinned by a traceback
            calls = PhotoCalls(self.transport, clock, request.deadline, self.window, _sync_stream)
            size = oriented_size(header)
            photo = Photo(plan, decoded, size, calls, clock, self.kernels,
                          {wire.SIZE: np.array(size, np.int32)})  # fmt: skip
            del decoded
            return self._branches(request, photo)
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
            if photo is not None and photo.frame is not None:
                cp.cuda.get_current_stream().synchronize()
                photo.frame = None
            self.budget.release(need)

    def _decode(self, data: bytes, header: JpegHeader) -> Any:
        """The oriented frame, or the status that says why there is none."""
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
        if (int(frame.shape[1]), int(frame.shape[0])) != oriented_size(header):
            return 'decoder'
        return frame

    def _decoder(self) -> CtcDecoder:
        """The recogniser's dictionary, read on the first ``ocr`` photo (a missing file fails
        that photo loudly: it is a deployment fault, not a branch outcome)."""
        with self._ocr_decoder_lock:
            if self._ocr_decoder is None:
                self._ocr_decoder = CtcDecoder.from_file(self.s.ocr_dictionary)
            return self._ocr_decoder

    def _plan_branches(self, plan: wire.EnginePlan) -> list[Branch]:
        """The branch modules the plan runs, in plan order (round 2 submits before host work)."""
        out: list[Branch] = []
        if plan.runs('detect') and plan.detector is not None:
            out.append(Detect(plan.detector))
        if plan.embed is not None and plan.runs('embed_frame'):
            out.append(EmbedFrame(plan.embed))
        if plan.embed is not None and plan.runs('embed_crops'):
            out.append(EmbedCrops(plan.embed, plan.embed_classes))
        if plan.ocr is not None and plan.runs('ocr'):
            out.append(Ocr(plan.ocr, self._decoder(), self.line_warp))
        if plan.derivatives:
            out.append(Derivatives(self.encode_pool))
        if plan.crops == 'inline':
            out.append(CropJpegs(self.encode_pool, self.s.jpeg_quality, self.s.crop_max_side))
        return out

    def _branches(self, request: EngineRequest, photo: Photo) -> EngineResult:
        plan = request.plan
        states = self._initial_states(plan)
        calls = photo.calls
        if request.boxes is not None:
            photo.rects = stored_box_rects(request.boxes, *photo.size)
        branches = self._plan_branches(plan)
        failed: set[int] = set()

        def phase(hook: Callable[[Branch], Callable[[Photo], None]]) -> None:
            for k, branch in enumerate(branches):
                if k not in failed:
                    try:
                        hook(branch)(photo)
                    except (ModelCallError, DependencyFailedError) as exc:
                        failed.add(k)
                        self._branch_failed(branch, exc, calls, states)

        try:
            with calls.together():
                phase(lambda b: b.round1)
            phase(lambda b: b.settle)
            phase(lambda b: b.round2)
            phase(lambda b: b.finish)
        finally:
            calls.close()
            for branch in branches:
                branch.close(photo)
        for k, branch in enumerate(branches):
            if k not in failed and branch.status is not None:
                states[branch.status] = ('ok', '')
        return EngineResult(
            wire.STATUS_OK, wire.DECODED_GPU, _ordered(plan, states),
            _ordered_outputs(plan, photo.out),
        )  # fmt: skip

    @staticmethod
    def _branch_failed(
        branch: Branch,
        exc: Exception,
        calls: PhotoCalls,
        states: dict[str, tuple[wire.BranchState, str]],
    ) -> None:
        """One branch failed: its status, its calls cancelled; the others go on."""
        calls.cancel(branch)
        if isinstance(exc, ModelCallError):
            logger.warning('%s: model call failed: %s', type(branch).__name__, exc)
            reason = wire.MODEL_ERROR
        else:
            reason = wire.DEPENDENCY_FAILED
        if branch.status is not None:
            states[branch.status] = ('error', reason)
