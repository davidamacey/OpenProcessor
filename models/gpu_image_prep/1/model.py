"""gpu_image_prep: decode a photo once on the GPU and derive every ingest input from that frame.

One request per image (the encoded bytes). Requests from every API worker meet in this one
process (one CUDA context, one decoder owner) and run on a pool of worker threads, so decoding,
kernels and the model calls of different images overlap; the detector and PE calls (BLS) of
concurrent images are batched by those models' own dynamic batchers. Per image:

1. Route by header (JPEG, 8-bit, 1 or 3 components, baseline, end-of-image marker present, size
   gate); anything else answers with a status and the API decodes it on the CPU, counted.
2. nvJPEG decode (GPU Huffman lanes), EXIF orientation applied on the GPU.
3. Detector letterbox (uint8, Pillow ``BILINEAR``); detector via BLS.
4. The whole-frame square (Pillow ``BILINEAR``, shorter edge to 336, centre crop) and,
   when asked, one square per detection cut from the full-resolution frame (``crop_pil``'s
   rectangle, plain bilinear as ``resize_crop_rgb``); the squares through PE (BLS) in one request.
5. Blur sums of the frame and of every detection's clamped rectangle (exact integers).
6. Every detection's crop as a JPEG (quality 90) for the crop cache, encoded on host threads.

Every request times its stages (``stage_metrics``) and publishes them once as Triton counters.

Only boxes, vectors, sums and the crop JPEGs leave the GPU. Requests beyond ``MAX_IN_FLIGHT``
wait in ``execute``, which holds Triton's queue: back-pressure instead of unbounded memory.

Stored-box mode (the reprocess embed pass, #259): a request that also carries ``boxes`` (stored
``bbox_norm`` rows) skips steps 3, 5 and 6 and embeds the whole-frame square and one square per
stored box, cut by the rectangle the CPU embed pass cuts (``prep_geometry.stored_box_rects``).

Measured on one RTX A6000 with 12-24 MP photos (set B, detect + embed): about 43 photos/s with the
GPU about 75 % busy (PE is most of it); the worker count past 32 and batching requests before the
stages change nothing.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cupy as cp
import cv2
import numpy as np
import triton_python_backend_utils as pb_utils
from nvjpeg_lanes import NvJpegLanes
from prep_geometry import (
    blur_rect,
    crop_rect,
    letterbox_geometry,
    oriented_size,
    route,
    stored_box_rects,
    undo_letterbox,
    whole_square_geometry,
)
from prep_kernels import PrepKernels
from stage_metrics import StageClock, StageMetrics


DET_OUTPUTS = ('num_dets', 'det_boxes', 'det_scores', 'det_classes')


def _orient(frame: Any, orientation: int) -> Any:
    """Re-orient an HWC frame by its EXIF tag as ``ImageOps.exif_transpose`` does."""
    t = frame
    if orientation in (5, 6, 7, 8):
        t = t.transpose(1, 0, 2)
    if orientation in (2, 3, 6, 7):
        t = t[:, ::-1]
    if orientation in (3, 4, 7, 8):
        t = t[::-1]
    return cp.ascontiguousarray(t) if t is not frame else frame


def _tensor(name: str, arr: Any) -> Any:
    return pb_utils.Tensor(name, np.ascontiguousarray(arr))


class _Image:
    __slots__ = (
        'boxes',
        'classes',
        'embed',
        'frame',
        'lb',
        'num',
        'reason',
        'rects',
        'scores',
        'size',
    )

    def __init__(self, reason: str, embed: bool) -> None:
        self.reason = reason
        self.embed = embed
        self.frame: Any = None
        self.num = 0
        self.rects: list[tuple[int, int, int, int]] = []


class TritonPythonModel:
    def initialize(self, args: dict[str, Any]) -> None:
        cfg = json.loads(args['model_config'])
        p = {k: v['string_value'] for k, v in cfg.get('parameters', {}).items()}
        self.det_model = p['DETECTOR_MODEL']
        self.pe_model = p['PE_MODEL']
        self.net = int(p.get('NET_SIZE', '640'))
        self.fill = int(p.get('LETTERBOX_FILL', '114'))
        self.pe_size = int(p.get('PE_SIZE', '336'))
        self.min_mp = float(p.get('MIN_MEGAPIXELS', '0'))
        self.max_mp = float(p.get('MAX_MEGAPIXELS', '100'))
        self.pe_batch = int(p.get('PE_BATCH', '32'))
        self.quality = int(p.get('JPEG_QUALITY', '90'))
        device = int(args['model_instance_device_id'])
        cp.cuda.Device(device).use()
        # CuPy keeps freed blocks for reuse; photo frames differ in size, so without a cap the pool
        # grows past 15 GB. At the cap it frees cached blocks before allocating.
        cp.get_default_memory_pool().set_limit(size=int(p.get('POOL_MIB', '4096')) * 1024 * 1024)
        self.lanes = NvJpegLanes(lanes=int(p.get('LANES', '16')))
        self.kernels = PrepKernels()
        self.encode_pool = ThreadPoolExecutor(
            int(p.get('ENCODE_THREADS', '8')), thread_name_prefix='fe-jpeg'
        )
        self.workers = ThreadPoolExecutor(
            int(p.get('WORKERS', '32')), thread_name_prefix='fe-image'
        )
        self.in_flight = threading.BoundedSemaphore(int(p.get('MAX_IN_FLIGHT', '96')))
        self.device = device
        self.local = threading.local()
        self.metrics = StageMetrics(pb_utils)
        self.stage_sync = p.get('STAGE_SYNC', '0') == '1'

    def execute(self, requests: list[Any]) -> None:
        for r in requests:
            arrived = time.perf_counter()
            self.in_flight.acquire()
            self.workers.submit(self._run, r, arrived)

    def _run(self, request: Any, arrived: float) -> None:
        clock = StageClock(self._sync_stream if self.stage_sync else None)
        clock.add('queue', time.perf_counter() - arrived)
        try:
            stream = getattr(self.local, 'stream', None)
            if stream is None:
                cp.cuda.Device(self.device).use()
                stream = self.local.stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                response = self._process(request, clock)
        except Exception as exc:  # one failed image must not take the model down
            err = pb_utils.TritonError(f'gpu_image_prep: {exc}')
            response = pb_utils.InferenceResponse(output_tensors=[], error=err)
        self.metrics.publish(clock)
        try:
            request.get_response_sender().send(
                response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )
        finally:
            self.in_flight.release()

    # ------------------------------------------------------------------ stages

    @staticmethod
    def _sync_stream() -> None:
        cp.cuda.get_current_stream().synchronize()

    def _decode(self, data: bytes, embed: bool) -> _Image:
        reason, header = route(data, self.min_mp, self.max_mp)
        im = _Image(reason, embed)
        if reason != 'gpu':
            return im
        try:
            frame, _backend = self.lanes.decode(data)
        except Exception:
            im.reason = 'decoder'
            return im
        im.frame = _orient(frame, header.orientation)
        w, h = oriented_size(header)
        if (int(im.frame.shape[1]), int(im.frame.shape[0])) != (w, h):
            im.reason, im.frame = 'decoder', None
            return im
        im.reason = 'ok'
        im.size = (w, h)
        return im

    def _bls(self, model: str, batch: Any, outputs: tuple[str, ...]) -> list[np.ndarray]:
        req = pb_utils.InferenceRequest(
            model_name=model,
            inputs=[pb_utils.Tensor.from_dlpack('images', batch)],
            requested_output_names=list(outputs),
        )
        resp = req.exec()
        if resp.has_error():
            raise pb_utils.TritonModelException(resp.error().message())
        out = []
        for name in outputs:
            t = pb_utils.get_output_tensor_by_name(resp, name)
            out.append(t.as_numpy() if t.is_cpu() else cp.from_dlpack(t.to_dlpack()).get())
        return out

    def _detect(self, im: _Image, clock: StageClock) -> None:
        w, h = im.size
        lb = letterbox_geometry(w, h, self.net)
        im.lb = lb
        with clock.stage('letterbox'):
            canvas = cp.full((1, 3, self.net, self.net), self.fill, cp.uint8)
            dst = canvas[0, :, lb.top : lb.top + lb.new_h, lb.left : lb.left + lb.new_w]
            self.kernels.resample_into(im.frame, (0, 0, w, h), (lb.new_w, lb.new_h), dst)
        with clock.stage('detect'):
            num, boxes, scores, classes = self._bls(self.det_model, canvas, DET_OUTPUTS)
            im.num = int(num[0][0])
            im.boxes = boxes[0][: im.num].astype(np.float32).reshape(-1, 4)
            im.scores = scores[0][: im.num].astype(np.float32)
            im.classes = classes[0][: im.num].astype(np.int32)
            im.rects = [crop_rect(undo_letterbox(tuple(b), self.net, lb), w, h) for b in im.boxes]

    def _embed(self, im: _Image, clock: StageClock) -> np.ndarray:
        """PE vectors of the whole-frame square and, when asked, of every detection's square."""
        t = self.pe_size
        rects = im.rects if im.embed else []
        sq = cp.empty((1 + len(rects), 3, t, t), cp.uint8)
        w, h = im.size
        with clock.stage('embed_frame'):
            g = whole_square_geometry(w, h, t)
            self.kernels.resample_into(
                im.frame, (0, 0, w, h), (g.new_w, g.new_h), sq[0], (g.left, g.top, t, t)
            )
        with clock.stage('embed_crops'):
            for k, (x1, y1, x2, y2) in enumerate(rects, start=1):
                g = whole_square_geometry(x2 - x1, y2 - y1, t)
                self.kernels.resample_into(
                    im.frame,
                    (x1, y1, x2 - x1, y2 - y1),
                    (g.new_w, g.new_h),
                    sq[k],
                    (g.left, g.top, t, t),
                    antialias=False,
                )
        with clock.stage('embed_pe'):
            rows = [
                self._bls(
                    self.pe_model,
                    cp.ascontiguousarray(sq[s : s + self.pe_batch]),
                    ('image_embeddings',),
                )[0]
                for s in range(0, int(sq.shape[0]), self.pe_batch)
            ]
            return np.concatenate(rows, axis=0).astype(np.float32)

    def _blur_sums(self, im: _Image) -> np.ndarray:
        """``[1 + detections, 3]`` (pixels, sum, sum of squares): the frame, then each box."""
        w, h = im.size
        clamped = [blur_rect(undo_letterbox(tuple(b), self.net, im.lb), w, h) for b in im.boxes]
        rects = [(0, 0, w, h)] + [r for r in clamped if r is not None]
        sums = iter(cp.asnumpy(self.kernels.lap_sums(im.frame, rects)))
        rows = [(w * h, *next(sums))]
        rows.extend(
            (0, 0, 0) if r is None else ((r[2] - r[0]) * (r[3] - r[1]), *next(sums))
            for r in clamped
        )
        return np.array(rows, np.int64).reshape(-1, 3)

    def _crop_pixels(self, im: _Image) -> list[np.ndarray]:
        """Each detection's ``crop_pil`` rectangle (inside the frame), BGR (host)."""
        return [
            cp.ascontiguousarray(im.frame[y1:y2, x1:x2, ::-1]).get() for x1, y1, x2, y2 in im.rects
        ]

    def _encode(self, pixels: np.ndarray) -> tuple[bytes, float]:
        """The JPEG and the seconds the encode took on its host thread."""
        start = time.perf_counter()
        ok, buf = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise RuntimeError('crop JPEG encode failed')
        return buf.tobytes(), time.perf_counter() - start

    def _embed_stored(self, im: _Image, boxes: np.ndarray, clock: StageClock) -> Any:
        """Stored-box mode: the whole-frame vector and one vector per stored box, in order."""
        w, h = im.size
        im.rects = stored_box_rects(boxes, w, h)
        im.embed = True
        vectors = self._embed(im, clock)
        im.frame = None
        return pb_utils.InferenceResponse(
            output_tensors=[
                _tensor('status', np.array(['ok'], np.object_)),
                _tensor('size', np.array([w, h], np.int32)),
                _tensor('whole_embedding', vectors[0]),
                _tensor('crop_embeddings', vectors[1:].reshape(-1, vectors.shape[1])),
            ]
        )

    def _process(self, request: Any, clock: StageClock) -> Any:
        data = pb_utils.get_input_tensor_by_name(request, 'jpeg').as_numpy().tobytes()
        flag = pb_utils.get_input_tensor_by_name(request, 'embed_crops')
        stored = pb_utils.get_input_tensor_by_name(request, 'boxes')
        with clock.stage('decode'):
            im = self._decode(data, True if flag is None else bool(flag.as_numpy().reshape(-1)[0]))
        if im.reason != 'ok':
            return pb_utils.InferenceResponse(
                output_tensors=[_tensor('status', np.array([im.reason], np.object_))]
            )
        if stored is not None:
            return self._embed_stored(im, stored.as_numpy(), clock)
        self._detect(im, clock)
        with clock.stage('crop_copy'):
            pixels = self._crop_pixels(im)
        jpegs = [self.encode_pool.submit(self._encode, px) for px in pixels]
        vectors = self._embed(im, clock)
        with clock.stage('blur'):
            sums = self._blur_sums(im)
        im.frame = None
        done = [f.result() for f in jpegs]
        encoded = [data for data, _ in done]
        clock.add('crop_jpeg', sum(secs for _, secs in done))
        w, h = im.size
        return pb_utils.InferenceResponse(
            output_tensors=[
                _tensor('status', np.array(['ok'], np.object_)),
                _tensor('size', np.array([w, h], np.int32)),
                _tensor('letterbox', np.array([im.lb.scale, im.lb.pad_w, im.lb.pad_h], np.float64)),
                _tensor('det_boxes', im.boxes),
                _tensor('det_scores', im.scores),
                _tensor('det_classes', im.classes),
                _tensor('whole_embedding', vectors[0]),
                _tensor('crop_embeddings', vectors[1:].reshape(-1, vectors.shape[1])),
                _tensor('blur_sums', sums),
                _tensor('crop_jpegs', np.array(encoded, dtype=np.object_)),
            ]
        )
