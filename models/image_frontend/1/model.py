"""image_frontend: decode a photo once on the GPU and derive every ingest input from that frame.

One request per image (the encoded bytes). Requests from every API worker meet in this one
process (one CUDA context, one decoder owner) and are grouped into batches; a few batches run at
once on worker threads so decoding, kernels and the model calls of different batches overlap. Per
batch:

1. Route by header (JPEG, 8-bit, 1 or 3 components, baseline, end-of-image marker present, size
   gate); anything else answers with a status and the API decodes it on the CPU, counted.
2. nvJPEG decode (GPU Huffman lanes), EXIF orientation applied on the GPU.
3. Detector letterbox (uint8, Pillow ``BILINEAR``) written into one batch canvas; detector via BLS.
4. Per image: the whole-frame square (Pillow ``BILINEAR``, shorter edge to 336, centre crop) and,
   when asked, one square per detection cut from the full-resolution frame (``crop_pil``'s
   rectangle, plain bilinear as ``resize_crop_rgb``); every square in one batch through PE (BLS).
5. Blur sums of the frame and of every detection's clamped rectangle (exact integers).
6. Every detection's crop as a JPEG (quality 90) for the crop cache, encoded on host threads.

Only boxes, vectors, sums and the crop JPEGs leave the GPU. Requests beyond ``MAX_IN_FLIGHT``
wait in ``execute``, which holds Triton's queue: back-pressure instead of unbounded memory.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cupy as cp
import cv2
import numpy as np
import triton_python_backend_utils as pb_utils
from frontend_geometry import (
    blur_rect,
    crop_rect,
    letterbox_geometry,
    oriented_size,
    route,
    undo_letterbox,
    whole_square_geometry,
)
from frontend_kernels import FrontendKernels
from nvjpeg_lanes import NvJpegLanes


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
    __slots__ = ('boxes', 'classes', 'embed', 'frame', 'jpegs', 'lb', 'num', 'reason', 'rects', 'scores', 'size')

    def __init__(self, reason: str, embed: bool) -> None:
        self.reason = reason
        self.embed = embed
        self.frame: Any = None
        self.num = 0
        self.rects: list[tuple[int, int, int, int]] = []
        self.jpegs: list[bytes] = []


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
        self.det_batch = int(p.get('DET_BATCH', '16'))
        self.pe_batch = int(p.get('PE_BATCH', '32'))
        self.max_batch = int(p.get('MAX_BATCH', '16'))
        self.wait_s = float(p.get('BATCH_WAIT_MS', '5')) / 1000
        self.quality = int(p.get('JPEG_QUALITY', '90'))
        device = int(args['model_instance_device_id'])
        cp.cuda.Device(device).use()
        lanes = int(p.get('LANES', '8'))
        self.lanes = NvJpegLanes(lanes=lanes)
        self.kernels = FrontendKernels()
        self.decode_pool = ThreadPoolExecutor(lanes, thread_name_prefix='fe-decode')
        self.encode_pool = ThreadPoolExecutor(int(p.get('ENCODE_THREADS', '8')), thread_name_prefix='fe-jpeg')
        self.batch_pool = ThreadPoolExecutor(int(p.get('BATCH_WORKERS', '3')), thread_name_prefix='fe-batch')
        self.in_flight = threading.BoundedSemaphore(int(p.get('MAX_IN_FLIGHT', '96')))
        self.inbox: queue.Queue[Any] = queue.Queue()
        self.device = device
        threading.Thread(target=self._dispatch, name='fe-dispatch', daemon=True).start()

    def execute(self, requests: list[Any]) -> None:
        for r in requests:
            self.in_flight.acquire()
            self.inbox.put(r)

    def _dispatch(self) -> None:
        while True:
            batch = [self.inbox.get()]
            deadline = time.perf_counter() + self.wait_s
            while len(batch) < self.max_batch:
                left = deadline - time.perf_counter()
                if left <= 0:
                    break
                try:
                    batch.append(self.inbox.get(timeout=left))
                except queue.Empty:
                    break
            self.batch_pool.submit(self._run, batch)

    def _run(self, batch: list[Any]) -> None:
        try:
            cp.cuda.Device(self.device).use()
            with cp.cuda.Stream(non_blocking=True):
                responses = self._process(batch)
        except Exception as exc:  # one failed batch must not take the model down
            err = pb_utils.TritonError(f'image_frontend: {exc}')
            responses = [pb_utils.InferenceResponse(output_tensors=[], error=err) for _ in batch]
        for r, resp in zip(batch, responses, strict=True):
            try:
                r.get_response_sender().send(resp, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            finally:
                self.in_flight.release()

    # ------------------------------------------------------------------ stages

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

    def _detect(self, ims: list[_Image]) -> None:
        canvas = cp.full((len(ims), 3, self.net, self.net), self.fill, cp.uint8)
        for j, im in enumerate(ims):
            w, h = im.size
            lb = letterbox_geometry(w, h, self.net)
            im.lb = lb
            dst = canvas[j, :, lb.top : lb.top + lb.new_h, lb.left : lb.left + lb.new_w]
            self.kernels.resample_into(im.frame, (0, 0, w, h), (lb.new_w, lb.new_h), dst)
        for s in range(0, len(ims), self.det_batch):
            chunk = cp.ascontiguousarray(canvas[s : s + self.det_batch])
            num, boxes, scores, classes = self._bls(self.det_model, chunk, DET_OUTPUTS)
            for j, im in enumerate(ims[s : s + self.det_batch]):
                im.num = int(num[j][0])
                im.boxes = boxes[j][: im.num].astype(np.float32)
                im.scores = scores[j][: im.num].astype(np.float32)
                im.classes = classes[j][: im.num].astype(np.int32)
                im.rects = [
                    crop_rect(undo_letterbox(tuple(b), self.net, im.lb)) for b in im.boxes
                ]

    def _squares(self, ims: list[_Image]) -> Any:
        """One uint8 batch: per image its whole square, then its detection squares if asked."""
        n = sum(1 + (im.num if im.embed else 0) for im in ims)
        sq = cp.empty((n, 3, self.pe_size, self.pe_size), cp.uint8)
        k, t = 0, self.pe_size
        for im in ims:
            w, h = im.size
            g = whole_square_geometry(w, h, t)
            self.kernels.resample_into(im.frame, (0, 0, w, h), (g.new_w, g.new_h), sq[k], (g.left, g.top, t, t))
            k += 1
            if not im.embed:
                continue
            for x1, y1, x2, y2 in im.rects:
                cw, ch = x2 - x1, y2 - y1
                g = whole_square_geometry(cw, ch, t)
                self.kernels.resample_into(
                    im.frame, (x1, y1, cw, ch), (g.new_w, g.new_h), sq[k], (g.left, g.top, t, t), antialias=False
                )
                k += 1
        return sq

    def _embed(self, squares: Any) -> np.ndarray:
        rows = [
            self._bls(self.pe_model, cp.ascontiguousarray(squares[s : s + self.pe_batch]), ('image_embeddings',))[0]
            for s in range(0, int(squares.shape[0]), self.pe_batch)
        ]
        return np.concatenate(rows, axis=0).astype(np.float32)

    def _blur(self, im: _Image) -> tuple[Any, list[tuple[int, int, int, int] | None]]:
        w, h = im.size
        clamped = [blur_rect(undo_letterbox(tuple(b), self.net, im.lb), w, h) for b in im.boxes]
        rects = [(0, 0, w, h)] + [r for r in clamped if r is not None]
        return self.kernels.lap_sums(im.frame, rects), clamped

    def _crop_pixels(self, im: _Image) -> list[np.ndarray]:
        """Each detection's ``crop_pil`` rectangle, BGR, zeros outside the frame (host)."""
        w, h = im.size
        out = []
        for x1, y1, x2, y2 in im.rects:
            sx2, sy2 = min(x2, w), min(y2, h)
            inside = cp.ascontiguousarray(im.frame[y1:sy2, x1:sx2, ::-1]) if sx2 > x1 and sy2 > y1 else None
            host = np.zeros((y2 - y1, x2 - x1, 3), np.uint8)
            if inside is not None:
                host[: sy2 - y1, : sx2 - x1] = inside.get()
            out.append(host)
        return out

    def _encode(self, pixels: np.ndarray) -> bytes:
        ok, buf = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise RuntimeError('crop JPEG encode failed')
        return buf.tobytes()

    def _process(self, requests: list[Any]) -> list[Any]:
        datas, embeds = [], []
        for r in requests:
            datas.append(pb_utils.get_input_tensor_by_name(r, 'jpeg').as_numpy().tobytes())
            flag = pb_utils.get_input_tensor_by_name(r, 'embed_crops')
            embeds.append(True if flag is None else bool(flag.as_numpy().reshape(-1)[0]))
        ims = list(self.decode_pool.map(self._decode, datas, embeds))
        ok = [im for im in ims if im.reason == 'ok']
        vectors: np.ndarray | None = None
        blur: dict[int, tuple[np.ndarray, list[Any]]] = {}
        if ok:
            self._detect(ok)
            vectors = self._embed(self._squares(ok))
            sums = {id(im): self._blur(im) for im in ok}
            jpeg_jobs = {id(im): [self.encode_pool.submit(self._encode, px) for px in self._crop_pixels(im)] for im in ok}
            for im in ok:
                dev, clamped = sums[id(im)]
                blur[id(im)] = (cp.asnumpy(dev), clamped)
                im.frame = None
            for im in ok:
                im.jpegs = [f.result() for f in jpeg_jobs[id(im)]]
        responses = []
        k = 0
        for im in ims:
            if im.reason != 'ok':
                responses.append(pb_utils.InferenceResponse(output_tensors=[_tensor('status', np.array([im.reason], np.object_))]))
                continue
            host, clamped = blur[id(im)]
            it = iter(host[1:])
            w, h = im.size
            box_sums = [
                (0, 0, 0) if r is None else ((r[2] - r[0]) * (r[3] - r[1]), *next(it)) for r in clamped
            ]
            sums = np.array([(w * h, *host[0]), *box_sums], np.int64).reshape(-1, 3)
            n_vec = 1 + (im.num if im.embed else 0)
            assert vectors is not None
            vec = vectors[k : k + n_vec]
            k += n_vec
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        _tensor('status', np.array(['ok'], np.object_)),
                        _tensor('size', np.array([w, h], np.int32)),
                        _tensor('letterbox', np.array([im.lb.scale, im.lb.pad_w, im.lb.pad_h], np.float64)),
                        _tensor('det_boxes', im.boxes.reshape(-1, 4)),
                        _tensor('det_scores', im.scores),
                        _tensor('det_classes', im.classes),
                        _tensor('whole_embedding', vec[0]),
                        _tensor('crop_embeddings', vec[1:].reshape(-1, vec.shape[1])),
                        _tensor('blur_sums', sums),
                        _tensor('crop_jpegs', np.array(im.jpegs, dtype=np.object_)),
                    ]
                )
            )
        return responses
