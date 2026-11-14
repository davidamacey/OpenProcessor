"""``ocr``: the detector's resized frame (round 1), text boxes from its map (settle), lines cut from
the full-resolution frame and recognised in batches (round 2), the decoded lines (finish).

* The detector input is the frame resized as PaddleX ``DetResizeForTest`` does
  (``ocr.det_input.det_size``), written as uint8 BGR into the staged slice; the model normalises
  (``export/export_paddleocr_det.py --uint8``) and answers the probability map ``[1, 1, H, W]``.
* The boxes are PaddleOCR's DB post-process on the host (``ocr.db_boxes``), sorted in reading
  order; past ``max_lines`` the lowest detector scores are dropped and counted in ``ocr_overflow``.
* Each line is warped from the frame to 48 pixels high straight into its slot of a recogniser
  batch (``ocr.kernels``, ``ocr.crop_geometry``); batches are PaddleOCR's recipe (``ocr.batching``);
  every batch is its own bounded slice and all are submitted before any is collected.
* The recogniser answers ``indices`` and ``scores`` ``[B, T]`` (arg-max in its graph); the host
  collapses them (``ocr.ctc``). A photo with no text costs the detector call and nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from frame_engine.branches.base import Branch, Call, Photo
from frame_engine.ocr.batching import REC_HEIGHT, plan_batches, resized_width
from frame_engine.ocr.crop_geometry import rec_map, rec_ratio, taps_for
from frame_engine.ocr.db_boxes import DbParams, boxes_from_map, sort_reading_order
from frame_engine.ocr.det_input import det_size
from frame_engine.transport import ModelInput
from src.domain import frame_engine_wire as wire


if TYPE_CHECKING:
    from frame_engine.ocr.ctc import CtcDecoder
    from frame_engine.ocr.kernels import LineWarp


DET_INPUT = 'x'
DET_OUTPUT = 'fetch_name_0'
REC_INPUT = 'x'
REC_INDICES = 'indices'
REC_SCORES = 'scores'
#: The widest recogniser input of the build's optimisation profile (``export_paddleocr_rec.py``).
REC_MAX_WIDTH = 2048


class Ocr(Branch):
    status = 'ocr'

    def __init__(self, spec: wire.OcrSpec, decoder: CtcDecoder, warp: LineWarp) -> None:
        self.spec = spec
        self.decoder = decoder
        self.warp = warp
        self.det_call: Call | None = None
        self.size: tuple[int, int] | None = None  # detector input (height, width)
        self.quads = np.empty((0, 4, 2), np.int32)
        self.det_scores = np.empty(0, np.float32)
        self.overflow = 0
        self.batches: list[tuple[list[int], Call]] = []

    def round1(self, photo: Photo) -> None:
        w, h = photo.size
        s = self.spec
        rh, rw = self.size = det_size(h, w, s.det_side, s.det_limit, s.det_max_side)
        tensor = ModelInput(DET_INPUT, 'UINT8', (1, 3, rh, rw))
        staged = photo.calls.stage(tensor.nbytes, 'ocr_prep')
        with photo.clock.stage('ocr_prep'):
            canvas = staged.device.reshape(tensor.shape)
            # the frame is RGB, PP-OCR reads BGR: write the planes in reverse order
            photo.kernels.resample_into(
                photo.frame, (0, 0, w, h), (rw, rh), canvas[0, ::-1], antialias=s.det_antialias
            )
        self.det_call = photo.calls.submit(
            self, 'ocr_detect', staged, s.detector, tensor, (DET_OUTPUT,)
        )

    def settle(self, photo: Photo) -> None:
        if self.det_call is None:
            raise RuntimeError('ocr settled before round 1')
        pred = photo.calls.result(self.det_call)[DET_OUTPUT][0, 0]
        w, h = photo.size
        s = self.spec
        with photo.clock.stage('ocr_boxes'):
            boxes, scores = boxes_from_map(
                pred, w, h, DbParams(s.thresh, s.box_thresh, s.unclip_ratio)
            )
            keep = [k for k in range(len(boxes)) if rec_ratio(boxes[k]) is not None]
            boxes, scores = boxes[keep], scores[keep]
            if len(boxes) > s.max_lines:
                self.overflow = len(boxes) - s.max_lines
                top = np.sort(np.argsort(-scores, kind='stable')[: s.max_lines])
                boxes, scores = boxes[top], scores[top]
            order = sort_reading_order(boxes)
            self.quads, self.det_scores = boxes[order], scores[order]

    def round2(self, photo: Photo) -> None:
        s = self.spec
        ratios = [rec_ratio(q) or 0.0 for q in self.quads]
        for batch in plan_batches(ratios, s.batch, max_width=REC_MAX_WIDTH):
            n, width = len(batch.indices), batch.width
            maps = np.empty((n, 9), np.float32)
            widths = np.empty(n, np.int32)
            taps = np.empty((n, 2), np.int32)
            with photo.clock.stage('ocr_crops'):
                for k, i in enumerate(batch.indices):
                    widths[k] = resized_width(ratios[i], width)
                    g = rec_map(self.quads[i], int(widths[k]))
                    maps[k] = g.reshape(9)
                    taps[k] = taps_for(g, int(widths[k]), s.taps)
            tensor = ModelInput(REC_INPUT, 'UINT8', (n, 3, REC_HEIGHT, width))
            staged = photo.calls.stage(tensor.nbytes, 'ocr_crops')
            with photo.clock.stage('ocr_crops'):
                self.warp.warp_into(
                    photo.frame, maps, widths, taps, staged.device.reshape(tensor.shape)
                )
            call = photo.calls.submit(
                self, 'ocr_rec', staged, s.recogniser, tensor, (REC_INDICES, REC_SCORES)
            )
            self.batches.append((batch.indices, call))

    def finish(self, photo: Photo) -> None:
        n = len(self.quads)
        texts: list[bytes] = [b''] * n
        conf = np.zeros(n, np.float32)
        low = np.zeros(n, np.float32)
        for indices, call in self.batches:
            answer = photo.calls.result(call)
            with photo.clock.stage('ocr_decode'):
                for k, i in enumerate(indices):
                    text, mean, lowest = self.decoder.decode(
                        answer[REC_INDICES][k], answer[REC_SCORES][k]
                    )
                    texts[i], conf[i], low[i] = text.encode('utf-8'), mean, lowest
        photo.out[wire.OCR_QUADS] = self.quads.reshape(n, 8).astype(np.float32)
        photo.out[wire.OCR_TEXTS] = texts
        photo.out[wire.OCR_SCORES] = conf
        photo.out[wire.OCR_CHAR_MIN] = low
        photo.out[wire.OCR_DET_SCORES] = self.det_scores
        photo.out[wire.OCR_OVERFLOW] = np.array([self.overflow], np.int32)
