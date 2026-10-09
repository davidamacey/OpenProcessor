"""OCR pipeline reader: framing, response parsing and text-driven regions."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from tritonclient.grpc import InferInput, InferRequestedOutput

from src.services.detection.cascade_detect.preprocess import _decode_jpeg
from src.services.detection.region_text import OcrLine


if TYPE_CHECKING:
    from src.clients.triton_pool import AsyncTritonPool
    from src.config import DetectionProfile


logger = logging.getLogger(__name__)


# =============================================================================
# PaddleOcrTextRecognizer — text reader / text-driven detection
# =============================================================================


@dataclass
class OcrRegion:
    """One text region detected + recognized by an OCR pipeline model.

    ``profile`` carries the aspect/length/score thresholds this
    region's shape-and-text checks are evaluated against — defaults to
    Required -- no example profile is substituted silently.
    """

    bbox_norm: tuple[float, float, float, float]
    text: str  # canonicalized (uppercase, alphanumeric + space/dash)
    text_raw: str  # exact OCR string (may include unicode / punctuation)
    det_score: float
    rec_score: float
    profile: DetectionProfile

    @property
    def is_region_shaped(self) -> bool:
        """Aspect-ratio test used by the general region sanity gate."""
        x1, y1, x2, y2 = self.bbox_norm
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if h <= 0.0 or w <= 0.0:
            return False
        ar = w / h
        return self.profile.aspect_min <= ar <= self.profile.aspect_max

    @property
    def is_region_text_candidate(self) -> bool:
        """Stricter test for text-hint detection promotion.

        The text must sit in the profile's tighter text-hint aspect range,
        read at or above ``text_hint_rec_floor``, match ``text_pattern`` and
        have a length within ``text_hint_len_min..text_hint_len_max``.

        With ``text_hint_require_letters_and_digits`` on, it must also mix
        letters and digits -- for a region whose text always does, this
        rejects lettering-only or number-only text elsewhere on the item
        (and, as accepted false negatives, a region text that happens to be
        all letters or all digits). Off by default.
        """
        x1, y1, x2, y2 = self.bbox_norm
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if h <= 0.0 or w <= 0.0:
            return False
        ar = w / h
        p = self.profile
        if not (p.text_hint_aspect_min <= ar <= p.text_hint_aspect_max):
            return False
        if self.rec_score < p.text_hint_rec_floor:
            return False
        pattern = re.compile(p.text_pattern)
        if not pattern.fullmatch(self.text):
            return False
        compact = self.text.replace(' ', '').replace('-', '')
        if not (p.text_hint_len_min <= len(compact) <= p.text_hint_len_max):
            return False
        if not p.text_hint_require_letters_and_digits:
            return True
        has_letter = any(c.isalpha() for c in compact)
        has_digit = any(c.isdigit() for c in compact)
        return has_letter and has_digit


def _canonicalize_text(s: str) -> str:
    """Uppercase, strip non-[A-Z0-9 -], collapse runs of whitespace.

    Drops punctuation so a single canonical form can be compared across
    OCR engines, but preserves the raw string in ``OcrRegion.text_raw``
    for debugging non-ASCII text.
    """
    up = s.upper()
    kept = [c if (c.isalnum() or c in ' -') else ' ' for c in up]
    return ' '.join(''.join(kept).split())


# Side-length window the OCR text-detection engine accepts. The shipped
# export (scripts/export_paddleocr.sh) builds up to 960; some TensorRT
# builds reject sides below 320, so a wide, short crop (a region crop)
# must be scaled up on its short side rather than sent at e.g. 640x256.
OCR_DET_MIN_SIDE = 320
OCR_DET_MAX_SIDE = 960
OCR_DET_TARGET_LONG_SIDE = 640


def ocr_det_input_size(w: int, h: int) -> tuple[int, int]:
    """``(width, height)`` for the OCR detection input of a ``w`` x ``h`` crop.

    Long side scaled to :data:`OCR_DET_TARGET_LONG_SIDE`, raised so the
    short side reaches :data:`OCR_DET_MIN_SIDE`, each side rounded to a
    multiple of 32 and clamped to the engine window. Clamping stretches
    only extreme aspect ratios; that is safe because the OCR pipeline
    maps detection boxes back per axis (``orig / det`` for x and y
    separately).
    """
    scale = OCR_DET_TARGET_LONG_SIDE / max(w, h)
    if min(w, h) * scale < OCR_DET_MIN_SIDE:
        scale = OCR_DET_MIN_SIDE / min(w, h)

    def _side(v: int) -> int:
        r = round(v * scale / 32) * 32
        return max(OCR_DET_MIN_SIDE, min(OCR_DET_MAX_SIDE, r))

    return _side(w), _side(h)


# Neutral fill around a framed region crop (same gray the detectors
# letterbox with).
_OCR_FRAME_FILL = (114, 114, 114)
# Free canvas border around the framed crop, in pixels, so text touching
# the crop edge still has context for the detector's box expansion.
_OCR_FRAME_PAD = 32


def frame_for_ocr(
    img: Image.Image, *, min_height: int
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Upscale a small crop and center it on a detector-sized canvas.

    Returns ``(canvas, (x, y, w, h))`` -- where the scaled crop sits on
    the canvas, in canvas pixels. The canvas sides are multiples of 32
    within ``[OCR_DET_MIN_SIDE, OCR_DET_MAX_SIDE]`` so it is sent to the
    detector at native size (no second resize).
    """
    w, h = img.size
    scale = max(1.0, min_height / max(h, 1))
    max_w = OCR_DET_MAX_SIDE - 2 * _OCR_FRAME_PAD
    scale = min(scale, max_w / max(w, 1), max_w / max(h, 1))
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    scaled = img.resize((sw, sh), Image.Resampling.BICUBIC) if (sw, sh) != (w, h) else img

    def _side(v: int) -> int:
        padded = -(-(v + 2 * _OCR_FRAME_PAD) // 32) * 32
        return max(OCR_DET_MIN_SIDE, min(OCR_DET_MAX_SIDE, padded))

    cw, ch = _side(sw), _side(sh)
    canvas = Image.new('RGB', (cw, ch), _OCR_FRAME_FILL)
    ox, oy = (cw - sw) // 2, (ch - sh) // 2
    canvas.paste(scaled, (ox, oy))
    return canvas, (ox, oy, sw, sh)


def _unframe_line(
    line: OcrLine, canvas_size: tuple[int, int], placement: tuple[int, int, int, int]
) -> OcrLine | None:
    """Map a canvas-normalized line box back to the framed crop's frame."""
    cw, ch = canvas_size
    ox, oy, sw, sh = placement
    x1, y1, x2, y2 = line.box
    bx = (
        max(0.0, min(1.0, (x1 * cw - ox) / sw)),
        max(0.0, min(1.0, (y1 * ch - oy) / sh)),
        max(0.0, min(1.0, (x2 * cw - ox) / sw)),
        max(0.0, min(1.0, (y2 * ch - oy) / sh)),
    )
    if bx[2] <= bx[0] or bx[3] <= bx[1]:
        return None
    return OcrLine(text=line.text, box=bx, score=line.score, det_score=line.det_score)


def _parse_ocr_pipeline_result(result: Any) -> list[OcrLine]:
    """Decode the OCR BLS response into :class:`OcrLine` objects.

    Degenerate boxes and empty strings are dropped; boxes are clamped to
    ``[0, 1]``.
    """
    num_arr = result.as_numpy('num_texts')
    if num_arr is None:
        return []
    n = int(num_arr.flatten()[0])
    if n <= 0:
        return []
    boxes = result.as_numpy('text_boxes_normalized')
    texts = result.as_numpy('texts')
    det_scores = result.as_numpy('text_scores')
    rec_scores = result.as_numpy('rec_scores')
    if boxes is None or texts is None or det_scores is None or rec_scores is None:
        return []
    lines: list[OcrLine] = []
    for i in range(min(n, len(texts))):
        raw_bytes = texts[i]
        raw = (
            raw_bytes.decode('utf-8', errors='replace')
            if isinstance(raw_bytes, bytes)
            else str(raw_bytes)
        )
        if not raw.strip():
            continue
        # The BLS emits '' + -1.0 together for a failed recognition (see
        # models/ocr_pipeline/1/model.py), so `raw.strip()` above already
        # drops every sentinel line in practice. This is a defense-in-depth
        # guard in case a future BLS build ever pairs -1.0 with non-empty
        # text: the -1.0 sentinel must never reach an OcrLine.score.
        rec_score = float(rec_scores[i])
        if rec_score < 0.0:
            continue
        x1, y1, x2, y2 = (max(0.0, min(1.0, float(v))) for v in boxes[i])
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(
            OcrLine(
                text=raw,
                box=(x1, y1, x2, y2),
                score=rec_score,
                det_score=float(det_scores[i]),
            )
        )
    return lines


class PaddleOcrTextRecognizer:
    """OCR wrapper around a Triton Python BLS OCR pipeline model.

    Used for two distinct jobs:

    * **Region-text reader** (``read_region_text``): given an already-
      cropped region JPEG, return the concatenated recognized text and
      confidence. Backstop / cross-check for the VLM's verify-and-read.
    * **Text-driven detector** (``detect_regions``): given a full item
      crop, return every text region with a plausible region-shaped
      string. Powers the text-hint path — when the primary + secondary
      detectors all miss but the VLM said the region is visible, the
      OCR pipeline finds it by looking for text.
    """

    def __init__(
        self,
        triton_pool: AsyncTritonPool,
        profile: DetectionProfile,
        *,
        model_name: str | None = None,
        rec_score_floor: float = 0.5,
        det_score_floor: float = 0.5,
    ) -> None:
        self.profile = profile
        self.triton_pool = triton_pool
        self.model_name = model_name or profile.ocr_pipeline_model
        self.rec_score_floor = rec_score_floor
        self.det_score_floor = det_score_floor

    async def read_lines(self, crop_jpeg: bytes) -> list[OcrLine]:
        """Every line the OCR pipeline detected + recognized, unfiltered.

        Boxes are axis-aligned and normalized to the crop; ``text`` is the
        recognizer's exact string. An undecodable crop yields ``[]``; a
        Triton failure raises (callers decide whether a failed read is
        "no text" or "retry later").
        """
        try:
            img = _decode_jpeg(crop_jpeg)
        except ValueError:
            return []
        return await self._infer_lines(img)

    async def read_region_lines(self, region_jpeg: bytes, *, min_height: int) -> list[OcrLine]:
        """Like :meth:`read_lines`, for a small region crop.

        Region crops are often a few dozen pixels tall. Stretched straight
        to the detector's input size the text blurs past what the text
        detector finds, so the crop is instead upscaled to ``min_height``
        pixels tall (never downscaled) and centered on a neutral canvas of
        at least the detector's minimum side (see :func:`frame_for_ocr`).
        Boxes come back normalized to the region crop.
        """
        try:
            img = _decode_jpeg(region_jpeg)
        except ValueError:
            return []
        canvas, placement = frame_for_ocr(img, min_height=min_height)
        lines = await self._infer_lines(canvas, det_size=canvas.size)
        return [ln for ln in (_unframe_line(ln, canvas.size, placement) for ln in lines) if ln]

    async def _infer_lines(
        self, img: Image.Image, det_size: tuple[int, int] | None = None
    ) -> list[OcrLine]:
        try:
            ocr_in, orig_in, orig_shape = self._preprocess(img, det_size)
        except ValueError:
            return []

        inputs = [
            InferInput('ocr_images', list(ocr_in.shape), 'FP32'),
            InferInput('original_image', list(orig_in.shape), 'FP32'),
            InferInput('orig_shape', list(orig_shape.shape), 'INT32'),
        ]
        inputs[0].set_data_from_numpy(ocr_in)
        inputs[1].set_data_from_numpy(orig_in)
        inputs[2].set_data_from_numpy(orig_shape)
        outputs = [
            InferRequestedOutput('num_texts'),
            InferRequestedOutput('text_boxes_normalized'),
            InferRequestedOutput('texts'),
            InferRequestedOutput('text_scores'),
            InferRequestedOutput('rec_scores'),
        ]
        result = await self.triton_pool.infer(self.model_name, inputs, outputs=outputs)
        return _parse_ocr_pipeline_result(result)

    def regions_from_lines(self, lines: list[OcrLine]) -> list[OcrRegion]:
        """Apply this recognizer's score floors + canonicalization to raw
        lines (what :meth:`detect_regions` returns for the same crop)."""
        regions: list[OcrRegion] = []
        for ln in lines:
            canon = _canonicalize_text(ln.text)
            if not canon:
                continue
            if ln.det_score < self.det_score_floor or ln.score < self.rec_score_floor:
                continue
            regions.append(
                OcrRegion(
                    bbox_norm=ln.box,
                    text=canon,
                    text_raw=ln.text,
                    det_score=ln.det_score,
                    rec_score=ln.score,
                    profile=self.profile,
                )
            )
        return regions

    async def detect_regions(self, crop_jpeg: bytes) -> list[OcrRegion]:
        """Return every text region in the crop, region-shaped or not.

        Callers filter for region-shape and region-text-regex; this
        just runs the pipeline and parses the response.
        """
        try:
            lines = await self.read_lines(crop_jpeg)
        except Exception:
            logger.exception('ocr_pipeline_infer_failed')
            return []
        return self.regions_from_lines(lines)

    async def read_region_text(self, region_jpeg: bytes) -> tuple[str, float] | None:
        """Concatenate every recognized line on an already-cropped region.

        Returns ``(canonical_text, mean_rec_score)`` or ``None`` when
        the pipeline produced nothing usable. Text can span multiple
        lines (e.g. a motorcycle "MOTO" + number, an EU province +
        main); we join with a single space in reading order from the
        pipeline.
        """
        regions = await self.detect_regions(region_jpeg)
        if not regions:
            return None
        # Sort top-to-bottom, left-to-right by bbox center.
        regions.sort(key=lambda r: (r.bbox_norm[1] + r.bbox_norm[3], r.bbox_norm[0]))
        text = ' '.join(r.text for r in regions if r.text).strip()
        if not text:
            return None
        avg_rec = sum(r.rec_score for r in regions) / len(regions)
        return text, avg_rec

    def pick_best_text_region(self, regions: list[OcrRegion]) -> OcrRegion | None:
        """Pick a region good enough to promote as a text-hint candidate.

        Uses ``OcrRegion.is_region_text_candidate`` (tighter than the
        general sanity gate): tightened aspect range, length window, OCR
        confidence floor and, when the profile asks for it, a
        letters-and-digits mix. Drops unrelated text elsewhere on the item
        that would otherwise sneak through the loose region-text regex.
        The text-hint path also only fires for crops the VLM already said
        contain the region of interest, so the filter need only reject
        obvious-non-region text — borderline cases route to the VLM
        verify anyway, which is the final gate.

        Returns the largest qualifying region, with rec_score as the
        tie-break (higher-confidence read wins when two regions are
        similar in area).
        """
        region_like = [r for r in regions if r.is_region_text_candidate]
        if not region_like:
            return None

        def _key(r: OcrRegion) -> tuple[float, float]:
            x1, y1, x2, y2 = r.bbox_norm
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            return (area, r.rec_score)

        return max(region_like, key=_key)

    def _preprocess(
        self, img: Image.Image, det_size: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the three input tensors the OCR pipeline model expects.

        * ``ocr_images``: detection-network input. BGR, scaled to a
          multiple-of-32 size from :func:`ocr_det_input_size`,
          normalized to ``[-1, 1]``.
        * ``original_image``: full-resolution **BGR** normalized to
          ``[0, 1]``. The BLS (``models*/ocr_pipeline/1/model.py``)
          reads this tensor straight into HWC uint8 and perspective-warps
          text crops from it with no channel swap; the recognition
          network's preprocessing (``_resize_norm_img*``) documents its
          input as BGR and never converts, so this tensor must already be
          BGR on arrival (matches every other OCR caller — /ocr/*,
          /analyze, generic ingest — which all decode via
          ``cv2.imdecode`` and send the resulting BGR array unchanged).
          DF4: this used to be sent RGB (PIL decodes RGB, and no swap was
          applied), which channel-swapped every worker recognition crop.
        * ``orig_shape``: ``[H, W]`` of ``original_image``.
        """
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError('degenerate crop size')
        # The BLS's tensor shapes per Triton config.pbtxt are
        # un-batched C-H-W (original_image and ocr_images as
        # [3, H, W], orig_shape as [2]). The model is a Python BLS
        # that does its own internal batching from the C-H-W input,
        # so the worker MUST NOT add a leading batch axis here.
        # `img` is a PIL image (RGB); flip to BGR to match the BLS's
        # expected channel order (see docstring above).
        orig_bgr = np.asarray(img, dtype=np.float32)[:, :, ::-1] / 255.0
        orig_chw = np.transpose(orig_bgr, (2, 0, 1))
        orig_shape = np.asarray([h, w], dtype=np.int32)

        new_w, new_h = det_size or ocr_det_input_size(w, h)
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)[:, :, ::-1]  # BGR
        arr = arr / 127.5 - 1.0
        ocr_chw = np.transpose(arr, (2, 0, 1))
        return (
            np.ascontiguousarray(ocr_chw, dtype=np.float32),
            np.ascontiguousarray(orig_chw, dtype=np.float32),
            orig_shape,
        )
