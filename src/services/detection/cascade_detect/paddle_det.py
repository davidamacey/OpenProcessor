"""PaddleOCR DBNet last-resort region detector."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from tritonclient.grpc import InferInput, InferRequestedOutput

from src.services.detection.cascade_detect.candidate import RegionCandidate
from src.services.detection.cascade_detect.preprocess import _decode_jpeg


if TYPE_CHECKING:
    from src.clients.triton_pool import AsyncTritonPool
    from src.config import DetectionProfile


logger = logging.getLogger(__name__)


# =============================================================================
# PaddleOcrRegionDetector — "last cheap try" detector
# =============================================================================


class PaddleOcrRegionDetector:
    """Last-ditch region detector using a PaddleOCR text-detection model (DBNet).

    Routing fallback: when both the primary detector and the secondary
    segmenter came up empty, we run PaddleOCR's text detector on the
    crop and treat the bounding box of its strongest text region as a
    region candidate. Text-bearing regions (labels, placards, printed
    tags) are by construction text-rich rectangles, so this is a cheap
    high-recall rescue for crops the dedicated detectors missed.

    The detector returns the **single bounding box** that encloses the
    strongest connected region of "text-ish" pixels in the model's
    probability map — coarse but enough to feed the human-review tab
    or to seed the next training round.

    The detector is intentionally minimal: no full PP-OCR pipeline, no
    DBNet-style polygon decoding, no recognition step. We only need a
    rough region-shaped rectangle.
    """

    def __init__(
        self,
        triton_pool: AsyncTritonPool,
        profile: DetectionProfile,
        *,
        model_name: str | None = None,
        input_size: int | None = None,
        prob_floor: float | None = None,
    ) -> None:
        """Initialize the detector.

        Args:
            triton_pool: An initialized :class:`AsyncTritonPool`.
            profile: Model identity + thresholds for this region type.
            model_name: Override for tests; production uses
                ``profile.ocr_det_model``.
            input_size: Square input H = W (multiple of 32). The model
                accepts dynamic H/W up to 960; 640 keeps latency low
                while preserving region-sized text.
            prob_floor: Minimum DBNet probability to count as text.
        """
        self.profile = profile
        self.triton_pool = triton_pool
        self.model_name = model_name or profile.ocr_det_model
        self.input_size = input_size if input_size is not None else profile.ocr_det_input_size
        self.prob_floor = prob_floor if prob_floor is not None else profile.ocr_det_prob_floor

    async def detect(self, crop_jpeg: bytes) -> RegionCandidate | None:
        """Run PaddleOCR detection on one crop. Returns the strongest box.

        Args:
            crop_jpeg: JPEG-encoded item crop.

        Returns:
            A :class:`RegionCandidate` (``source=<profile.ocr_det_model>``)
            in the crop's normalized frame, or ``None`` if the model
            produced nothing above the probability floor or the crop
            failed to decode.
        """
        try:
            img = _decode_jpeg(crop_jpeg)
        except ValueError as exc:
            logger.warning('paddle_decode_failed: %s', exc)
            return None

        crop_w, crop_h = img.size
        try:
            chw = self._preprocess(img)
        except ValueError as exc:
            logger.warning('paddle_preprocess_failed: %s', exc)
            return None

        inp = InferInput('x', list(chw.shape), 'FP32')
        inp.set_data_from_numpy(chw)
        outs = [InferRequestedOutput('fetch_name_0')]

        try:
            result = await self.triton_pool.infer(self.model_name, [inp], outputs=outs)
        except Exception:
            logger.exception('paddle_infer_failed')
            return None

        prob = result.as_numpy('fetch_name_0')
        if prob is None:
            logger.warning('paddle_infer_missing_output: model=%s', self.model_name)
            return None

        return self._decode(prob, crop_w=crop_w, crop_h=crop_h)

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        """Resize-and-pad ``img`` to ``(input_size, input_size)`` BGR FP32.

        PP-OCRv5 expects ``BGR`` order normalized to ``(x / 127.5) - 1``.
        We resize the long edge to ``input_size`` and pad the rest with
        zeros so the network sees a square — the model's TRT plan was
        built with dynamic H/W but we keep things simple here.
        """
        w, h = img.size
        if w == 0 or h == 0:
            msg = f'degenerate crop size: ({w}, {h})'
            raise ValueError(msg)
        target = self.input_size
        scale = min(target / w, target / h)
        new_w = max(32, round(w * scale))
        new_h = max(32, round(h * scale))
        # Round dimensions to multiples of 32 (model constraint).
        new_w = (new_w // 32) * 32 or 32
        new_h = (new_h // 32) * 32 or 32
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new('RGB', (target, target), (0, 0, 0))
        canvas.paste(resized, (0, 0))
        # PIL is RGB; PaddleOCR expects BGR.
        arr = np.asarray(canvas, dtype=np.float32)[:, :, ::-1]
        arr = arr / 127.5 - 1.0
        chw = np.transpose(arr, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(chw, dtype=np.float32)

    def _decode(
        self,
        prob: np.ndarray,
        *,
        crop_w: int,
        crop_h: int,
    ) -> RegionCandidate | None:
        """Find the strongest connected region of text-ish pixels.

        The model returns a per-pixel probability map at the network's
        output resolution. We threshold, take the bounding box of the
        argmax row/col extents (cheaper than scipy.label), then map back
        to the crop's normalized frame. A single rectangle is enough —
        we don't need polygon-perfect text boxes.
        """
        arr = np.asarray(prob)
        if arr.ndim == 4:
            arr = arr[0, 0]
        elif arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            return None

        mask = arr >= self.prob_floor
        if not mask.any():
            return None

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        h_out, w_out = arr.shape
        # Map from output grid → square network input → original crop.
        # The preprocessor pads bottom/right with zeros, so output
        # coordinates already align 1:1 with input coordinates after
        # scaling by output / input ratios.
        nx1 = cmin / max(w_out - 1, 1)
        ny1 = rmin / max(h_out - 1, 1)
        nx2 = (cmax + 1) / max(w_out, 1)
        ny2 = (rmax + 1) / max(h_out, 1)

        # The square network input was filled with the resized crop in
        # the top-left and zero padding elsewhere. Recover the crop
        # coordinates by un-scaling against the same letterbox math.
        # (We scaled by min(target/w, target/h); the un-scale is
        # symmetric in x/y because we kept aspect ratio.)
        scale = min(self.input_size / max(crop_w, 1), self.input_size / max(crop_h, 1))
        scaled_w = max(crop_w * scale, 1.0) / self.input_size  # fraction of net input occupied
        scaled_h = max(crop_h * scale, 1.0) / self.input_size

        nx1 = max(0.0, min(1.0, nx1 / scaled_w))
        ny1 = max(0.0, min(1.0, ny1 / scaled_h))
        nx2 = max(0.0, min(1.0, nx2 / scaled_w))
        ny2 = max(0.0, min(1.0, ny2 / scaled_h))

        if nx2 <= nx1 or ny2 <= ny1:
            return None

        # Score = peak probability inside the bbox. Text-rich regions
        # light up strongly, so this trends toward 1.0 on real hits.
        score = float(arr[mask].max())
        return RegionCandidate(
            bbox_norm=(nx1, ny1, nx2, ny2),
            score=score,
            source=self.model_name,
            rectangularity=None,
        )
