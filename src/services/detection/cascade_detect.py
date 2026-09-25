"""Generic sub-region detection cascade — YOLO-style detector + PaddleOCR.

See ``docs/design/curation_design_rationale.md`` §2.3 / §5; this
is one of the ratchet-exempt oversize files.
Wraps a YOLO-style Triton detector to produce sub-region bounding boxes
in the **item crop's** coordinate frame (normalized to ``[0, 1]``), plus
a PaddleOCR-based text detector/recognizer used as a last-resort
rescue path and text-hint source.

Every heuristic (detector identity, confidence floors, aspect bands, OCR
wiring) lives on a :class:`~src.config.DetectionProfile` instance, so a
deployment can describe any region type (a printed label, a box, an
ID plate, …) without forking this module. **No profile ships
built in.** ``RegionDetector`` / ``PaddleOcrRegionDetector`` /
``PaddleOcrTextRecognizer`` all require a profile explicitly — the
caller resolves it from :mod:`src.services.detection.profile_registry`
(``get_active_region_profile()``, or a deployment's own registered
profile) or raises, rather than silently falling back to any example
domain. See ``examples/region_profiles/`` for a worked example a
deployment can point ``OP_REGION_PROFILE_PATH`` at.
"""

from __future__ import annotations

import io
import logging
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from tritonclient.grpc import InferInput, InferRequestedOutput

from src.config import DetectionProfile, get_region_fields
from src.services.detection.geometry import letterbox_to_square, undo_letterbox
from src.services.detection.profile_registry import ensure_env_region_profile
from src.services.detection.region_text import OcrLine


if TYPE_CHECKING:
    from src.clients.triton_pool import AsyncTritonPool


logger = logging.getLogger(__name__)


# Resolve (and register) the deployment's region profile from the
# environment at import time, so a bad OP_REGION_PROFILE name fails loudly
# at startup instead of on the first detection request. Unconfigured is
# valid: no profile is registered and region detection stays off.
ensure_env_region_profile()

# The detector engine is exported with a fixed
# [1, 3, N, N] input — Triton's dynamic batching layers multiple
# requests onto the GPU but each request is still a single image. We
# respect that by sending many concurrent single-image requests rather
# than stacking on the Python side.


# =============================================================================
# Result dataclass
# =============================================================================


@dataclass(frozen=True)
class RegionCandidate:
    """One sub-region detection in the crop's coordinate frame.

    Attributes:
        bbox_norm: ``(x1, y1, x2, y2)`` normalized to ``[0, 1]`` of the
            crop. Always axis-aligned with ``x2 > x1`` and ``y2 > y1``.
        score: Detector confidence in ``[0, 1]``.
        source: Detector identifier. Kept as a field so candidate
            objects from different detectors (SAM3, PaddleOCR) can be
            merged later without losing provenance.
        rectangularity: Mask-area / bbox-area ratio. ``None`` for a
            box-only detector; populated by mask-based detectors.
    """

    bbox_norm: tuple[float, float, float, float]
    score: float
    source: str = ''
    rectangularity: float | None = None


# =============================================================================
# Sanity gate + provenance helpers (Phase A2 / A3)
# =============================================================================


def _now_iso() -> str:
    """ISO-8601 UTC timestamp used for region_detected_at / class_labeled_at."""
    return datetime.now(UTC).isoformat()


def is_plausible_region_bbox(
    region_in_crop: tuple[float, float, float, float],
    parent_in_source: tuple[float, float, float, float] | None = None,
) -> tuple[bool, str]:
    """Return ``(ok, reason)``; ``reason='ok'`` on pass.

    GEOMETRY GUARD ONLY. This gate does not apply aspect-ratio or
    region-vs-parent size heuristics: shape bands built from a single
    domain's assumptions (e.g. "this region type is always wider than
    tall") silently reject legitimate detections from other domains
    (foreshortened / angled views, naturally-square sub-regions). This
    gate runs BEFORE the VLM verification step, so a rejected box never
    gets a chance to be confirmed — the shape/not-shape decision belongs
    to the verifier and the detector confidence floors, not to a shape
    prior baked into the cascade.

    The region bbox is in the **item crop** coordinate frame (normalized
    to ``[0, 1]``). ``parent_in_source`` is kept for signature stability
    (all cascade call sites pass it); it's only used for a degeneracy
    check.

    Rejects only:

    * Non-finite or non-numeric coordinates.
    * Degenerate region box (``x2 <= x1`` or ``y2 <= y1``).
    * Degenerate parent bbox, when supplied.
    """
    try:
        x1, y1, x2, y2 = region_in_crop
    except (TypeError, ValueError):
        return False, 'bbox_unpack_failed'
    for name, v in (('x1', x1), ('y1', y1), ('x2', x2), ('y2', y2)):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return False, f'{name}_not_numeric'
        if not math.isfinite(f):
            return False, f'{name}_non_finite'
    w = float(x2) - float(x1)
    h = float(y2) - float(y1)
    if w <= 0.0 or h <= 0.0:
        return False, 'degenerate_zero_size'
    if parent_in_source is not None:
        try:
            vx1, vy1, vx2, vy2 = parent_in_source
            vw = float(vx2) - float(vx1)
            vh = float(vy2) - float(vy1)
        except (TypeError, ValueError):
            return False, 'vehicle_bbox_unpack_failed'
        if vw <= 0.0 or vh <= 0.0:
            return False, 'vehicle_bbox_degenerate'
    return True, 'ok'


def region_provenance(
    detector: str,
    detector_version: str,
    *,
    bbox_frame: str = 'source',
    verifier: str | None = None,
    verifier_version: str | None = None,
    detected_at: str | None = None,
    verified_at: str | None = None,
) -> dict[str, Any]:
    """Build the region-provenance dict to merge into every region update_doc.

    Always emits ``RegionFields.detector``, ``.detector_version``,
    ``.bbox_frame``, and ``.detected_at``. Verifier fields are included
    only when supplied — the cascade writers use this for the
    VLM-verified path, the human-PUT writer uses it to stamp itself as
    its own verifier.
    """
    fields = get_region_fields()
    doc: dict[str, Any] = {
        fields.detector: detector,
        fields.detector_version: detector_version,
        fields.bbox_frame: bbox_frame,
        fields.detected_at: detected_at or _now_iso(),
    }
    if verifier is not None:
        doc[fields.verifier] = verifier
        if verifier_version is not None:
            doc[fields.verifier_version] = verifier_version
        doc[fields.verified_at] = verified_at or doc[fields.detected_at]
    return doc


def class_provenance(
    detector: str,
    detector_version: str,
    *,
    labeler: str,
    labeled_at: str | None = None,
) -> dict[str, Any]:
    """Build the class-provenance dict for crop class label writers.

    Not RegionFields-governed — ``class_*`` fields are the item's class
    label provenance, orthogonal to the region-of-interest sub-annotation.
    """
    return {
        'class_detector': detector,
        'class_detector_version': detector_version,
        'class_labeler': labeler,
        'class_labeled_at': labeled_at or _now_iso(),
    }


# =============================================================================
# Preprocessing helpers
# =============================================================================


def _decode_jpeg(jpeg_bytes: bytes) -> Image.Image:
    """Decode JPEG bytes to an EXIF-transposed RGB PIL image.

    Raises:
        ValueError: If the bytes cannot be parsed as an image.
    """
    if not jpeg_bytes:
        msg = 'empty crop bytes'
        raise ValueError(msg)
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    except UnidentifiedImageError as exc:
        msg = f'crop is not a recognizable image: {exc}'
        raise ValueError(msg) from exc
    return img


def _letterbox(
    img: Image.Image,
    target: int = 640,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Letterbox a PIL image to ``target`` by ``target`` for the detector.

    Thin wrapper over :func:`src.services.detection.geometry.letterbox_to_square`
    (shared with the curation ingest service) kept here so call sites in
    this module don't need to change; see that function for the return
    shape contract.
    """
    return letterbox_to_square(img, target=target, fill=fill)


# =============================================================================
# Output decoder
# =============================================================================


def _decode_yolo_output(
    raw: np.ndarray,
    scale: float,
    pad: tuple[float, float],
    crop_w: int,
    crop_h: int,
    confidence_floor: float = 0.4,
    input_size: int = 640,
    source: str = '',
) -> RegionCandidate | None:
    """Decode a YOLOv11-shaped ``[1, 5, N]`` raw output to a normalized region box.

    The detector is single-class, so the fifth row is the
    class-0 score. We pick the highest-scoring anchor, apply the
    confidence floor, undo the letterbox into crop-pixel space, then
    normalize to ``[0, 1]`` of the crop.

    Args:
        raw: Triton response array. Shape ``[1, 5, N]`` or ``[5, N]`` or
            ``[N, 5]`` (each accepted for robustness).
        scale: Scale factor from the letterbox transform.
        pad: ``(pad_w, pad_h)`` from the letterbox transform.
        crop_w: Original crop width in pixels.
        crop_h: Original crop height in pixels.
        confidence_floor: Drop detections below this score.
        input_size: Network input size (for the pixel-vs-normalized heuristic).
        source: Detector identifier stamped on the returned candidate.

    Returns:
        A :class:`RegionCandidate` in the crop's normalized frame, or
        ``None`` if no anchor cleared the floor.
    """
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[0]
    # Accept either [5, N] or [N, 5].
    if arr.ndim != 2:
        return None
    if arr.shape[0] == 5 and arr.shape[1] != 5:
        arr = arr.T
    if arr.shape[1] < 5:
        return None

    confs = arr[:, 4]
    best = int(np.argmax(confs))
    conf = float(confs[best])
    if conf < confidence_floor:
        return None

    cx, cy, w, h = (float(v) for v in arr[best, 0:4])

    # The Ultralytics-style TRT export emits boxes in **network-pixel
    # space** (i.e. multiplied by input_size). If a future engine ships
    # with normalized [0, 1] outputs, ``cx`` will be < 1. We detect that
    # case here: a box center smaller than one pixel only happens with
    # normalized coords. This keeps us compatible with both export styles.
    if max(abs(cx), abs(cy), abs(w), abs(h)) <= 1.5:
        cx *= input_size
        cy *= input_size
        w *= input_size
        h *= input_size

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    # Undo letterbox: subtract pad, divide by scale → crop-pixel space.
    cx1, cy1, cx2, cy2 = undo_letterbox((x1, y1, x2, y2), scale, pad)

    # Normalize to crop frame, clamp, and enforce x2 > x1 / y2 > y1.
    nx1 = max(0.0, min(1.0, cx1 / max(crop_w, 1)))
    ny1 = max(0.0, min(1.0, cy1 / max(crop_h, 1)))
    nx2 = max(0.0, min(1.0, cx2 / max(crop_w, 1)))
    ny2 = max(0.0, min(1.0, cy2 / max(crop_h, 1)))

    # Drop fully-collapsed boxes — they're decoder noise, not signal.
    if nx2 <= nx1 or ny2 <= ny1:
        return None

    return RegionCandidate(
        bbox_norm=(nx1, ny1, nx2, ny2),
        score=conf,
        source=source,
        rectangularity=None,
    )


# =============================================================================
# RegionDetector — async Triton client wrapper
# =============================================================================


class RegionDetector:
    """Async wrapper around a YOLO-style sub-region Triton model.

    The detector takes an item crop (JPEG bytes) and returns the
    highest-scoring region detection in the crop's coordinate frame, or
    ``None`` if the model produced nothing above the confidence floor.

    The model itself is loaded as part of the ``triton-server`` service.
    We don't manage its lifecycle here — we only call it.

    Example:
        ```python
        pool = AsyncTritonPool(url='triton-server:8001')
        await pool.initialize()
        profile = get_active_region_profile()  # or a deployment's own DetectionProfile
        detector = RegionDetector(pool, profile)

        region = await detector.detect(crop_jpeg_bytes)
        if region is not None:
            print(region.bbox_norm, region.score)
        ```
    """

    def __init__(
        self,
        triton_pool: AsyncTritonPool,
        profile: DetectionProfile,
        *,
        confidence_floor: float | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize the detector.

        Args:
            triton_pool: An initialized :class:`AsyncTritonPool`.
            profile: Aspect/area heuristics + model identity for this
                region type. Required -- the caller resolves the active
                profile (or its own) rather than getting a silent
                example default.
            confidence_floor: Override the profile's confidence floor;
                tests sometimes want stricter or looser filtering for
                one-off jobs.
            model_name: Override the Triton model name; tests sometimes
                point at a stubbed model.
        """
        self.profile = profile
        self.triton_pool = triton_pool
        self.confidence_floor = (
            confidence_floor if confidence_floor is not None else profile.confidence_floor
        )
        self.model_name = model_name or profile.detector_model

    # ------------------------------------------------------------------
    # Single-crop API
    # ------------------------------------------------------------------

    async def detect(self, crop_jpeg: bytes) -> RegionCandidate | None:
        """Run region detection on one item crop.

        Args:
            crop_jpeg: JPEG-encoded item crop bytes.

        Returns:
            The highest-scoring region in the crop's normalized frame,
            or ``None`` if the model returned nothing above the
            confidence floor (or if the crop could not be decoded).
        """
        try:
            img = _decode_jpeg(crop_jpeg)
        except ValueError as exc:
            logger.warning('region_decode_failed: %s', exc)
            return None

        crop_w, crop_h = img.size
        try:
            chw, scale, pad = _letterbox(
                img, target=self.profile.input_size, fill=self.profile.letterbox_fill
            )
        except ValueError as exc:
            logger.warning('region_letterbox_failed: %s', exc)
            return None

        inp = InferInput('images', list(chw.shape), 'FP32')
        inp.set_data_from_numpy(chw)
        outs = [InferRequestedOutput('output0')]

        try:
            result = await self.triton_pool.infer(self.model_name, [inp], outputs=outs)
        except Exception:
            logger.exception('region_infer_failed')
            return None

        raw = result.as_numpy('output0')
        if raw is None:
            logger.warning('region_infer_missing_output: model=%s', self.model_name)
            return None

        return _decode_yolo_output(
            raw,
            scale=scale,
            pad=pad,
            crop_w=crop_w,
            crop_h=crop_h,
            confidence_floor=self.confidence_floor,
            input_size=self.profile.input_size,
            source=self.model_name,
        )

    # ------------------------------------------------------------------
    # Batched API
    # ------------------------------------------------------------------

    async def detect_batch(
        self,
        crops_jpeg: list[bytes],
    ) -> list[RegionCandidate | None]:
        """Run region detection on many crops in parallel.

        Sends N concurrent batch=1 ``infer`` calls and lets Triton's
        dynamic batching coalesce them into real GPU batches. Measured
        (on the region-detector model): this is faster than Python-side
        stacking — Triton forms tighter batches across the whole
        instance group than we can in one process, and we don't pay the
        ``np.stack`` + per-batch decode loop overhead.

        Args:
            crops_jpeg: JPEG-encoded crop bytes, one per item crop.

        Returns:
            A list aligned 1:1 with ``crops_jpeg``. Failed crops yield
            ``None``.
        """
        import asyncio

        if not crops_jpeg:
            return []

        results: list[RegionCandidate | None] = [None] * len(crops_jpeg)

        # Chunk so a single huge call doesn't pin every channel in the
        # pool. ``profile.batch_limit`` is well under the pool's typical
        # ``max_concurrent``, leaving headroom for other Triton users to
        # overlap.
        batch_limit = self.profile.batch_limit
        for start in range(0, len(crops_jpeg), batch_limit):
            chunk = crops_jpeg[start : start + batch_limit]
            chunk_results = await asyncio.gather(
                *[self.detect(crop) for crop in chunk],
                return_exceptions=True,
            )
            for offset, res in enumerate(chunk_results):
                idx = start + offset
                if isinstance(res, BaseException):
                    logger.warning('region_batch_item_failed: idx=%d err=%s', idx, res)
                    results[idx] = None
                else:
                    results[idx] = res

        return results


# =============================================================================
# Coordinate-frame transforms
# =============================================================================


def crop_norm_to_source_norm(
    region_in_crop: tuple[float, float, float, float],
    parent_in_source: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Re-project a region bbox from the item-crop frame to the source-image frame.

    Both inputs and the output are normalized to ``[0, 1]``. The region
    bbox is in the crop's coordinate frame (which is what
    :class:`RegionDetector.detect` returns); the parent bbox tells us
    where that crop sits inside the original image. We need the
    source-image frame for YOLO training labels — Ultralytics expects
    every label row in the source image's coordinate system, regardless
    of any cropping we did during ingest.

    Math: if the parent box in source is ``(vx1, vy1, vx2, vy2)`` and
    the region in crop is ``(px1, py1, px2, py2)``, then region in
    source is::

        region_x1 = vx1 + px1 * (vx2 - vx1)
        region_y1 = vy1 + py1 * (vy2 - vy1)
        region_x2 = vx1 + px2 * (vx2 - vx1)
        region_y2 = vy1 + py2 * (vy2 - vy1)

    The result is clamped to ``[0, 1]`` defensively.

    Args:
        region_in_crop: ``(x1, y1, x2, y2)`` of the region, normalized to
            the item crop's frame.
        parent_in_source: ``(x1, y1, x2, y2)`` of the parent box,
            normalized to the source image's frame.

    Returns:
        ``(x1, y1, x2, y2)`` of the region, normalized to the source image.
    """
    px1, py1, px2, py2 = region_in_crop
    vx1, vy1, vx2, vy2 = parent_in_source
    vw = vx2 - vx1
    vh = vy2 - vy1
    sx1 = max(0.0, min(1.0, vx1 + px1 * vw))
    sy1 = max(0.0, min(1.0, vy1 + py1 * vh))
    sx2 = max(0.0, min(1.0, vx1 + px2 * vw))
    sy2 = max(0.0, min(1.0, vy1 + py2 * vh))
    return (sx1, sy1, sx2, sy2)


# =============================================================================
# PaddleOcrRegionDetector — "last cheap try" detector
# =============================================================================


class PaddleOcrRegionDetector:
    """Last-ditch region detector using a PaddleOCR text-detection model (DBNet).

    Routing fallback: when both the primary detector and the secondary
    segmenter came up empty, we run PaddleOCR's text detector on the
    crop and treat the bounding box of its strongest text region as a
    region candidate. Text-bearing regions (plates, labels, placards)
    are by construction text-rich rectangles, so this is a cheap
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
    def looks_like_region_text(self) -> bool:
        """Surface check: characters are region-text-valid + length plausible."""
        pattern = re.compile(self.profile.text_pattern)
        return bool(pattern.fullmatch(self.text)) and 4 <= len(self.text) <= 10

    @property
    def is_region_text_candidate(self) -> bool:
        """Stricter test for text-hint detection promotion.

        Real plate-like text regions almost always contain BOTH letters
        and digits, sit in a tighter aspect range than the broad sanity
        gate, and read with high OCR confidence. Bumper stickers like
        ``"FORD"`` or ``"COOLBUMPER"`` fail the letter+digit test;
        ``"DEALER"`` fails it too. Vanity plates ``"LUV2DRV"`` and
        standard plates ``"ABC1234"`` pass.

        Edge cases this intentionally rejects (acceptable false-
        negatives — the segmenter / OCR-det path should have caught
        them):
          * all-digit plates (e.g. some EU mopeds)
          * all-letter custom plates (rare)
          * short custom plates (below ``profile.text_hint_len_min``)
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
        x1, y1, x2, y2 = (max(0.0, min(1.0, float(v))) for v in boxes[i])
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(
            OcrLine(
                text=raw,
                box=(x1, y1, x2, y2),
                score=float(rec_scores[i]),
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
        general sanity gate): both letters AND digits in the canonical
        text, tightened aspect range, OCR confidence floor. Drops
        bumper-sticker / window-decal / dealer-frame matches that would
        otherwise sneak through the loose region-text regex. The
        text-hint path also only fires for crops the VLM already said
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


__all__ = [
    'OcrRegion',
    'PaddleOcrRegionDetector',
    'PaddleOcrTextRecognizer',
    'RegionCandidate',
    'RegionDetector',
    'class_provenance',
    'crop_norm_to_source_norm',
    'is_plausible_region_bbox',
    'region_provenance',
]
