"""YOLO-style region detector: output decoders + async Triton wrapper."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from tritonclient.grpc import InferInput, InferRequestedOutput

from src.services.detection.cascade_detect.candidate import RegionCandidate
from src.services.detection.cascade_detect.preprocess import _decode_jpeg, _letterbox
from src.services.detection.geometry import undo_letterbox


if TYPE_CHECKING:
    from src.clients.triton_pool import AsyncTritonPool
    from src.config import DetectionProfile


logger = logging.getLogger(__name__)

# The detector engine is exported with a fixed
# [1, 3, N, N] input — Triton's dynamic batching layers multiple
# requests onto the GPU but each request is still a single image. We
# respect that by sending many concurrent single-image requests rather
# than stacking on the Python side.


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


# Pre-NMS safety cap on a raw detector output before it reaches W8's
# ``select_region_candidates`` (region_candidates.py): a 640-input YOLO
# head emits thousands of anchors, and greedy NMS there is O(n^2) --
# keeping only the top-scoring ``_MAX_PRE_NMS_ANCHORS`` bounds that cost
# regardless of how noisy the raw output is. Comfortably above any
# profile's ``max_regions_per_item`` (single digits in practice).
_MAX_PRE_NMS_ANCHORS = 300


def _decode_yolo_output_multi(
    raw: np.ndarray,
    scale: float,
    pad: tuple[float, float],
    crop_w: int,
    crop_h: int,
    confidence_floor: float = 0.4,
    input_size: int = 640,
    source: str = '',
) -> list[RegionCandidate]:
    """Decode a YOLOv11-shaped ``[1, 5, N]`` raw output to every candidate
    region above ``confidence_floor`` (W8: multi-candidate detector leg).

    Same geometry pipeline as :func:`_decode_yolo_output` (letterbox
    undo, crop-frame normalization, degenerate-box drop) but keeps every
    anchor clearing the floor instead of only the best one. Anchors are
    capped to the highest-scoring :data:`_MAX_PRE_NMS_ANCHORS` before
    returning -- caller (:func:`select_region_candidates`) does the real
    greedy NMS + per-profile cap.

    Returns candidates in no particular order (the caller sorts).
    """
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return []
    if arr.shape[0] == 5 and arr.shape[1] != 5:
        arr = arr.T
    if arr.shape[1] < 5:
        return []

    confs = arr[:, 4]
    keep_idx = np.nonzero(confs >= confidence_floor)[0]
    if keep_idx.size == 0:
        return []
    if keep_idx.size > _MAX_PRE_NMS_ANCHORS:
        top = np.argsort(confs[keep_idx])[::-1][:_MAX_PRE_NMS_ANCHORS]
        keep_idx = keep_idx[top]

    out: list[RegionCandidate] = []
    for idx in keep_idx:
        conf = float(confs[idx])
        cx, cy, w, h = (float(v) for v in arr[idx, 0:4])
        if max(abs(cx), abs(cy), abs(w), abs(h)) <= 1.5:
            cx *= input_size
            cy *= input_size
            w *= input_size
            h *= input_size
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        cx1, cy1, cx2, cy2 = undo_letterbox((x1, y1, x2, y2), scale, pad)
        nx1 = max(0.0, min(1.0, cx1 / max(crop_w, 1)))
        ny1 = max(0.0, min(1.0, cy1 / max(crop_h, 1)))
        nx2 = max(0.0, min(1.0, cx2 / max(crop_w, 1)))
        ny2 = max(0.0, min(1.0, cy2 / max(crop_h, 1)))
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        out.append(
            RegionCandidate(
                bbox_norm=(nx1, ny1, nx2, ny2),
                score=conf,
                source=source,
                rectangularity=None,
            )
        )
    return out


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

    async def _infer_raw(
        self, crop_jpeg: bytes
    ) -> tuple[np.ndarray, float, tuple[float, float], int, int] | None:
        """Shared preprocess + Triton call for :meth:`detect` / :meth:`detect_multi`.

        Returns ``(raw, scale, pad, crop_w, crop_h)`` or ``None`` on any
        decode / letterbox / infer failure (already logged).
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
        return raw, scale, pad, crop_w, crop_h

    async def detect(self, crop_jpeg: bytes) -> RegionCandidate | None:
        """Run region detection on one item crop.

        Args:
            crop_jpeg: JPEG-encoded item crop bytes.

        Returns:
            The highest-scoring region in the crop's normalized frame,
            or ``None`` if the model returned nothing above the
            confidence floor (or if the crop could not be decoded), or
            when no detector model is configured (no Triton call).
        """
        if not self.model_name:
            return None
        decoded = await self._infer_raw(crop_jpeg)
        if decoded is None:
            return None
        raw, scale, pad, crop_w, crop_h = decoded
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

    async def detect_multi(self, crop_jpeg: bytes) -> list[RegionCandidate]:
        """W8: every region candidate above the confidence floor (not just
        the top one), for :func:`~src.services.detection.region_candidates
        .select_region_candidates` to floor/NMS/cap.

        Returns ``[]`` on any decode/infer failure or with no detector
        model configured (same conditions :meth:`detect` returns
        ``None`` for).
        """
        if not self.model_name:
            return []
        decoded = await self._infer_raw(crop_jpeg)
        if decoded is None:
            return []
        raw, scale, pad, crop_w, crop_h = decoded
        return _decode_yolo_output_multi(
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
            ``None``; with no detector model configured every entry is
            ``None`` and Triton is never called.
        """
        import asyncio

        if not crops_jpeg:
            return []
        if not self.model_name:
            return [None] * len(crops_jpeg)

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

    async def detect_batch_multi(
        self,
        crops_jpeg: list[bytes],
    ) -> list[list[RegionCandidate]]:
        """W8: :meth:`detect_batch`'s multi-candidate counterpart.

        Returns a list aligned 1:1 with ``crops_jpeg``; each entry is
        every candidate :meth:`detect_multi` found for that crop (``[]``
        on failure or a floor miss).
        """
        import asyncio

        if not crops_jpeg:
            return []
        if not self.model_name:
            return [[] for _ in crops_jpeg]

        results: list[list[RegionCandidate]] = [[] for _ in crops_jpeg]
        batch_limit = self.profile.batch_limit
        for start in range(0, len(crops_jpeg), batch_limit):
            chunk = crops_jpeg[start : start + batch_limit]
            chunk_results = await asyncio.gather(
                *[self.detect_multi(crop) for crop in chunk],
                return_exceptions=True,
            )
            for offset, res in enumerate(chunk_results):
                idx = start + offset
                if isinstance(res, BaseException):
                    logger.warning('region_batch_item_failed: idx=%d err=%s', idx, res)
                    results[idx] = []
                else:
                    results[idx] = res

        return results
