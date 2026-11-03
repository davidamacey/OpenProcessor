"""YOLO-style region detector: output decoders + async Triton wrapper."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from tritonclient.grpc import InferRequestedOutput

from src.clients.triton_pool_core import images_input
from src.services.detection.cascade_detect.candidate import RegionCandidate
from src.services.detection.cascade_detect.preprocess import _decode_jpeg, _letterbox
from src.services.detection.geometry import undo_letterbox


if TYPE_CHECKING:
    from src.clients.triton_pool_core import AsyncTritonPool
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


def _anchors_xyxy_score(raw: np.ndarray, input_size: int) -> tuple[np.ndarray, np.ndarray] | None:
    """The head's rows as network-pixel ``xyxy`` boxes and their scores.

    The head is picked from the output shape, never assumed. A YOLO11-style
    single-class head is ``[1, 5, N]`` (or ``[5, N]`` / ``[N, 5]``) with
    ``cx, cy, w, h, score`` per anchor. A promoted YOLO26 / end2end model emits
    ``[1, 300, 6]``: one ``x1, y1, x2, y2, score, class`` row per detection,
    already NMS-free. ``None`` for any other shape.
    """
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    if arr.shape[1] == _END2END_COLS and arr.shape[0] != _END2END_COLS:
        boxes = arr[:, 0:4].astype(np.float64)
    else:
        if arr.shape[0] == 5 and arr.shape[1] != 5:
            arr = arr.T
        if arr.shape[1] < 5:
            return None
        c = arr[:, 0:4].astype(np.float64)
        boxes = np.stack(
            [
                c[:, 0] - c[:, 2] / 2,
                c[:, 1] - c[:, 3] / 2,
                c[:, 0] + c[:, 2] / 2,
                c[:, 1] + c[:, 3] / 2,
            ],
            axis=1,
        )
    # The Ultralytics-style TRT export emits boxes in network-pixel space. An engine
    # shipping normalized [0, 1] outputs is told apart per row: no pixel box stays
    # within 1.5 of the origin on both corners.
    normalized = np.abs(boxes).max(axis=1) <= 1.5
    boxes[normalized] *= input_size
    return boxes, arr[:, 4]


#: Columns of an end2end row: ``x1, y1, x2, y2, score, class``.
_END2END_COLS = 6


def _candidate(
    box: np.ndarray,
    score: float,
    scale: float,
    pad: tuple[float, float],
    crop_w: int,
    crop_h: int,
    source: str,
) -> RegionCandidate | None:
    """One network-pixel box undone through the letterbox into the crop's normalized frame."""
    cx1, cy1, cx2, cy2 = undo_letterbox(
        (float(box[0]), float(box[1]), float(box[2]), float(box[3])), scale, pad
    )
    nx1 = max(0.0, min(1.0, cx1 / max(crop_w, 1)))
    ny1 = max(0.0, min(1.0, cy1 / max(crop_h, 1)))
    nx2 = max(0.0, min(1.0, cx2 / max(crop_w, 1)))
    ny2 = max(0.0, min(1.0, cy2 / max(crop_h, 1)))
    # A fully collapsed box is decoder noise, not signal.
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return RegionCandidate(
        bbox_norm=(nx1, ny1, nx2, ny2), score=score, source=source, rectangularity=None
    )


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
    """Decode a detector output to the highest-scoring region, normalized to the crop.

    Handles both heads (see :func:`_anchors_xyxy_score`). The detector is
    treated as single-class: the best-scoring row wins, the confidence floor
    applies, then the letterbox is undone.

    Returns:
        A :class:`RegionCandidate` in the crop's normalized frame, or ``None``
        if no row cleared the floor or the output shape is not a known head.
    """
    decoded = _anchors_xyxy_score(raw, input_size)
    if decoded is None:
        return None
    boxes, confs = decoded
    best = int(np.argmax(confs))
    conf = float(confs[best])
    if conf < confidence_floor:
        return None
    return _candidate(boxes[best], conf, scale, pad, crop_w, crop_h, source)


# Pre-NMS safety cap on a raw detector output before it reaches
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
    """Decode a detector output to every candidate region above ``confidence_floor``
    (the multi-candidate detector leg).

    Same geometry as :func:`_decode_yolo_output` but keeps every row clearing the
    floor, capped to the highest-scoring :data:`_MAX_PRE_NMS_ANCHORS`; the caller
    (:func:`select_region_candidates`) does the real greedy NMS and per-profile cap.
    Candidates come back in no particular order.
    """
    decoded = _anchors_xyxy_score(raw, input_size)
    if decoded is None:
        return []
    boxes, confs = decoded
    keep_idx = np.nonzero(confs >= confidence_floor)[0]
    if keep_idx.size > _MAX_PRE_NMS_ANCHORS:
        top = np.argsort(confs[keep_idx])[::-1][:_MAX_PRE_NMS_ANCHORS]
        keep_idx = keep_idx[top]
    out: list[RegionCandidate] = []
    for idx in keep_idx:
        cand = _candidate(boxes[idx], float(confs[idx]), scale, pad, crop_w, crop_h, source)
        if cand is not None:
            out.append(cand)
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

        outs = [InferRequestedOutput('output0')]

        try:
            datatype = await self.triton_pool.get_model_input_datatype(self.model_name)
            # A promoted model declares UINT8 (the graph scales); an older one FP32.
            inp = images_input(chw, datatype)
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
        """Every region candidate above the confidence floor (not just
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

    async def detect_batch_multi(
        self,
        crops_jpeg: list[bytes],
    ) -> list[list[RegionCandidate]]:
        """Run :meth:`detect_multi` over a batch of crops.

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
