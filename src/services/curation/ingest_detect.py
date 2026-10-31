"""Whole-image detector inference for the curation ingest pipeline.

Split out of :mod:`src.services.curation.ingest` so that module keeps one
concern (per-image pipeline orchestration + document writing) and this
one keeps another: turning PIL images into
:class:`~src.services.curation.item_doc.DetectedItem` lists by talking to
Triton.

Everything here is driven by :class:`~src.config.DetectionProfile` —
model name, network input size, letterbox fill, confidence floor and the
engine's ``max_batch_size`` (``batch_limit``). No model name, class
taxonomy or input size is hardcoded.

Two detectors are supported:

* **Primary** — an *end2end* export whose Triton response is already
  NMS'd (``num_dets`` / ``det_boxes`` / ``det_scores`` / ``det_classes``).
* **Secondary** (optional) — a raw-output ensemble/backbone detector
  whose single ``output0`` tensor needs client-side NMS
  (:func:`~src.services.detection.ensemble_nms.apply_ensemble_nms`).
  Its hits override the primary's class assignment on IoU-matched boxes.

Each detector exposes a single-image method and a **batched** one. The
batched variants stack ``N`` letterboxed tensors and issue one Triton
call per ``batch_limit`` chunk rather than ``N`` single-image
round-trips; they return exactly the shape ``N`` sequential single-image
calls would, so a caller cannot tell which path produced a result.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.clients.model_adapters import END2END_OUTPUTS
from src.clients.triton_pool_core import uint8_images_input
from src.config.ingest_profiles import proposer_label
from src.core.logging import get_logger
from src.services.curation.item_doc import DetectedItem
from src.services.detection.geometry import iou as _iou_fn, letterbox_to_square, undo_letterbox
from src.utils.class_names import normalize_class_name
from src.utils.cpu_pool import run_cpu


if TYPE_CHECKING:
    from PIL import Image

    from src.clients.curation_opensearch.registry import ClassRegistry
    from src.clients.triton_pool_core import AsyncTritonPool
    from src.config import DetectionProfile


# Minimum IoU for a secondary-detector box to be considered the same
# object as a primary-detector box. Kept independent of either profile's
# confidence_floor since the two detectors are calibrated differently.
SECONDARY_IOU_MATCH = 0.3

_RAW_OUTPUT = 'output0'

logger = get_logger(__name__)


def _rgb_bytes(img: Image.Image) -> int:
    return img.width * img.height * 3


@dataclass
class SecondaryOutput:
    """One image's secondary-detector response.

    ``raw`` is the ``(N_anchors, 5 + nc)`` detection tensor (or with a
    leading batch dim of 1).
    """

    raw: np.ndarray


class WholeImageDetector:
    """Runs the configured detector(s) over full frames, single or batched."""

    def __init__(
        self,
        *,
        triton_pool: AsyncTritonPool | Any,
        registry: ClassRegistry | Any,
        profile: DetectionProfile,
        secondary_profile: DetectionProfile | None = None,
    ) -> None:
        self.triton_pool = triton_pool
        self.registry = registry
        self.profile = profile
        self.secondary_profile = secondary_profile

    # ------------------------------------------------------------------
    # Primary (end2end) detector
    # ------------------------------------------------------------------

    async def run_primary(self, img: Image.Image) -> list[DetectedItem]:
        """Run the primary end2end detector over one full image.

        Contract: the model returns already-NMS'd detections as
        ``num_dets`` / ``det_boxes`` (normalized ``[0, 1]`` of the network
        input) / ``det_scores`` / ``det_classes`` — the same Ultralytics
        TensorRT end2end export shape this repo's own ``/detect`` endpoint
        serves from.
        """
        from tritonclient.grpc import InferRequestedOutput

        chw, scale, pad = await run_cpu(
            letterbox_to_square,
            img,
            target=self.profile.input_size,
            fill=self.profile.letterbox_fill,
            nbytes=_rgb_bytes(img),
        )
        inp = await run_cpu(uint8_images_input, chw, nbytes=chw.nbytes)
        outs = [InferRequestedOutput(name) for name in END2END_OUTPUTS]
        result = await self.triton_pool.infer(self.profile.detector_model, [inp], outputs=outs)
        return self.decode_primary_row(
            result.as_numpy('num_dets')[0],
            result.as_numpy('det_boxes')[0],
            result.as_numpy('det_scores')[0],
            result.as_numpy('det_classes')[0],
            scale,
            pad,
            self.profile.input_size,
        )

    async def run_primary_batch(self, imgs: list[Image.Image]) -> list[list[DetectedItem]]:
        """Batched variant of :meth:`run_primary`.

        Stacks per-image letterboxed CHW tensors into an ``(N, 3, S, S)``
        batch and issues **one Triton call per ``batch_limit`` chunk** —
        not one call per image. This is the whole point of the batch
        ingest path: the GPU sees a few large inferences instead of ``N``
        round-trips each paying full request + H2D/D2H latency.
        """
        if not imgs:
            return []
        from tritonclient.grpc import InferRequestedOutput

        net_size = self.profile.input_size
        letterboxed = await asyncio.gather(
            *(
                run_cpu(
                    letterbox_to_square,
                    img,
                    target=net_size,
                    fill=self.profile.letterbox_fill,
                    nbytes=_rgb_bytes(img),
                )
                for img in imgs
            )
        )
        # ``letterbox_to_square`` returns (1, 3, S, S); strip the leading dim.
        scales = [scale for _chw, scale, _pad in letterboxed]
        pads = [pad for _chw, _scale, pad in letterboxed]
        tensors = [chw[0] for chw, _scale, _pad in letterboxed]
        full_batch = await run_cpu(np.stack, tensors, axis=0, nbytes=sum(t.nbytes for t in tensors))

        outs = [InferRequestedOutput(name) for name in END2END_OUTPUTS]
        # Per-chunk rows are collected individually rather than
        # concatenated: max_dets can differ between chunks (the end2end
        # export pads to the largest detection count in the request),
        # which would make np.concatenate raise.
        rows: dict[str, list[np.ndarray]] = {name: [] for name in END2END_OUTPUTS}
        step = max(1, self.profile.batch_limit)
        for start in range(0, full_batch.shape[0], step):
            chunk = full_batch[start : start + step]
            inp = await run_cpu(uint8_images_input, chunk, nbytes=chunk.nbytes)
            result = await self.triton_pool.infer(self.profile.detector_model, [inp], outputs=outs)
            for name in END2END_OUTPUTS:
                rows[name].extend(result.as_numpy(name))

        return [
            self.decode_primary_row(
                rows['num_dets'][i],
                rows['det_boxes'][i],
                rows['det_scores'][i],
                rows['det_classes'][i],
                scales[i],
                pads[i],
                net_size,
            )
            for i in range(len(imgs))
        ]

    def decode_primary_row(
        self,
        num_dets_row: np.ndarray,
        boxes_row: np.ndarray,
        scores_row: np.ndarray,
        classes_row: np.ndarray,
        scale: float,
        pad: tuple[float, float],
        net_size: int,
    ) -> list[DetectedItem]:
        """Decode one image's end2end detector output row into items.

        A detection below ``profile.confidence_floor`` stays an unlabeled
        proposal: its class fields are left unset but the proposed name
        and raw score are kept for lineage. A non-empty
        ``profile.class_ids`` drops detections of any other class. With
        ``profile.assigns_class`` false every detection is an unlabeled
        ``{name}_proposal`` carrying the proposer's own label.
        """
        num_dets = int(num_dets_row[0])
        floor = self.profile.confidence_floor

        out: list[DetectedItem] = []
        for box, score, cls in zip(
            boxes_row[:num_dets],
            scores_row[:num_dets],
            classes_row[:num_dets],
            strict=False,
        ):
            full = undo_letterbox(
                (
                    float(box[0]) * net_size,
                    float(box[1]) * net_size,
                    float(box[2]) * net_size,
                    float(box[3]) * net_size,
                ),
                scale,
                pad,
            )
            cls_id = int(cls)
            conf = float(score)
            if not self.profile.assigns_class:
                # Generic proposer: its class ids are not registry ids, so
                # never label from them -- keep its own label for lineage.
                out.append(
                    DetectedItem(
                        bbox_pixel=full,
                        score=conf,
                        class_source=f'{self.profile.name}_proposal',
                        proposal_name=(label := proposer_label(self.profile, cls_id)),
                        detector_class_name=normalize_class_name(label) or None,
                        detector_confidence=conf,
                        class_detector=self.profile.detector_model or self.profile.name,
                        class_detector_version=self.profile.detector_version,
                    )
                )
                continue
            entry = self.registry.get(cls_id)
            class_name = entry.class_name if entry is not None else None
            confident = conf >= floor
            out.append(
                DetectedItem(
                    bbox_pixel=full,
                    score=conf,
                    class_id=cls_id if confident else None,
                    class_name=class_name if confident else None,
                    class_source=(
                        f'{self.profile.name}_model'
                        if confident
                        else f'{self.profile.name}_low_conf'
                    ),
                    proposal_name=class_name,
                    detector_class_name=normalize_class_name(class_name or '') or None,
                    detector_class_id=cls_id if entry is not None else None,
                    detector_confidence=conf,
                    class_detector=self.profile.detector_model or self.profile.name,
                    class_detector_version=self.profile.detector_version,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Secondary (raw-output ensemble) detector
    # ------------------------------------------------------------------

    @staticmethod
    def _secondary_outputs() -> list[Any]:
        from tritonclient.grpc import InferRequestedOutput

        return [InferRequestedOutput(_RAW_OUTPUT)]

    async def run_secondary_raw(self, img: Image.Image) -> SecondaryOutput:
        """Run the secondary (raw-output) ensemble detector on one image."""
        profile = self.secondary_profile
        assert profile is not None
        chw, _scale, _pad = await run_cpu(
            letterbox_to_square,
            img,
            target=profile.input_size,
            fill=profile.letterbox_fill,
            nbytes=_rgb_bytes(img),
        )
        inp = await run_cpu(uint8_images_input, chw, nbytes=chw.nbytes)
        result = await self.triton_pool.infer(
            profile.detector_model, [inp], outputs=self._secondary_outputs()
        )
        return SecondaryOutput(raw=result.as_numpy(_RAW_OUTPUT))

    async def run_secondary_raw_batch(self, imgs: list[Image.Image]) -> list[SecondaryOutput]:
        """Batched variant of :meth:`run_secondary_raw`.

        One Triton call per ``batch_limit`` chunk instead of one per
        image. Returns each image's raw ``(N_anchors, 5 + nc)`` slice,
        index-aligned with ``imgs``; client-side NMS still runs per image
        in :meth:`resolve_with_secondary` because each image's item list
        is enriched independently.
        """
        if not imgs:
            return []
        profile = self.secondary_profile
        assert profile is not None
        outs = self._secondary_outputs()

        letterboxed = await asyncio.gather(
            *(
                run_cpu(
                    letterbox_to_square,
                    img,
                    target=profile.input_size,
                    fill=profile.letterbox_fill,
                    nbytes=_rgb_bytes(img),
                )
                for img in imgs
            )
        )
        tensors = [chw[0] for chw, _scale, _pad in letterboxed]
        full_batch = await run_cpu(np.stack, tensors, axis=0, nbytes=sum(t.nbytes for t in tensors))

        rows: list[SecondaryOutput] = []
        step = max(1, profile.batch_limit)
        for start in range(0, full_batch.shape[0], step):
            chunk = full_batch[start : start + step]
            inp = await run_cpu(uint8_images_input, chunk, nbytes=chunk.nbytes)
            result = await self.triton_pool.infer(profile.detector_model, [inp], outputs=outs)
            raws = result.as_numpy(_RAW_OUTPUT)
            rows.extend(SecondaryOutput(raw=raws[i]) for i in range(raws.shape[0]))
        return rows[: len(imgs)]

    def resolve_with_secondary(
        self,
        items: list[DetectedItem],
        raw_output: np.ndarray,
        scale: float,
        pad: tuple[float, float],
    ) -> None:
        """Enrich ``items`` in place using the secondary detector's NMS output.

        A secondary detection above its own confidence floor overrides
        the IoU-matched primary box's class assignment; this is how a
        two-detector ``DetectionProfile`` pair takes
        :func:`apply_ensemble_nms` off zero production callers.
        """
        from src.services.detection.ensemble_nms import apply_ensemble_nms

        profile = self.secondary_profile
        assert profile is not None
        per_image = apply_ensemble_nms(
            raw_output[None, ...] if raw_output.ndim == 2 else raw_output,
            conf_thres=profile.confidence_floor,
        )
        detections = per_image[0] if per_image else []
        secondary_boxes: list[tuple[tuple[float, float, float, float], float, int]] = []
        for det in detections:
            full = undo_letterbox(tuple(det['box']), scale, pad)
            secondary_boxes.append((full, float(det['score']), int(det['class_id'])))

        for item in items:
            best_iou = 0.0
            best: tuple[tuple[float, float, float, float], float, int] | None = None
            for sec_box, sec_score, sec_cls in secondary_boxes:
                score = _iou_fn(item.bbox_pixel, sec_box)
                if score > best_iou:
                    best_iou = score
                    best = (sec_box, sec_score, sec_cls)
            if best is None or best_iou < SECONDARY_IOU_MATCH:
                continue
            _, sec_score, sec_cls = best
            entry = self.registry.get(sec_cls)
            item.class_id = sec_cls
            item.class_name = entry.class_name if entry is not None else None
            item.class_source = f'{profile.name}_model'
            item.class_detector = profile.detector_model or profile.name
            item.class_detector_version = profile.detector_version
            item.score = sec_score


__all__ = ['SECONDARY_IOU_MATCH', 'SecondaryOutput', 'WholeImageDetector']
