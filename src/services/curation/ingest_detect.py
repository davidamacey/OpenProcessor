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

* **Primary** — a fused export whose single ``output0`` tensor
  (``[batch, 300, 6]``, NMS already applied in the engine, letterbox pixels) is read through
  :class:`~src.clients.model_adapters.FusedDetAdapter`. Every family serves it.
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

from src.clients.model_adapters import FUSED_OUTPUT, FusedDetAdapter
from src.clients.triton_pool_core import images_input
from src.config.ingest_profiles import model_class_name, proposer_label
from src.core.logging import get_logger
from src.services.curation.detector_class_map import registry_class_for
from src.services.curation.ingest_policy import registry_name_index
from src.services.curation.item_doc import DetectedItem
from src.services.detection.geometry import iou as _iou_fn, letterbox_to_square, undo_letterbox
from src.utils.class_names import normalize_class_name
from src.utils.cpu_pool import run_cpu


if TYPE_CHECKING:
    from PIL import Image

    from src.clients.curation_opensearch import ClassRegistryFile
    from src.clients.triton_pool_core import AsyncTritonPool
    from src.config import DetectionProfile


# Minimum IoU for a secondary-detector box to be considered the same
# object as a primary-detector box. Kept independent of either profile's
# confidence_floor since the two detectors are calibrated differently.
SECONDARY_IOU_MATCH = 0.3

RAW_OUTPUT = 'output0'

_FUSED = FusedDetAdapter(FUSED_OUTPUT)

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
        registry: ClassRegistryFile,
        profile: DetectionProfile,
        secondary_profile: DetectionProfile | None = None,
    ) -> None:
        self.triton_pool = triton_pool
        self.registry = registry
        self.profile = profile
        self.secondary_profile = secondary_profile

    async def _images(self, model: str, pixels: np.ndarray) -> Any:
        """The ``images`` input for ``model`` in the datatype it declares."""
        datatype = await self.triton_pool.get_model_input_datatype(model)
        return await run_cpu(images_input, pixels, datatype, nbytes=pixels.nbytes)

    # ------------------------------------------------------------------
    # Primary (fused) detector
    # ------------------------------------------------------------------

    async def run_primary(self, img: Image.Image) -> list[DetectedItem]:
        """Run the primary fused detector over one full image.

        Contract: the model returns one ``[300, 6]`` tensor of already-NMS'd rows
        ``x1, y1, x2, y2, score, class`` in letterboxed network-input pixels, the same tensor
        this repo's own ``/detect`` endpoint serves from.
        """
        from tritonclient.grpc import InferRequestedOutput

        chw, scale, pad = await run_cpu(
            letterbox_to_square,
            img,
            target=self.profile.input_size,
            fill=self.profile.letterbox_fill,
            nbytes=_rgb_bytes(img),
        )
        inp = await self._images(self.profile.detector_model, chw)
        outs = [InferRequestedOutput(FUSED_OUTPUT)]
        result = await self.triton_pool.infer(self.profile.detector_model, [inp], outputs=outs)
        found = _FUSED.parse(result, 1)[0]
        return self.decode_primary_row(
            found['boxes'], found['scores'], found['classes'], scale, pad
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

        letterboxed = await asyncio.gather(
            *(
                run_cpu(
                    letterbox_to_square,
                    img,
                    target=self.profile.input_size,
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

        outs = [InferRequestedOutput(FUSED_OUTPUT)]
        found: list[dict[str, Any]] = []
        step = max(1, self.profile.batch_limit)
        for start in range(0, full_batch.shape[0], step):
            chunk = full_batch[start : start + step]
            inp = await self._images(self.profile.detector_model, chunk)
            result = await self.triton_pool.infer(self.profile.detector_model, [inp], outputs=outs)
            found.extend(_FUSED.parse(result, chunk.shape[0]))

        return [
            self.decode_primary_row(f['boxes'], f['scores'], f['classes'], scales[i], pads[i])
            for i, f in enumerate(found)
        ]

    def decode_primary_row(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        scale: float,
        pad: tuple[float, float],
    ) -> list[DetectedItem]:
        """Decode one image's detections (``boxes`` in letterboxed network-input pixels, one
        row per real detection, as :class:`~src.clients.model_adapters.FusedDetAdapter`
        returns them) into items.

        A detection below ``profile.confidence_floor`` stays an unlabeled
        proposal: its class fields are left unset but the proposed name
        and raw score are kept for lineage. A non-empty
        ``profile.class_ids`` drops detections of any other class. With
        ``profile.assigns_class`` false every detection is an unlabeled
        ``{name}_proposal`` carrying the proposer's own label. With it true the registry class is
        the active one named like the model's output (never the output index); a model name
        with no such class stays an unlabeled ``{name}_low_conf`` proposal.
        """
        floor = self.profile.confidence_floor
        by_name = registry_name_index(self.registry) if self.profile.assigns_class else {}

        out: list[DetectedItem] = []
        for box, score, cls in zip(boxes, scores, classes, strict=True):
            full = undo_letterbox(
                (float(box[0]), float(box[1]), float(box[2]), float(box[3])), scale, pad
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
            model_name = model_class_name(self.profile, cls_id)
            found = registry_class_for(self.profile, cls_id, by_name)
            registry_id, class_name = found if found is not None else (None, None)
            confident = conf >= floor and found is not None
            out.append(
                DetectedItem(
                    bbox_pixel=full,
                    score=conf,
                    class_id=registry_id if confident else None,
                    class_name=class_name if confident else None,
                    class_source=(
                        f'{self.profile.name}_model'
                        if confident
                        else f'{self.profile.name}_low_conf'
                    ),
                    proposal_name=class_name or model_name,
                    detector_class_name=normalize_class_name(model_name or class_name or '')
                    or None,
                    detector_class_id=registry_id,
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

        return [InferRequestedOutput(RAW_OUTPUT)]

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
        inp = await self._images(profile.detector_model, chw)
        result = await self.triton_pool.infer(
            profile.detector_model, [inp], outputs=self._secondary_outputs()
        )
        return SecondaryOutput(raw=result.as_numpy(RAW_OUTPUT))

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
            inp = await self._images(profile.detector_model, chunk)
            result = await self.triton_pool.infer(profile.detector_model, [inp], outputs=outs)
            raws = result.as_numpy(RAW_OUTPUT)
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
        by_name = registry_name_index(self.registry)
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
            found = registry_class_for(profile, sec_cls, by_name)
            if found is None:
                continue
            item.class_id, item.class_name = found
            item.class_source = f'{profile.name}_model'
            item.class_detector = profile.detector_model or profile.name
            item.class_detector_version = profile.detector_version
            item.score = sec_score


__all__ = ['SECONDARY_IOU_MATCH', 'SecondaryOutput', 'WholeImageDetector']
