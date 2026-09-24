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

If the secondary model also exposes a backbone feature-map output
(``DetectionProfile.feature_output``, ``sppf_feat`` by default — the
second head of ``export/export_detector_dual_head.py``), it is requested
alongside ``output0`` and RoI-pooled over every item's bbox into the
items-index backbone embedding
(:func:`~src.services.detection.geometry.roi_pool_sppf`). Whether the
model has that output is learned once per model from Triton metadata;
a model without it, or a pool that can't report outputs, is called
exactly as a single-head detector.

Each detector exposes a single-image method and a **batched** one. The
batched variants stack ``N`` letterboxed tensors and issue one Triton
call per ``batch_limit`` chunk rather than ``N`` single-image
round-trips; they return exactly the shape ``N`` sequential single-image
calls would, so a caller cannot tell which path produced a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config.ingest_profiles import proposer_label
from src.core.logging import get_logger
from src.services.curation.item_doc import DetectedItem
from src.services.detection.geometry import (
    iou as _iou_fn,
    letterbox_to_square,
    roi_pool_sppf,
    undo_letterbox,
)


if TYPE_CHECKING:
    from PIL import Image

    from src.clients.curation_opensearch import ClassRegistry
    from src.clients.triton_pool import AsyncTritonPool
    from src.config import DetectionProfile


# Minimum IoU for a secondary-detector box to be considered the same
# object as a primary-detector box. Kept independent of either profile's
# confidence_floor since the two detectors are calibrated differently.
SECONDARY_IOU_MATCH = 0.3

_END2END_OUTPUTS = ('num_dets', 'det_boxes', 'det_scores', 'det_classes')
_RAW_OUTPUT = 'output0'

logger = get_logger(__name__)


@dataclass
class SecondaryOutput:
    """One image's secondary-detector response.

    ``raw`` is the ``(N_anchors, 5 + nc)`` detection tensor (or with a
    leading batch dim of 1); ``feature_map`` is the ``(C, H, W)`` backbone
    feature map, or ``None`` when the model does not expose one.
    """

    raw: np.ndarray
    feature_map: np.ndarray | None = None


class WholeImageDetector:
    """Runs the configured detector(s) over full frames, single or batched."""

    def __init__(
        self,
        *,
        triton_pool: AsyncTritonPool | Any,
        registry: ClassRegistry | Any,
        profile: DetectionProfile,
        secondary_profile: DetectionProfile | None = None,
        backbone_embedding_dim: int = 1024,
    ) -> None:
        self.triton_pool = triton_pool
        self.registry = registry
        self.profile = profile
        self.secondary_profile = secondary_profile
        self.backbone_embedding_dim = backbone_embedding_dim
        # model name -> feature output name (None = model has none).
        # Only definitive probe answers are cached; a failed probe is
        # retried on the next call.
        self._feature_outputs: dict[str, str | None] = {}
        self._warned_feature_dims: set[tuple[str, int]] = set()

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
        from tritonclient.grpc import InferInput, InferRequestedOutput

        chw, scale, pad = letterbox_to_square(
            img, target=self.profile.input_size, fill=self.profile.letterbox_fill
        )
        inp = InferInput('images', list(chw.shape), 'FP32')
        inp.set_data_from_numpy(chw)
        outs = [InferRequestedOutput(name) for name in _END2END_OUTPUTS]
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
        from tritonclient.grpc import InferInput, InferRequestedOutput

        net_size = self.profile.input_size
        chws: list[np.ndarray] = []
        scales: list[float] = []
        pads: list[tuple[float, float]] = []
        for img in imgs:
            chw, scale, pad = letterbox_to_square(
                img, target=net_size, fill=self.profile.letterbox_fill
            )
            # ``letterbox_to_square`` returns (1, 3, S, S); strip the leading dim.
            chws.append(chw[0])
            scales.append(scale)
            pads.append(pad)
        full_batch = np.stack(chws, axis=0)

        outs = [InferRequestedOutput(name) for name in _END2END_OUTPUTS]
        # Per-chunk rows are collected individually rather than
        # concatenated: max_dets can differ between chunks (the end2end
        # export pads to the largest detection count in the request),
        # which would make np.concatenate raise.
        rows: dict[str, list[np.ndarray]] = {name: [] for name in _END2END_OUTPUTS}
        step = max(1, self.profile.batch_limit)
        for start in range(0, full_batch.shape[0], step):
            chunk = full_batch[start : start + step]
            inp = InferInput('images', list(chunk.shape), 'FP32')
            inp.set_data_from_numpy(chunk)
            result = await self.triton_pool.infer(self.profile.detector_model, [inp], outputs=outs)
            for name in _END2END_OUTPUTS:
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
        allowed = self.profile.class_ids

        out: list[DetectedItem] = []
        for box, score, cls in zip(
            boxes_row[:num_dets],
            scores_row[:num_dets],
            classes_row[:num_dets],
            strict=False,
        ):
            if allowed and int(cls) not in allowed:
                continue
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
                        proposal_name=proposer_label(self.profile, cls_id),
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
                    class_detector=self.profile.detector_model or self.profile.name,
                    class_detector_version=self.profile.detector_version,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Secondary (raw-output ensemble) detector
    # ------------------------------------------------------------------

    async def secondary_feature_output(self) -> str | None:
        """Name of the secondary model's feature-map output, if it has one.

        Asks Triton once per model (``get_model_output_names`` on the
        pool) whether ``profile.feature_output`` is among the model's
        outputs. A pool without that probe, an empty ``feature_output``,
        or a failing probe all mean "don't request it" — the request is
        then byte-identical to a single-head detector's.
        """
        profile = self.secondary_profile
        assert profile is not None
        wanted = profile.feature_output
        model = profile.detector_model
        if not wanted:
            return None
        if model in self._feature_outputs:
            return self._feature_outputs[model]
        probe = getattr(self.triton_pool, 'get_model_output_names', None)
        if probe is None:
            self._feature_outputs[model] = None
            return None
        try:
            names = await probe(model)
        except Exception as exc:
            logger.warning('ingest_feature_output_probe_failed', model=model, error=str(exc))
            return None
        found = wanted if wanted in names else None
        self._feature_outputs[model] = found
        logger.info('ingest_secondary_feature_output', model=model, feature_output=found)
        return found

    def _secondary_outputs(self, feature: str | None) -> list[Any]:
        from tritonclient.grpc import InferRequestedOutput

        names = [_RAW_OUTPUT] if feature is None else [_RAW_OUTPUT, feature]
        return [InferRequestedOutput(name) for name in names]

    async def run_secondary_raw(self, img: Image.Image) -> SecondaryOutput:
        """Run the secondary (raw-output) ensemble detector on one image."""
        from tritonclient.grpc import InferInput

        profile = self.secondary_profile
        assert profile is not None
        feature = await self.secondary_feature_output()
        chw, _scale, _pad = letterbox_to_square(
            img, target=profile.input_size, fill=profile.letterbox_fill
        )
        inp = InferInput('images', list(chw.shape), 'FP32')
        inp.set_data_from_numpy(chw)
        result = await self.triton_pool.infer(
            profile.detector_model, [inp], outputs=self._secondary_outputs(feature)
        )
        return SecondaryOutput(
            raw=result.as_numpy(_RAW_OUTPUT),
            feature_map=result.as_numpy(feature)[0] if feature is not None else None,
        )

    async def run_secondary_raw_batch(self, imgs: list[Image.Image]) -> list[SecondaryOutput]:
        """Batched variant of :meth:`run_secondary_raw`.

        One Triton call per ``batch_limit`` chunk instead of one per
        image. Returns each image's raw ``(N_anchors, 5 + nc)`` slice
        (plus its ``(C, H, W)`` feature map when the model has one),
        index-aligned with ``imgs``; client-side NMS still runs per image
        in :meth:`resolve_with_secondary` because each image's item list
        is enriched independently.
        """
        if not imgs:
            return []
        from tritonclient.grpc import InferInput

        profile = self.secondary_profile
        assert profile is not None
        feature = await self.secondary_feature_output()
        outs = self._secondary_outputs(feature)

        chws: list[np.ndarray] = []
        for img in imgs:
            chw, _scale, _pad = letterbox_to_square(
                img, target=profile.input_size, fill=profile.letterbox_fill
            )
            chws.append(chw[0])
        full_batch = np.stack(chws, axis=0)

        rows: list[SecondaryOutput] = []
        step = max(1, profile.batch_limit)
        for start in range(0, full_batch.shape[0], step):
            chunk = full_batch[start : start + step]
            inp = InferInput('images', list(chunk.shape), 'FP32')
            inp.set_data_from_numpy(chunk)
            result = await self.triton_pool.infer(profile.detector_model, [inp], outputs=outs)
            raws = result.as_numpy(_RAW_OUTPUT)
            fmaps = result.as_numpy(feature) if feature is not None else None
            rows.extend(
                SecondaryOutput(raw=raws[i], feature_map=fmaps[i] if fmaps is not None else None)
                for i in range(raws.shape[0])
            )
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

    def attach_backbone_embeddings(
        self,
        items: list[DetectedItem],
        feature_map: np.ndarray,
        scale: float,
        pad: tuple[float, float],
    ) -> None:
        """RoI-pool the secondary's backbone feature map over every item.

        Each item's full-image bbox is mapped into the secondary's
        letterbox space (``scale``/``pad`` from its own letterbox) and
        pooled to ``backbone_embedding_dim`` — the items-index mapping's
        dimension. A map with fewer channels is zero-padded (lossless for
        cosine similarity); one with *more* channels is refused rather
        than truncated, since a truncated vector would silently mean
        something else — the field is then left unset for these items.
        """
        profile = self.secondary_profile
        assert profile is not None
        dim = self.backbone_embedding_dim
        if feature_map.ndim != 3:
            logger.error(
                'ingest_backbone_feature_map_bad_shape',
                model=profile.detector_model,
                shape=list(feature_map.shape),
            )
            return
        channels = int(feature_map.shape[0])
        if channels != dim:
            key = (profile.detector_model, channels)
            if key not in self._warned_feature_dims:
                self._warned_feature_dims.add(key)
                log = logger.error if channels > dim else logger.warning
                log(
                    'ingest_backbone_feature_dim_mismatch',
                    model=profile.detector_model,
                    feature_channels=channels,
                    backbone_embedding_dim=dim,
                    action='skipped' if channels > dim else 'zero_padded',
                )
            if channels > dim:
                return
        pad_w, pad_h = pad
        for item in items:
            x1, y1, x2, y2 = item.bbox_pixel
            item.backbone_embedding = roi_pool_sppf(
                feature_map,
                (x1 * scale + pad_w, y1 * scale + pad_h, x2 * scale + pad_w, y2 * scale + pad_h),
                input_size=profile.input_size,
                target_dim=dim,
            )


__all__ = ['SECONDARY_IOU_MATCH', 'SecondaryOutput', 'WholeImageDetector']
