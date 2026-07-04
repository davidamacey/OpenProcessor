"""
Detection output adapters — serve YOLO11 end2end and YOLO26 side by side.

Two detector output contracts exist in the Triton model repository:

- **End2End NMS** (YOLO11 exported with the EfficientNMS_TRT plugin):
  four tensors — ``num_dets [1]``, ``det_boxes [300,4]``,
  ``det_scores [300]``, ``det_classes [300]``.
- **Fused NMS-free** (YOLO26 native export, or ultralytics ``nms=True``
  exports): one tensor shaped ``(batch, max_det, 6)`` of
  ``[x1, y1, x2, y2, score, class]`` rows, zero-padded past the last
  real detection.

The adapter for a model is resolved from **Triton model metadata** (output
names/shapes), not from the model's name — so any conforming engine works
regardless of naming convention. Both adapters normalize to the same
result dict consumed by the rest of the pipeline:
``{'num_dets', 'boxes', 'scores', 'classes'}`` per image.
"""

import logging
from typing import Any, Protocol

import numpy as np


logger = logging.getLogger(__name__)

_END2END_OUTPUTS = ('num_dets', 'det_boxes', 'det_scores', 'det_classes')


class DetectionAdapter(Protocol):
    """Parses a Triton detection response into per-image result dicts."""

    requested_outputs: tuple[str, ...]

    def parse(self, response: Any, batch_size: int) -> list[dict[str, Any]]:
        """Return one ``{'num_dets','boxes','scores','classes'}`` per image."""
        ...


class End2EndNMSAdapter:
    """EfficientNMS_TRT contract: num_dets/det_boxes/det_scores/det_classes."""

    requested_outputs: tuple[str, ...] = _END2END_OUTPUTS

    def parse(self, response: Any, batch_size: int) -> list[dict[str, Any]]:
        num_dets_batch = response.as_numpy('num_dets')
        boxes_batch = response.as_numpy('det_boxes')
        scores_batch = response.as_numpy('det_scores')
        classes_batch = response.as_numpy('det_classes')

        results = []
        for i in range(batch_size):
            num_dets = int(num_dets_batch[i][0])
            results.append(
                {
                    'num_dets': num_dets,
                    'boxes': boxes_batch[i][:num_dets],
                    'scores': scores_batch[i][:num_dets],
                    'classes': classes_batch[i][:num_dets],
                }
            )
        return results


class FusedDetAdapter:
    """Fused NMS-free contract: one ``(batch, max_det, 6)`` tensor.

    Rows are ``[x1, y1, x2, y2, score, class]`` in letterboxed input
    coordinates (same frame as the end2end contract); padding rows carry
    score 0 and are dropped.
    """

    def __init__(self, output_name: str):
        self.output_name = output_name
        self.requested_outputs: tuple[str, ...] = (output_name,)

    def parse(self, response: Any, batch_size: int) -> list[dict[str, Any]]:
        fused = response.as_numpy(self.output_name)

        results = []
        for i in range(batch_size):
            rows = fused[i]
            valid = rows[:, 4] > 0.0
            kept = rows[valid]
            results.append(
                {
                    'num_dets': int(kept.shape[0]),
                    'boxes': kept[:, :4],
                    'scores': kept[:, 4],
                    'classes': kept[:, 5].astype(np.int32),
                }
            )
        return results


def resolve_adapter(metadata: Any) -> DetectionAdapter:
    """Pick the adapter for a model from its Triton metadata.

    Args:
        metadata: ``get_model_metadata()`` response (protobuf) — each
            entry in ``metadata.outputs`` exposes ``name`` and ``shape``.

    Raises:
        ValueError: when the output signature matches neither contract.
    """
    outputs = list(metadata.outputs)
    names = {o.name for o in outputs}

    if set(_END2END_OUTPUTS) <= names:
        logger.info('Detection adapter: End2EndNMS (%s)', sorted(names))
        return End2EndNMSAdapter()

    if len(outputs) == 1:
        out = outputs[0]
        dims = list(out.shape)
        # Shape may or may not include the dynamic batch axis (-1) depending
        # on the model config; the detection row is always the LAST axis.
        if dims and dims[-1] == 6:
            logger.info('Detection adapter: FusedDet (output=%s, shape=%s)', out.name, dims)
            return FusedDetAdapter(out.name)

    raise ValueError(
        f'Model outputs {sorted(names)} match neither the end2end NMS contract '
        f'{_END2END_OUTPUTS} nor the fused single-tensor (…, 6) contract. '
        'Is this a detection model?'
    )
