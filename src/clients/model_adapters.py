"""Detection output adapter: the one fused contract every served detector speaks.

A served detector has exactly one output tensor shaped ``(batch, 300, 6)``: rows of
``[x1, y1, x2, y2, score, class]`` in letterboxed input pixels, zero-padded past the last
real detection (every family, YOLO11 through YOLO26, exports to it; see
:mod:`src.domain.model_families`). The adapter is resolved from **Triton model metadata**
(output names/shapes), not from the model's name, and a model that does not match fails
closed: an engine that predates the fused contract has to be exported again.

The adapter normalizes to the result dict consumed by the rest of the pipeline:
``{'num_dets', 'boxes', 'scores', 'classes', 'box_frame'}`` per image.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from src.domain.model_families import MAX_DET


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.utils.affine import BoxFrame

logger = logging.getLogger(__name__)

#: The tensor a fused detector serves: ``[batch, MAX_DET, 6]`` rows of ``x1, y1, x2, y2, score,
#: class`` in letterboxed input pixels. The export writes this name into the model config.
FUSED_OUTPUT = 'output0'


class DetectionAdapter(Protocol):
    """Parses a Triton detection response into per-image result dicts."""

    requested_outputs: tuple[str, ...]
    box_frame: BoxFrame

    def parse(self, response: Any, batch_size: int) -> list[dict[str, Any]]:
        """Return one ``{'num_dets','boxes','scores','classes','box_frame'}`` per image."""
        ...


class FusedDetAdapter:
    """Fused NMS-free contract: one ``(batch, max_det, 6)`` tensor.

    Rows are ``[x1, y1, x2, y2, score, class]`` in letterboxed input
    pixels; padding rows carry score 0 and are dropped.
    """

    box_frame: BoxFrame = 'pixels'

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
                    'box_frame': self.box_frame,
                }
            )
        return results


def is_fused_shape(shape: Sequence[int]) -> bool:
    """Whether a declared output shape is the fused ``[..., MAX_DET, 6]`` detection tensor.

    The batch axis is optional (a model config with ``max_batch_size`` omits it) and ``-1``
    marks a dynamic dim; the detection rows are always the last two axes.
    """
    return [int(d) for d in shape][-2:] == [MAX_DET, 6]


def resolve_adapter(metadata: Any) -> DetectionAdapter:
    """Pick the adapter for a model from its Triton metadata.

    Args:
        metadata: ``get_model_metadata()`` response (protobuf) — each
            entry in ``metadata.outputs`` exposes ``name`` and ``shape``.

    Raises:
        ValueError: ``unsupported_detector_contract`` when the model's outputs are not the
            single ``[..., 300, 6]`` tensor; the message names the export command.
    """
    outputs = list(metadata.outputs)
    if len(outputs) == 1 and is_fused_shape(outputs[0].shape):
        out = outputs[0]
        logger.info('Detection adapter: FusedDet (output=%s, shape=%s)', out.name, list(out.shape))
        return FusedDetAdapter(out.name)

    described = [(o.name, [int(d) for d in o.shape]) for o in outputs]
    raise ValueError(
        f'unsupported_detector_contract: model outputs {described} are not the single '
        f'[..., {MAX_DET}, 6] detection tensor. Export the model again with '
        '`./openprocessor export` (or POST /models/{model_name}/export) to get the fused '
        'contract.'
    )
