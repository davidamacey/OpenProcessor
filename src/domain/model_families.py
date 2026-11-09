"""What export a detection head needs, in one place the API and the trainer both read.

Standard library only: the trainer image carries ``src/domain`` and nothing of the web stack.

Ultralytics' ``nms`` export argument picks the head that is served
(https://docs.ultralytics.com/modes/export/): ``True`` embeds NMS for a head that needs it,
``False`` selects the NMS-free one-to-one head, and the default (``None``) exports the raw
one-to-many head, which this repository never serves. Both served exports end in the same
``[batch, 300, 6]`` tensor (``x1, y1, x2, y2, score, class``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ONE2ONE = 'one2one'
NMS = 'nms'
ONNX_OBJECTNESS = 'onnx_objectness'

# The kinds an export job can carry: a ``.pt`` upload is one of these two. ``onnx_objectness`` is
# the original YOLOv5 ONNX, which is probed and decoded but never exported here.
HEAD_KINDS = (ONE2ONE, NMS)

MAX_DET = 300
NMS_IOU = 0.7
NMS_CONF = 0.25


@dataclass(frozen=True)
class FamilySpec:
    """One detection family: its head, whether a ``.pt`` loads, how it is probed and decoded."""

    name: str
    head_kind: str
    pt_supported: bool
    # The ``architecture`` a probe request names for this family; ``None`` when it is not
    # selectable there (the Ultralytics loader dispatches on the checkpoint, so the named
    # architectures are the ones the probe has always accepted).
    probe_architecture: str | None
    # ``fused``: the served output is the ``[batch, 300, 6]`` tensor. ``objectness``: the raw
    # original-YOLOv5 ``[anchors, 5 + nc]`` output, decoded in Python.
    decoder: str


# ``yolov5_original`` is last so ``PROBE_ARCHITECTURES`` keeps its documented order.
FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec('yolov5u', NMS, True, None, 'fused'),
    FamilySpec('yolov8', NMS, True, None, 'fused'),
    FamilySpec('yolov9', NMS, True, None, 'fused'),
    FamilySpec('yolov10', ONE2ONE, True, None, 'fused'),
    FamilySpec('yolo11', NMS, True, 'yolo11', 'fused'),
    FamilySpec('yolo12', NMS, True, None, 'fused'),
    FamilySpec('yolo26', ONE2ONE, True, 'yolo26', 'fused'),
    FamilySpec('yolov5_original', ONNX_OBJECTNESS, False, 'yolov5_objectness', 'objectness'),
)

PROBE_ARCHITECTURES: tuple[str, ...] = tuple(
    spec.probe_architecture for spec in FAMILIES if spec.probe_architecture is not None
)


def probe_decoder(architecture: str) -> str:
    """``fused`` or ``objectness`` for a probe architecture; an unknown name is refused."""
    for spec in FAMILIES:
        if spec.probe_architecture == architecture:
            return spec.decoder
    raise ValueError(
        f'unknown probe architecture {architecture!r} (expected one of {PROBE_ARCHITECTURES})'
    )


def head_kind_of(head_module: object) -> str:
    """``one2one`` for the NMS-free head (YOLO26, YOLOv10), else ``nms``.

    Only ``one2one_cv2`` identifies the branch: ``model.end2end`` reads ``False`` for a loaded
    YOLO26 checkpoint, and 8.4 heads expose a ``one2one`` property whether or not the branch
    exists.
    """
    return ONE2ONE if hasattr(head_module, 'one2one_cv2') else NMS


def export_kwargs(head_kind: str, max_batch: int) -> dict[str, Any]:
    """The ``model.export`` arguments that select the served head, for ``format='onnx'``.

    ``max_batch`` is the Triton ``max_batch_size``: the export batch caps the dynamic axis, and
    an NMS export returns empty rows for images past it.
    """
    if head_kind == ONE2ONE:
        return {'nms': False, 'batch': max_batch}
    if head_kind == NMS:
        return {
            'nms': True,
            'agnostic_nms': True,
            'iou': NMS_IOU,
            'conf': NMS_CONF,
            'max_det': MAX_DET,
            'batch': max_batch,
        }
    raise ValueError(f'no export recipe for head kind {head_kind!r}')
