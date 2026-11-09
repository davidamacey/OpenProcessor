"""What export a detection head needs, in one place the API and the trainer both read.

Standard library only: the trainer image carries ``src/domain`` and nothing of the web stack.

Ultralytics' ``nms`` export argument picks the head that is served
(https://docs.ultralytics.com/modes/export/): ``True`` embeds NMS for a head that needs it,
``False`` selects the NMS-free one-to-one head, and the default (``None``) exports the raw
one-to-many head, which this repository never serves. Both served exports end in the same
``[batch, 300, 6]`` tensor (``x1, y1, x2, y2, score, class``).
"""

from __future__ import annotations

from typing import Any


ONE2ONE = 'one2one'
NMS = 'nms'

MAX_DET = 300
NMS_IOU = 0.7
NMS_CONF = 0.25


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
