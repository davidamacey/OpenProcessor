"""Read a trained detector ONNX's output and decide whether Triton may serve it as YOLO26.

The promoted model is registered with the one-to-one contract of
:mod:`src.services.training.yolo_triton_config`: a single output of ``(batch, 300, 6)`` rows of
``[x1, y1, x2, y2, score, class]``. An ONNX with the raw one-to-many head
(``(batch, 4 + nc, anchors)``) would be served and decoded as if it were NMS-free, so it is
refused here instead. Ultralytics exports the NMS-free head with ``nms=False``
(https://docs.ultralytics.com/modes/export/).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx

from src.services.training.yolo_triton_config import YOLO26_DET_DIMS, YOLO26_TOPK


if TYPE_CHECKING:
    from pathlib import Path


def nms_free_head_problem(onnx_path: Path) -> str | None:
    """``None`` when the ONNX has the single ``[300, 6]`` output, else the reason it does not."""
    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    outputs = list(graph.output)
    if len(outputs) != 1:
        return f'expected one output, found {[o.name for o in outputs]}'
    out = outputs[0]
    dims = [d.dim_value if d.dim_value > 0 else -1 for d in out.type.tensor_type.shape.dim[1:]]
    if dims != [YOLO26_TOPK, YOLO26_DET_DIMS]:
        return (
            f'output {out.name!r} has per-item dims {dims}, expected '
            f'[{YOLO26_TOPK}, {YOLO26_DET_DIMS}] (a raw one-to-many head: export with nms=False)'
        )
    return None
