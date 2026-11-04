"""The YOLO26 export selects the NMS-free head and refuses any other (issue 447).

Documented argument: ``nms=False`` ("selects the NMS-free head when available",
https://docs.ultralytics.com/modes/export/). The synthetic-graph tests run offline; the
``families`` test exports the official ``yolo26n.pt`` with the production function.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import onnx
import pytest
from onnx import TensorProto, helper


if TYPE_CHECKING:
    from collections.abc import Iterator

EXPORT_DIR = Path(__file__).resolve().parent.parent / 'export'


@pytest.fixture
def export_yolo26(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """``export/export_yolo26.py`` imported without TensorRT (the export image has it)."""
    monkeypatch.syspath_prepend(str(EXPORT_DIR))
    stub = ModuleType('tensorrt')
    monkeypatch.setitem(sys.modules, 'tensorrt', stub)
    for name in ('trt_utils', 'export_yolo26'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        yield importlib.import_module('export_yolo26')
    finally:
        sys.modules.pop('export_yolo26', None)
        sys.modules.pop('trt_utils', None)


def _write_onnx(path: Path, out_dims: list[int | str], *, outputs: int = 1) -> Path:
    outs = [
        helper.make_tensor_value_info(f'output{i}', TensorProto.FLOAT, out_dims)
        for i in range(outputs)
    ]
    nodes = [helper.make_node('Identity', ['images'], [o.name]) for o in outs]
    graph = helper.make_graph(
        nodes,
        'head',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, out_dims)],
        outs,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


def test_inspect_refuses_the_raw_one_to_many_head(export_yolo26: ModuleType, tmp_path: Path):
    raw = _write_onnx(tmp_path / 'raw.onnx', ['batch', 84, 'anchors'])
    with pytest.raises(RuntimeError, match=r'raw one-to-many head.*\[84, -1\]|\[84, -1\].*raw'):
        export_yolo26.inspect_fused_output(raw)


def test_inspect_accepts_the_fused_300_by_6_head(export_yolo26: ModuleType, tmp_path: Path):
    fused = _write_onnx(tmp_path / 'fused.onnx', ['batch', 300, 6])
    assert export_yolo26.inspect_fused_output(fused) == ('output0', [300, 6])


def test_inspect_still_refuses_several_outputs(export_yolo26: ModuleType, tmp_path: Path):
    many = _write_onnx(tmp_path / 'many.onnx', ['batch', 300, 6], outputs=2)
    with pytest.raises(RuntimeError, match='single fused output'):
        export_yolo26.inspect_fused_output(many)


@pytest.mark.families
def test_official_yolo26n_exports_the_nms_free_head(
    export_yolo26: ModuleType, op_weights, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from ultralytics import YOLO

    monkeypatch.setattr(export_yolo26, 'EXPORT_DIR', tmp_path)
    model = YOLO(str(op_weights('yolo26n.pt')))
    onnx_path = export_yolo26.export_fused_onnx(model, {'triton_name': 'yolo26_t', 'max_batch': 2})

    assert export_yolo26.inspect_fused_output(onnx_path) == ('output0', [300, 6])
    assert 'TopK' in {n.op_type for n in onnx.load(str(onnx_path)).graph.node}


@pytest.mark.families
def test_a_raw_head_export_is_refused(export_yolo26: ModuleType, op_weights, tmp_path: Path):
    from ultralytics import YOLO

    model = YOLO(str(op_weights('yolo26n.pt')))
    raw = Path(model.export(format='onnx', imgsz=640, dynamic=True, device='cpu', nms=None))
    with pytest.raises(RuntimeError, match='raw one-to-many head'):
        export_yolo26.inspect_fused_output(raw)
