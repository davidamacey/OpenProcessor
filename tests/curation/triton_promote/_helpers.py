"""Promoter factory, class map and promoted-model directory builder shared by the Triton promoter tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.training.triton_promote import TritonPromoter


if TYPE_CHECKING:
    from pathlib import Path


def _promoter(models_dir: Path) -> TritonPromoter:
    return TritonPromoter(triton_models_dir=models_dir, triton_http_url='http://unused:8000')


CLASS_MAP = {0: 'pickup', 1: 'sedan'}


def _make_promoted_model_dir(scratch_models_dir: Path, name: str = 'op_smoke_unload_v1') -> Path:
    model_dir = scratch_models_dir / name
    v1 = model_dir / '1'
    v1.mkdir(parents=True)
    (v1 / 'model.onnx').write_bytes(b'THROWAWAY-SMOKE-WEIGHTS')
    (model_dir / 'config.pbtxt').write_text('# throwaway smoke config')
    (model_dir / 'promote.json').write_text('{"job_id": "job-abc"}')
    return model_dir


def write_float_image_onnx(path: Path) -> None:
    """A tiny valid ONNX with a float32 ``images`` input, like the trainer's ``best.onnx``."""
    import onnx
    from onnx import TensorProto, helper

    node = helper.make_node('Identity', ['images'], ['output0'])
    graph = helper.make_graph(
        [node],
        'toy',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, ['b', 3, 8, 8])],
        [helper.make_tensor_value_info('output0', TensorProto.FLOAT, ['b', 3, 8, 8])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
