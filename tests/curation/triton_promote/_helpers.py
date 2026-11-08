"""Promoter factory, class map and promoted-model directory builder shared by the Triton promoter tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.services.training.triton_promote import TritonPromoter


if TYPE_CHECKING:
    from pathlib import Path


def _promoter(models_dir: Path) -> TritonPromoter:
    return TritonPromoter(triton_models_dir=models_dir, triton_http_url='http://unused:8000')


CLASS_MAP = {0: 'pickup', 1: 'sedan'}


@dataclass
class FakeRecorder:
    """Stands in for the database recorder: hands the promoter the labels it would have stored."""

    class_id_to_name: dict[int, str]
    class_remap: dict[str, Any] | None = None
    recorded: list[tuple[str, int]] = field(default_factory=list)
    withdrawn: int = 0

    @property
    def labels(self) -> list[str]:
        return [self.class_id_to_name[i] for i in sorted(self.class_id_to_name)]

    async def record(self, *, triton_name: str, triton_version: int) -> FakeRecorder:
        self.recorded.append((triton_name, triton_version))
        return self

    async def withdraw(self) -> None:
        self.withdrawn += 1


def _make_promoted_model_dir(scratch_models_dir: Path, name: str = 'op_smoke_unload_v1') -> Path:
    model_dir = scratch_models_dir / name
    v1 = model_dir / '1'
    v1.mkdir(parents=True)
    (v1 / 'model.onnx').write_bytes(b'THROWAWAY-SMOKE-WEIGHTS')
    (model_dir / 'config.pbtxt').write_text('# throwaway smoke config')
    import _fake_model_registry

    if _fake_model_registry.CURRENT is not None:
        _fake_model_registry.CURRENT.promote(name, run_job_id='job-abc')
    return model_dir


def write_float_image_onnx(path: Path, out_dims: list[int | str] | None = None) -> None:
    """A tiny valid ONNX with a float32 ``images`` input and the NMS-free ``[300, 6]`` output
    of the trainer's ``best.onnx`` (``out_dims`` replaces the per-item output dims)."""
    import onnx
    from onnx import TensorProto, helper

    node = helper.make_node('Identity', ['images'], ['output0'])
    graph = helper.make_graph(
        [node],
        'toy',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, ['b', 3, 8, 8])],
        [
            helper.make_tensor_value_info(
                'output0', TensorProto.FLOAT, ['b', *(out_dims or [300, 6])]
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
