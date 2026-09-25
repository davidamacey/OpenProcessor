"""Tests for the F-18 IO-rename fix in ``export/export_face_recognition.py``.

The buffalo_l ``arcface_w600k_r50.onnx`` checkpoint's raw graph keeps
whatever tensor names its PyTorch tracer happened to emit ('input.1' /
'683'), but ``models/arcface_w600k_r50/config.pbtxt`` and the clients
(``src/clients/triton_client.py``, ``fast_face_client.py``) hardcode
'input'/'output'. Before this fix Triton rejected every inference call:
"unexpected inference input 'input', allowed inputs are: input.1".

Everything here runs on a tiny stand-in graph -- same tensor-name pattern
as the real checkpoint, no GPU, no TensorRT, no downloaded weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


onnx = pytest.importorskip('onnx')
ort = pytest.importorskip('onnxruntime')

EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import export_face_recognition as arcface_export  # noqa: E402


def _write_stub_graph(
    path: Path,
    *,
    input_name: str,
    output_name: str,
    batch_dim: str | int,
) -> Path:
    """A 1-node graph mirroring buffalo_l's raw IO naming: [B,3,112,112] ->
    [B,512] via a single MatMul over a flattened/pooled input."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    size = arcface_export.INPUT_SIZE
    dim = arcface_export.EMBEDDING_DIM
    images = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, [batch_dim, 3, size, size]
    )
    embeddings = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [batch_dim, dim])
    weight = numpy_helper.from_array(np.zeros((3, dim), dtype=np.float32), name='proj')
    graph = helper.make_graph(
        [
            helper.make_node('ReduceMean', [input_name], ['pooled'], axes=[2, 3], keepdims=0),
            helper.make_node('MatMul', ['pooled', 'proj'], [output_name]),
        ],
        'arcface_stub',
        [images],
        [embeddings],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


class TestMakeBatchDynamicRenamesIo:
    def test_renames_raw_io_names_to_the_client_contract(self, tmp_path: Path) -> None:
        src = _write_stub_graph(
            tmp_path / 'raw.onnx', input_name='input.1', output_name='683', batch_dim=1
        )
        out_path = arcface_export.make_batch_dynamic(src)
        assert out_path != src

        model = onnx.load(str(out_path))
        assert model.graph.input[0].name == 'input'
        assert model.graph.output[0].name == 'output'
        # Every node reference must follow the rename, or the graph is
        # disconnected from its own input/output.
        node_names = {n for node in model.graph.node for n in (*node.input, *node.output)}
        assert 'input.1' not in node_names
        assert '683' not in node_names
        assert 'input' in node_names

    def test_renamed_graph_still_runs_and_agrees_with_the_original(self, tmp_path: Path) -> None:
        import numpy as np

        src = _write_stub_graph(
            tmp_path / 'raw.onnx', input_name='input.1', output_name='683', batch_dim=1
        )
        out_path = arcface_export.make_batch_dynamic(src)

        x = np.random.rand(1, 3, arcface_export.INPUT_SIZE, arcface_export.INPUT_SIZE).astype(
            np.float32
        )
        original = ort.InferenceSession(str(src), providers=['CPUExecutionProvider'])
        renamed = ort.InferenceSession(str(out_path), providers=['CPUExecutionProvider'])

        (expected,) = original.run(['683'], {'input.1': x})
        (actual,) = renamed.run(['output'], {'input': x})
        np.testing.assert_array_equal(actual, expected)

    def test_dynamic_batch_after_rename_accepts_other_batch_sizes(self, tmp_path: Path) -> None:
        import numpy as np

        src = _write_stub_graph(
            tmp_path / 'raw.onnx', input_name='input.1', output_name='683', batch_dim=1
        )
        out_path = arcface_export.make_batch_dynamic(src)
        session = ort.InferenceSession(str(out_path), providers=['CPUExecutionProvider'])
        for batch in (1, 3, 8):
            x = np.random.rand(batch, 3, arcface_export.INPUT_SIZE, arcface_export.INPUT_SIZE)
            (out,) = session.run(['output'], {'input': x.astype(np.float32)})
            assert out.shape == (batch, arcface_export.EMBEDDING_DIM)

    def test_already_canonical_and_dynamic_graph_is_left_untouched(self, tmp_path: Path) -> None:
        src = _write_stub_graph(
            tmp_path / 'already_good.onnx',
            input_name='input',
            output_name='output',
            batch_dim='batch',
        )
        out_path = arcface_export.make_batch_dynamic(src)
        assert out_path == src

    def test_config_pbtxt_uses_the_same_canonical_names(self, tmp_path: Path) -> None:
        """The rendered config.pbtxt must describe the graph make_batch_dynamic
        actually produces -- not a second, independently-drifting source of
        truth for the tensor contract."""
        plan_path = tmp_path / 'models' / 'arcface_w600k_r50' / '1' / 'model.plan'
        config_path = arcface_export.create_triton_config(plan_path)
        body = config_path.read_text()
        assert f'name: "{arcface_export.CANONICAL_INPUT_NAME}"' in body
        assert f'name: "{arcface_export.CANONICAL_OUTPUT_NAME}"' in body
