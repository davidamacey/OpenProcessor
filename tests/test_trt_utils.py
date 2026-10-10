"""Tests for the shared TensorRT builder helpers in ``export/trt_utils.py``.

F-17 round 2 (fresh-start E2E findings 2026-09-25): ``export_paddleocr_rec.py``
used to build a raw ``NetworkDefinitionCreationFlag.EXPLICIT_BATCH`` lookup
(removed in TRT 11) and wrote engines straight to the destination
``model.plan``, deleting the old one first -- a failed rebuild left the
model with no plan at all. ``create_explicit_network`` and
``atomic_write_plan`` are the shared fix; these tests fake ``tensorrt``
itself so they run without a GPU or the real package.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))


class _FakeEngine:
    """Stands in for a deserialized ``ICudaEngine`` -- truthy, no I/O."""


def _install_fake_tensorrt(
    monkeypatch: pytest.MonkeyPatch,
    *,
    explicit_batch_flag_exists: bool,
    deserializes_ok: bool,
    fp16_builder_flag_exists: bool = True,
):
    fake_runtime = MagicMock()
    fake_runtime.deserialize_cuda_engine.return_value = _FakeEngine() if deserializes_ok else None

    ns_kwargs = {}
    if explicit_batch_flag_exists:
        ns_kwargs['EXPLICIT_BATCH'] = 0

    builder_flag_kwargs = {}
    if fp16_builder_flag_exists:
        builder_flag_kwargs['FP16'] = 1

    fake_trt = types.SimpleNamespace(
        Logger=lambda _level=None: MagicMock(),
        init_libnvinfer_plugins=lambda _logger, _ns: None,
        Runtime=lambda _logger: fake_runtime,
        BuilderFlag=types.SimpleNamespace(**builder_flag_kwargs),
        NetworkDefinitionCreationFlag=types.SimpleNamespace(**ns_kwargs),
    )
    fake_trt.Logger.ERROR = 0
    monkeypatch.setitem(sys.modules, 'tensorrt', fake_trt)
    monkeypatch.delitem(sys.modules, 'trt_utils', raising=False)
    return fake_trt


class TestCreateExplicitNetwork:
    def test_passes_the_flag_when_present_trt10(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_tensorrt(monkeypatch, explicit_batch_flag_exists=True, deserializes_ok=True)
        import trt_utils

        builder = MagicMock()
        trt_utils.create_explicit_network(builder)

        builder.create_network.assert_called_once_with(1)

    def test_no_attribute_error_when_flag_is_removed_trt11(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_tensorrt(monkeypatch, explicit_batch_flag_exists=False, deserializes_ok=True)
        import trt_utils

        builder = MagicMock()
        trt_utils.create_explicit_network(builder)

        builder.create_network.assert_called_once_with(0)


class TestAtomicWritePlan:
    def test_writes_a_valid_engine_and_returns_the_plan_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_tensorrt(monkeypatch, explicit_batch_flag_exists=True, deserializes_ok=True)
        import trt_utils

        plan_path = tmp_path / 'models' / 'some_model' / '1' / 'model.plan'
        result = trt_utils.atomic_write_plan(b'a-real-engine', plan_path)

        assert result == plan_path
        assert plan_path.read_bytes() == b'a-real-engine'
        # No leftover temp file next to it.
        assert sorted(p.name for p in plan_path.parent.iterdir()) == ['model.plan']

    def test_rejects_an_engine_that_fails_to_deserialize_and_keeps_the_old_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_tensorrt(monkeypatch, explicit_batch_flag_exists=True, deserializes_ok=False)
        import trt_utils

        plan_path = tmp_path / 'model.plan'
        plan_path.write_bytes(b'old-working-engine')

        with pytest.raises(ValueError, match='deserialize'):
            trt_utils.atomic_write_plan(b'garbage', plan_path)

        assert plan_path.read_bytes() == b'old-working-engine'

    def test_rejects_empty_bytes_without_touching_an_existing_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_tensorrt(monkeypatch, explicit_batch_flag_exists=True, deserializes_ok=True)
        import trt_utils

        plan_path = tmp_path / 'model.plan'
        plan_path.write_bytes(b'old-working-engine')

        with pytest.raises(ValueError, match='empty'):
            trt_utils.atomic_write_plan(b'', plan_path)

        assert plan_path.read_bytes() == b'old-working-engine'


class TestBakeFp16OnnxOrFallback:
    """F-16: PE's build_pe_trt.sh shells out to trtexec instead of driving
    the TRT Python builder, so it can't wrap bake_fp16_onnx in a Python
    try/except the way export_face_recognition.py / export_scrfd.py /
    export_mobileclip_image_encoder.py do. bake_fp16_onnx_or_fallback is
    the never-raises wrapper build_pe_trt.sh's CLI shim calls instead.

    onnx + onnxconverter-common are real (installed) here -- only
    ``tensorrt`` itself is faked, so the real FP16 graph rewrite runs.
    """

    @staticmethod
    def _tiny_onnx(path: Path) -> Path:
        """A minimal two-node FP32 graph -- ReduceMean + MatMul, matching
        the stand-in used by tests/test_pe_image_encoder_export.py."""
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        images = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 8, 8])
        embeddings = helper.make_tensor_value_info('image_embeddings', TensorProto.FLOAT, [1, 4])
        weight = numpy_helper.from_array(np.zeros((3, 4), dtype=np.float32), name='proj')
        graph = helper.make_graph(
            [
                helper.make_node('ReduceMean', ['images'], ['pooled'], axes=[2, 3], keepdims=0),
                helper.make_node('MatMul', ['pooled', 'proj'], ['image_embeddings']),
            ],
            'tiny',
            [images],
            [embeddings],
            initializer=[weight],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
        return path

    def test_trt11_typed_build_bakes_fp16_into_the_graph(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No BuilderFlag.FP16 (TRT >= 11) -- must actually rewrite the
        graph to FP16 weights and report used_fp16=True."""
        pytest.importorskip('onnxconverter_common')
        _install_fake_tensorrt(
            monkeypatch,
            explicit_batch_flag_exists=False,
            deserializes_ok=True,
            fp16_builder_flag_exists=False,
        )
        import onnx
        import trt_utils

        src = self._tiny_onnx(tmp_path / 'pe.onnx')
        out_path = tmp_path / 'pe.fp16.onnx'

        resolved, used_fp16 = trt_utils.bake_fp16_onnx_or_fallback(src, out_path)

        assert used_fp16 is True
        assert resolved == out_path
        assert resolved.exists()
        fp16_model = onnx.load(str(resolved))
        weight_init = next(i for i in fp16_model.graph.initializer if i.name == 'proj')
        assert weight_init.data_type == onnx.TensorProto.FLOAT16
        # keep_io_types=True -- graph I/O must stay FP32 so Triton's
        # TYPE_FP32 config and existing clients are unaffected.
        assert fp16_model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        assert fp16_model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    def test_pre_trt11_classic_fp16_flag_is_a_no_op(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BuilderFlag.FP16 present (pre-TRT-11) -- enable_fp16 handles
        precision via the builder flag; baking would be redundant."""
        _install_fake_tensorrt(
            monkeypatch,
            explicit_batch_flag_exists=False,
            deserializes_ok=True,
            fp16_builder_flag_exists=True,
        )
        import trt_utils

        src = self._tiny_onnx(tmp_path / 'pe.onnx')
        resolved, used_fp16 = trt_utils.bake_fp16_onnx_or_fallback(src)

        assert used_fp16 is True
        assert resolved == src

    def test_a_bake_failure_falls_back_to_fp32_instead_of_raising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing/unreadable ONNX (or any other bake_fp16_onnx failure)
        must never abort the build script -- an FP32 engine is always
        buildable from the original graph."""
        _install_fake_tensorrt(
            monkeypatch,
            explicit_batch_flag_exists=False,
            deserializes_ok=True,
            fp16_builder_flag_exists=False,
        )
        import trt_utils

        missing = tmp_path / 'does_not_exist.onnx'
        resolved, used_fp16 = trt_utils.bake_fp16_onnx_or_fallback(missing)

        assert used_fp16 is False
        assert resolved == missing


class TestTrtUtilsCli:
    """F-16: build_pe_trt.sh (bash) invokes `python trt_utils.py <onnx>
    [out]` to run the bake step without embedding Python in the shell
    script. Prints 'fp16'/'fp32' to stderr, the resolved path (and only
    that) to stdout -- build_pe_trt.sh captures stdout for the path it
    hands to trtexec."""

    def test_cli_prints_status_to_stderr_and_path_to_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip('onnxconverter_common')
        _install_fake_tensorrt(
            monkeypatch,
            explicit_batch_flag_exists=False,
            deserializes_ok=True,
            fp16_builder_flag_exists=False,
        )
        import trt_utils

        src = TestBakeFp16OnnxOrFallback._tiny_onnx(tmp_path / 'pe.onnx')
        out_path = tmp_path / 'pe.fp16.onnx'

        import contextlib
        import io

        stdout, stderr = io.StringIO(), io.StringIO()
        monkeypatch.setattr(sys, 'argv', ['trt_utils.py', str(src), str(out_path)])
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            runpy_globals: dict = {'__name__': '__main__'}
            exec(
                compile(Path(trt_utils.__file__).read_text(), trt_utils.__file__, 'exec'),
                runpy_globals,
            )

        assert stderr.getvalue().strip() == 'fp16'
        assert stdout.getvalue().strip() == str(out_path)


def _einsum_rope_onnx(path: Path) -> Path:
    """FP32 graph shaped like the PE image encoder's rotary-embedding
    prologue: a float32 ``Constant`` (the frequency table) and an explicit
    float32 ``Cast`` feeding an ``Einsum``, whose result goes through Cos."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4, 6])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 6, 8])
    freqs = numpy_helper.from_array(np.linspace(0.1, 3.0, 8, dtype=np.float32))
    graph = helper.make_graph(
        [
            helper.make_node('Add', ['x', 'x'], ['pos']),
            helper.make_node('Cast', ['pos'], ['pos_f32'], to=TensorProto.FLOAT),
            helper.make_node('Constant', [], ['freqs'], value=freqs),
            helper.make_node(
                'Einsum', ['pos_f32', 'freqs'], ['angles'], equation='..., f -> ... f'
            ),
            helper.make_node('Cos', ['angles'], ['y']),
        ],
        'rope',
        [x],
        [y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _mixed_float_inputs(model, ops: frozenset[str]) -> list[str]:
    """Names of ``ops`` nodes whose float inputs are not all one element type
    (the condition TensorRT's IEinsumLayer / ElementWiseOperation reject at
    ONNX parse). Types are re-derived from the nodes, not from ``value_info``:
    the converter rewrites those entries to FP16 even where an explicit Cast
    keeps the tensor FP32, and inference trusts existing entries."""
    import onnx

    stripped = onnx.ModelProto()
    stripped.CopyFrom(model)
    del stripped.graph.value_info[:]
    inferred = onnx.shape_inference.infer_shapes(stripped)
    types = {
        vi.name: vi.type.tensor_type.elem_type
        for vi in [*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output]
    }
    for init in inferred.graph.initializer:
        types[init.name] = init.data_type
    floats = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}
    return [
        node.name or node.output[0]
        for node in inferred.graph.node
        if node.op_type in ops and len({types.get(n) for n in node.input} & floats) > 1
    ]


def _install_trt11(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_tensorrt(
        monkeypatch,
        explicit_batch_flag_exists=False,
        deserializes_ok=True,
        fp16_builder_flag_exists=False,
    )


class TestBakeFp16OnnxEinsum:
    """PE's FP16 build failed at ONNX parse under TRT 11.1: the baked graph
    fed an Einsum one FP16 input (the converted Constant) and one FP32 input
    (an explicit FP32 Cast), and IEinsumLayer needs a single input type."""

    def test_baked_einsum_inputs_share_one_type(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip('onnxconverter_common')
        _install_trt11(monkeypatch)
        import onnx
        import trt_utils

        src = _einsum_rope_onnx(tmp_path / 'rope.onnx')
        assert _mixed_float_inputs(onnx.load(str(src)), frozenset({'Einsum'})) == []

        baked = trt_utils.bake_fp16_onnx(src, tmp_path / 'rope.fp16.onnx')

        model = onnx.load(str(baked))
        onnx.checker.check_model(model)
        assert _mixed_float_inputs(model, frozenset({'Einsum'})) == []
        # keep_io_types=True: the graph boundary stays FP32.
        assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT


def _fp32_cast_island_onnx(path: Path) -> Path:
    """FP32 graph where an explicit ``Cast(to=FLOAT)`` (PyTorch's ``.float()``)
    produces a table that is multiplied with, and concatenated to, the
    FP16-converted activations -- the pattern behind the PE encoder's
    ElementWiseOperation parse error."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 16])
    weight = numpy_helper.from_array(np.ones((8, 8), dtype=np.float32), name='w')
    table = numpy_helper.from_array(np.full((4, 8), 0.5, dtype=np.float32), name='table')
    graph = helper.make_graph(
        [
            helper.make_node('MatMul', ['x', 'w'], ['h']),
            helper.make_node('Cast', ['table'], ['table_f32'], to=TensorProto.FLOAT),
            helper.make_node('Cos', ['table_f32'], ['cos']),
            helper.make_node('Mul', ['h', 'cos'], ['rotated']),
            helper.make_node('Concat', ['rotated', 'cos'], ['y'], axis=1),
        ],
        'island',
        [x],
        [y],
        initializer=[weight, table],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


class TestBakeFp16OnnxReconcilesFloatTypes:
    SAME_TYPE_OPS = frozenset({'Mul', 'Concat', 'MatMul', 'Einsum', 'Add'})

    def test_an_explicit_fp32_cast_does_not_leave_mixed_operands(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip('onnxconverter_common')
        _install_trt11(monkeypatch)
        import onnx
        import trt_utils

        src = _fp32_cast_island_onnx(tmp_path / 'island.onnx')
        assert _mixed_float_inputs(onnx.load(str(src)), self.SAME_TYPE_OPS) == []

        baked = trt_utils.bake_fp16_onnx(src, tmp_path / 'island.fp16.onnx')

        model = onnx.load(str(baked))
        onnx.checker.check_model(model)
        assert _mixed_float_inputs(model, self.SAME_TYPE_OPS) == []
        weight = next(i for i in model.graph.initializer if i.name == 'w')
        assert weight.data_type == onnx.TensorProto.FLOAT16  # still a real FP16 graph
        assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    def test_the_baked_graph_computes_the_same_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ort = pytest.importorskip('onnxruntime')
        pytest.importorskip('onnxconverter_common')
        import numpy as np
        import trt_utils

        _install_trt11(monkeypatch)
        src = _fp32_cast_island_onnx(tmp_path / 'island.onnx')
        baked = trt_utils.bake_fp16_onnx(src, tmp_path / 'island.fp16.onnx')
        x = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)

        def run(path: Path) -> np.ndarray:
            session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
            return session.run(None, {'x': x})[0]

        np.testing.assert_allclose(run(baked), run(src), rtol=2e-2, atol=2e-2)
