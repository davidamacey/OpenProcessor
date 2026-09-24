"""Tests for the PE-Core-L14-336 text-encoder export path.

``export/export_pe_text_encoder.py`` produces the graph
``src/clients/pe_encoder.py`` loads in-process (ONNX Runtime) or through
Triton (``pe_text_encoder``). Nothing in ``src/`` imports the exporter, so
the shared tensor contract is pinned here.

Mirrors ``tests/test_pe_image_encoder_export.py``: the config renderer and
CLI are pure, the ONNX QA gate runs against tiny stand-in graphs on the
ONNX Runtime CPU provider, and the parity orchestration runs over numpy
stand-ins. The real export (weights + ``torch.export``) is exercised by
running the script in the API image; its real-graph check lives in
``tests/curation/test_pe_text_backends.py::test_real_onnx_matches_pytorch_eager``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.clients import pe_encoder as client


EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import export_pe_text_encoder as text_export  # noqa: E402


# =============================================================================
# The contract with src/clients/pe_encoder.py
# =============================================================================


class TestClientContract:
    def test_exporter_constants_match_the_client(self) -> None:
        assert text_export.TRITON_MODEL_NAME == client.PE_TEXT_TRITON_MODEL
        assert text_export.INPUT_TENSOR == client.PE_TEXT_INPUT
        assert text_export.OUTPUT_TENSOR == client.PE_TEXT_OUTPUT
        assert text_export.CONTEXT_LENGTH == client.PE_TEXT_CONTEXT_LENGTH
        assert text_export.EMBEDDING_DIM == client.PE_EMBEDDING_DIM
        assert text_export.PE_VARIANT == client.PE_TEXT_CHECKPOINT

    def test_trim_to_eot_matches_the_client(self) -> None:
        tokens = np.zeros((3, 32), dtype=np.int64)
        tokens[0, :3] = [49406, 320, 49407]
        tokens[1, :6] = [49406, 320, 1125, 539, 320, 49407]
        np.testing.assert_array_equal(
            text_export.trim_to_eot(tokens), client.trim_text_tokens(tokens)
        )
        assert text_export.trim_to_eot(tokens).shape == (3, 6)
        assert text_export.EOT_TOKEN_ID == 49407

    def test_default_onnx_destination_is_the_client_default(self) -> None:
        args = text_export.build_parser().parse_args([])
        assert str(args.onnx_out) == client.DEFAULT_PE_TEXT_ONNX_PATH

    def test_rendered_config_declares_the_client_tensors(self) -> None:
        out = text_export.render_config(text_export.PETextTritonConfig())
        assert f'name: "{client.PE_TEXT_TRITON_MODEL}"' in out
        assert 'platform: "onnxruntime_onnx"' in out
        assert f'name: "{client.PE_TEXT_INPUT}"' in out
        assert f'name: "{client.PE_TEXT_OUTPUT}"' in out
        assert 'data_type: TYPE_INT64' in out
        # The client trims padding, so the token axis must be variable.
        assert 'dims: [ -1 ]' in out
        assert f'dims: [ {client.PE_EMBEDDING_DIM} ]' in out

    def test_committed_template_matches_the_renderer(self) -> None:
        committed = Path(__file__).resolve().parents[1] / 'models' / 'pe_text_encoder'
        body = (committed / 'config.pbtxt').read_text()
        assert body == text_export.render_config(text_export.PETextTritonConfig())


# =============================================================================
# render_config / validation
# =============================================================================


class TestRenderConfig:
    def test_cpu_instances_by_default(self) -> None:
        out = text_export.render_config(text_export.PETextTritonConfig())
        assert 'kind: KIND_CPU' in out
        assert 'gpus:' not in out

    def test_gpu_instances(self) -> None:
        out = text_export.render_config(
            text_export.PETextTritonConfig(kind='KIND_GPU', gpu_ids=(0, 2), instance_count=2)
        )
        assert 'kind: KIND_GPU' in out
        assert 'gpus: [ 0, 2 ]' in out
        assert 'count: 2' in out

    def test_preferred_ladder_is_bounded(self) -> None:
        out = text_export.render_config(text_export.PETextTritonConfig(max_batch_size=8))
        assert 'max_batch_size: 8' in out
        assert 'preferred_batch_size: [ 4, 8 ]' in out
        assert text_export.preferred_batch_sizes(2) == [2]

    @pytest.mark.parametrize(
        ('field', 'value'),
        [
            ('max_batch_size', 0),
            ('context_length', 0),
            ('embedding_dim', -1),
            ('instance_count', 0),
            ('max_queue_delay_us', -1),
            ('model_name', ''),
            ('input_name', ''),
            ('kind', 'KIND_TPU'),
        ],
    )
    def test_rejects_invalid_field(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field.split('_')[0]):
            text_export.PETextTritonConfig(**{field: value})

    def test_gpu_kind_requires_gpu_ids(self) -> None:
        with pytest.raises(ValueError, match='gpu_ids'):
            text_export.PETextTritonConfig(kind='KIND_GPU', gpu_ids=())


# =============================================================================
# CLI
# =============================================================================


class TestCli:
    def test_defaults(self) -> None:
        args = text_export.build_parser().parse_args([])
        assert args.install_triton is False
        assert args.config_only is False
        assert args.kind == 'cpu'
        assert args.verify_checkpoint is True
        assert args.checkpoint_path is None
        assert args.parity_threshold == text_export.PARITY_MIN_COSINE == 0.9999

    def test_config_from_args(self) -> None:
        args = text_export.build_parser().parse_args(
            ['--kind', 'gpu', '--gpus', '2', '--max-batch', '16', '--instance-count', '3']
        )
        cfg = text_export.config_from_args(args)
        assert (cfg.kind, cfg.gpu_ids, cfg.max_batch_size, cfg.instance_count) == (
            'KIND_GPU',
            (2,),
            16,
            3,
        )

    def test_config_only_writes_the_model_repo_entry(self, tmp_path: Path) -> None:
        assert text_export.main(['--config-only', '--models-dir', str(tmp_path)]) == 0
        body = (tmp_path / client.PE_TEXT_TRITON_MODEL / 'config.pbtxt').read_text()
        assert 'platform: "onnxruntime_onnx"' in body

    @pytest.mark.parametrize(
        'argv',
        [
            ['--max-batch', '0'],
            ['--opset', '0'],
            ['--parity-threshold', '1.5'],
            ['--parity-threshold', '0'],
        ],
    )
    def test_invalid_arguments_exit_2_before_loading_anything(
        self, tmp_path: Path, argv: list[str]
    ) -> None:
        assert text_export.main([*argv, '--config-only', '--models-dir', str(tmp_path)]) == 2
        assert not (tmp_path / client.PE_TEXT_TRITON_MODEL).exists()

    def test_install_triton_copies_the_graph(self, tmp_path: Path) -> None:
        onnx_file = tmp_path / 'text.onnx'
        onnx_file.write_bytes(b'graph-bytes')
        target = text_export.install_triton_model(
            onnx_file, tmp_path / 'models', text_export.PETextTritonConfig()
        )
        assert target == tmp_path / 'models' / client.PE_TEXT_TRITON_MODEL / '1' / 'model.onnx'
        assert target.read_bytes() == b'graph-bytes'
        assert (target.parent.parent / 'config.pbtxt').is_file()


# =============================================================================
# ONNX QA gate against real (tiny) graphs
# =============================================================================


def _stub_graph(path: Path, *, batch: str | int = 'batch', tokens: str | int = 'tokens') -> Path:
    """``text_tokens[batch, tokens] int64 -> text_embeddings[batch, 1024]``.

    A static ``tokens`` axis gets a Reshape to the constant width, so ORT
    genuinely rejects any other length (as a legacy-traced graph does).
    """
    pytest.importorskip('onnxruntime')
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inp = helper.make_tensor_value_info('text_tokens', TensorProto.INT64, [batch, tokens])
    out = helper.make_tensor_value_info(
        'text_embeddings', TensorProto.FLOAT, [batch, client.PE_EMBEDDING_DIM]
    )
    inits = [
        numpy_helper.from_array(
            np.ones((1, client.PE_EMBEDDING_DIM), dtype=np.float32), name='proj'
        ),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name='axes'),
    ]
    nodes = []
    src = 'text_tokens'
    if isinstance(tokens, int):
        inits.append(numpy_helper.from_array(np.array([-1, tokens], dtype=np.int64), name='shape'))
        nodes.append(helper.make_node('Reshape', ['text_tokens', 'shape'], ['fixed']))
        src = 'fixed'
    nodes += [
        helper.make_node('Cast', [src], ['f'], to=TensorProto.FLOAT),
        helper.make_node('ReduceSum', ['f', 'axes'], ['s'], keepdims=1),
        helper.make_node('MatMul', ['s', 'proj'], ['text_embeddings']),
    ]
    graph = helper.make_graph(nodes, 'stub', [inp], [out], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


class TestValidateOnnx:
    def test_fully_dynamic_graph_satisfies_the_contract(self, tmp_path: Path) -> None:
        report = text_export.validate_onnx(_stub_graph(tmp_path / 'ok.onnx'))
        assert report.skipped is False
        assert report.input_name == 'text_tokens'
        assert report.input_dtype == 'tensor(int64)'
        assert report.embedding_name == 'text_embeddings'
        assert report.embedding_dim == client.PE_EMBEDDING_DIM
        assert report.dynamic_batch is True
        assert report.dynamic_tokens is True
        assert text_export.check_client_contract(report, text_export.PETextTritonConfig()) == []

    def test_static_token_axis_is_flagged(self, tmp_path: Path) -> None:
        """What the legacy TorchScript tracer produces: works only at 32 tokens."""
        report = text_export.validate_onnx(_stub_graph(tmp_path / 's.onnx', tokens=32))
        assert report.dynamic_batch is True
        assert report.dynamic_tokens is False
        problems = text_export.check_client_contract(report, text_export.PETextTritonConfig())
        assert any('token axis is static' in p for p in problems)

    def test_static_batch_axis_is_flagged(self, tmp_path: Path) -> None:
        report = text_export.validate_onnx(_stub_graph(tmp_path / 'b.onnx', batch=1))
        assert report.dynamic_batch is False
        problems = text_export.check_client_contract(report, text_export.PETextTritonConfig())
        assert any('leading axis is static' in p for p in problems)


class TestCheckClientContract:
    @staticmethod
    def _good() -> text_export.OnnxReport:
        return text_export.OnnxReport(
            input_name='text_tokens',
            input_dtype='tensor(int64)',
            input_shape=['batch', 'tokens'],
            outputs=[('text_embeddings', ['batch', 1024])],
            embedding_name='text_embeddings',
            embedding_dim=1024,
            dynamic_batch=True,
            dynamic_tokens=True,
        )

    def test_good_report(self) -> None:
        assert (
            text_export.check_client_contract(self._good(), text_export.PETextTritonConfig()) == []
        )

    def test_hf_style_names_and_dtype_are_flagged(self) -> None:
        report = self._good()
        report.input_name = 'input_ids'
        report.input_dtype = 'tensor(int32)'
        report.embedding_name = 'text_embeds'
        problems = text_export.check_client_contract(report, text_export.PETextTritonConfig())
        assert len(problems) == 3

    def test_wrong_static_context_and_width_are_flagged(self) -> None:
        report = self._good()
        report.input_shape = ['batch', 77]
        report.embedding_dim = 512
        problems = text_export.check_client_contract(report, text_export.PETextTritonConfig())
        assert any('77' in p for p in problems)
        assert any('512' in p for p in problems)

    def test_skipped_probe_asserts_nothing(self) -> None:
        report = text_export.OnnxReport(skipped=True)
        assert text_export.check_client_contract(report, text_export.PETextTritonConfig()) == []


# =============================================================================
# Parity orchestration
# =============================================================================


class TestParity:
    def test_identical_outputs_pass(self) -> None:
        rng = np.random.default_rng(0)
        table = rng.normal(size=(50000, 8))

        def fn(tokens: np.ndarray) -> np.ndarray:
            return table[tokens].sum(axis=1)

        batches = [np.array([[1, 2, 3]]), np.array([[4, 5, 6], [7, 8, 9]])]
        report = text_export.run_parity(fn, fn, batches)
        assert report.n == 3
        assert report.batch_sizes == (1, 2)
        assert report.min_cosine == pytest.approx(1.0)
        assert report.passes()

    def test_a_single_divergent_row_fails_the_gate(self) -> None:
        def ref(tokens: np.ndarray) -> np.ndarray:
            return np.ones((tokens.shape[0], 4))

        def cand(tokens: np.ndarray) -> np.ndarray:
            out = np.ones((tokens.shape[0], 4))
            out[-1, 0] = 1.1  # cosine ~0.9989
            return out

        report = text_export.run_parity(ref, cand, [np.zeros((3, 5), dtype=np.int64)])
        assert report.min_cosine < 0.9999
        assert report.max_abs_diff == pytest.approx(0.1)
        assert not report.passes()

    def test_empty_run_never_passes(self) -> None:
        assert not text_export.ParityReport().passes()

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match='shape mismatch'):
            text_export.compare_embeddings(np.ones((2, 4)), np.ones((2, 5)))

    def test_parity_batches_cover_batch_1_and_8(self) -> None:
        def tok(prompts: list[str]) -> np.ndarray:
            return np.arange(len(prompts) * 4).reshape(len(prompts), 4)

        batches = text_export.parity_token_batches(tok)
        n = len(text_export.PARITY_PROMPTS)
        sizes = [b.shape[0] for b in batches]
        assert sizes[:n] == [1] * n
        assert sum(sizes[n:]) == n
        assert max(sizes) == 8

    def test_prompts_include_one_longer_than_the_context(self) -> None:
        """Exercises the tokenizer's truncation path in the parity gate."""
        assert max(len(p.split()) for p in text_export.PARITY_PROMPTS) > text_export.CONTEXT_LENGTH
