"""Tests for the PE-Core-L14-336 image-encoder export path.

``export/export_pe_image_encoder.py`` is the only way a fresh deployment can
build the ``pe_image_encoder`` Triton model that ``src/clients/pe_encoder.py``
requires. Nothing in ``src/`` imports the exporter, so a drift between the
rendered ``config.pbtxt`` and the client's hardcoded tensor contract would
otherwise only surface as a runtime Triton inference error.

Everything here runs without a GPU, without TensorRT and without downloading
model weights: the config renderer is pure, the CLI is exercised through its
``--config-only`` mode, and the ONNX probe runs against a two-node stand-in
graph on the ONNX Runtime CPU provider. The heavy halves (loading real PE
weights through ``torch.onnx.export``, and the ``trtexec`` engine build) are
covered only by a real export run against real hardware.

Import convention follows ``tests/curation/test_trainer_protocol.py``: the
export scripts ship as standalone entry points rather than an installed
package, so the directory is put on ``sys.path`` explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.clients.pe_encoder import PE_EMBEDDING_DIM, PE_IMAGE_MODEL
from src.services.detection.pe_preprocess import PE_SIZE


EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import export_pe_image_encoder as pe_export  # noqa: E402


# =============================================================================
# The contract with src/clients/pe_encoder.py
# =============================================================================


class TestClientContract:
    """The rendered config must match what PEEncoder actually sends/reads."""

    def test_defaults_match_the_client_constants(self) -> None:
        cfg = pe_export.PETritonConfig()
        assert cfg.model_name == PE_IMAGE_MODEL
        assert cfg.embedding_dim == PE_EMBEDDING_DIM
        assert cfg.image_size == PE_SIZE
        # PEEncoder.encode_images builds InferInput('images') and requests
        # InferRequestedOutput('image_embeddings').
        assert cfg.input_name == 'images'
        assert cfg.output_name == 'image_embeddings'

    def test_crop_batching_fits_under_max_batch(self) -> None:
        """PEEncoder chunks crops at PE_CROP_MAX_BATCH; Triton must accept that."""
        from src.clients.pe_encoder import PE_CROP_MAX_BATCH

        assert pe_export.PETritonConfig().max_batch_size >= PE_CROP_MAX_BATCH

    def test_committed_template_matches_the_renderer(self) -> None:
        committed = Path(__file__).resolve().parents[1] / 'models' / PE_IMAGE_MODEL
        body = (committed / 'config.pbtxt').read_text()
        assert body == pe_export.render_config(pe_export.PETritonConfig())

    def test_checkpoint_flags(self) -> None:
        parser = pe_export.build_parser()
        args = parser.parse_args([])
        assert args.checkpoint_path is None
        assert args.verify_checkpoint is True
        args = parser.parse_args(['--checkpoint-path', '/w/pe.pt', '--no-verify-checkpoint'])
        assert args.checkpoint_path == Path('/w/pe.pt')
        assert args.verify_checkpoint is False

    def test_rendered_config_declares_the_client_tensors(self) -> None:
        out = pe_export.render_config(pe_export.PETritonConfig())
        assert f'name: "{PE_IMAGE_MODEL}"' in out
        assert 'platform: "tensorrt_plan"' in out
        assert 'name: "images"' in out
        assert 'name: "image_embeddings"' in out
        assert f'dims: [ 3, {PE_SIZE}, {PE_SIZE} ]' in out
        assert f'dims: [ {PE_EMBEDDING_DIM} ]' in out
        assert out.count('data_type: TYPE_FP32') == 2


# =============================================================================
# render_config
# =============================================================================


class TestRenderConfig:
    def test_dynamic_batching_block_is_bounded_by_max_batch(self) -> None:
        out = pe_export.render_config(pe_export.PETritonConfig(max_batch_size=8))
        assert 'max_batch_size: 8' in out
        assert 'preferred_batch_size: [ 4, 8 ]' in out
        assert 'dynamic_batching' in out

    def test_max_batch_below_the_ladder_still_renders_a_valid_list(self) -> None:
        out = pe_export.render_config(pe_export.PETritonConfig(max_batch_size=2))
        assert 'preferred_batch_size: [ 2 ]' in out

    def test_ort_platform_switches_backend_and_provenance(self) -> None:
        out = pe_export.render_config(pe_export.PETritonConfig(platform='onnxruntime_onnx'))
        assert 'platform: "onnxruntime_onnx"' in out
        assert 'tensorrt_plan' not in out
        assert 'build_pe_ort_fallback.sh' in out

    def test_trt_platform_points_at_the_trt_builder(self) -> None:
        out = pe_export.render_config(pe_export.PETritonConfig())
        assert 'build_pe_trt.sh' in out
        assert 'onnxruntime_onnx' not in out

    def test_multi_gpu_instance_group(self) -> None:
        out = pe_export.render_config(pe_export.PETritonConfig(gpu_ids=(0, 2), instance_count=3))
        assert 'gpus: [ 0, 2 ]' in out
        assert 'count: 3' in out

    def test_custom_image_size_and_dim(self) -> None:
        out = pe_export.render_config(pe_export.PETritonConfig(image_size=224, embedding_dim=768))
        assert 'dims: [ 3, 224, 224 ]' in out
        assert 'dims: [ 768 ]' in out

    @pytest.mark.parametrize('max_batch', [1, 4, 32, 64, 128])
    def test_preferred_ladder_never_exceeds_max_batch(self, max_batch: int) -> None:
        ladder = pe_export.preferred_batch_sizes(max_batch)
        assert ladder
        assert max(ladder) <= max_batch


# =============================================================================
# Config validation
# =============================================================================


class TestConfigValidation:
    @pytest.mark.parametrize(
        ('field', 'value'),
        [
            ('max_batch_size', 0),
            ('max_batch_size', -1),
            ('image_size', 0),
            ('embedding_dim', 0),
            ('instance_count', 0),
            ('max_queue_delay_us', -1),
            ('model_name', ''),
            ('input_name', ''),
            ('output_name', ''),
            ('gpu_ids', ()),
            ('gpu_ids', (-1,)),
            ('platform', 'openvino'),
        ],
    )
    def test_rejects_invalid_field(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field.split('_')[0]):
            pe_export.PETritonConfig(**{field: value})

    def test_accepts_the_two_supported_platforms(self) -> None:
        for platform in pe_export.PLATFORMS:
            assert pe_export.PETritonConfig(platform=platform).platform == platform


# =============================================================================
# CLI
# =============================================================================


class TestCli:
    def test_defaults_describe_the_reference_model(self) -> None:
        args = pe_export.build_parser().parse_args([])
        assert args.method == 'perception-models'
        assert args.platform == 'tensorrt_plan'
        assert args.triton_name == PE_IMAGE_MODEL
        assert args.max_batch == 32
        assert args.image_size == PE_SIZE
        assert args.embedding_dim == PE_EMBEDDING_DIM
        assert args.write_config is True
        assert args.config_only is False

    def test_no_write_config_flips_the_default(self) -> None:
        args = pe_export.build_parser().parse_args(['--no-write-config'])
        assert args.write_config is False

    def test_no_trust_remote_code_flips_the_default(self) -> None:
        parser = pe_export.build_parser()
        assert parser.parse_args([]).trust_remote_code is True
        assert parser.parse_args(['--no-trust-remote-code']).trust_remote_code is False

    def test_unknown_platform_is_rejected_by_argparse(self) -> None:
        with pytest.raises(SystemExit) as exc:
            pe_export.build_parser().parse_args(['--platform', 'openvino'])
        assert exc.value.code == 2

    def test_unknown_method_is_rejected_by_argparse(self) -> None:
        with pytest.raises(SystemExit):
            pe_export.build_parser().parse_args(['--method', 'tensorflow'])

    def test_config_from_args_threads_overrides_through(self) -> None:
        args = pe_export.build_parser().parse_args(
            ['--max-batch', '16', '--gpus', '1', '2', '--instance-count', '4']
        )
        cfg = pe_export.config_from_args(args)
        assert cfg.max_batch_size == 16
        assert cfg.gpu_ids == (1, 2)
        assert cfg.instance_count == 4


class TestConfigOnlyMode:
    """``--config-only`` must not touch torch, ONNX Runtime or the network."""

    def test_writes_a_loadable_config_into_the_model_repository(self, tmp_path: Path) -> None:
        rc = pe_export.main(['--config-only', '--models-dir', str(tmp_path)])
        assert rc == 0
        written = tmp_path / PE_IMAGE_MODEL / 'config.pbtxt'
        assert written.is_file()
        body = written.read_text()
        assert f'name: "{PE_IMAGE_MODEL}"' in body
        assert 'platform: "tensorrt_plan"' in body

    def test_ort_platform_is_honored(self, tmp_path: Path) -> None:
        rc = pe_export.main(
            ['--config-only', '--platform', 'onnxruntime_onnx', '--models-dir', str(tmp_path)]
        )
        assert rc == 0
        body = (tmp_path / PE_IMAGE_MODEL / 'config.pbtxt').read_text()
        assert 'platform: "onnxruntime_onnx"' in body

    def test_rewriting_is_idempotent(self, tmp_path: Path) -> None:
        pe_export.main(['--config-only', '--models-dir', str(tmp_path)])
        first = (tmp_path / PE_IMAGE_MODEL / 'config.pbtxt').read_text()
        pe_export.main(['--config-only', '--models-dir', str(tmp_path)])
        assert (tmp_path / PE_IMAGE_MODEL / 'config.pbtxt').read_text() == first

    def test_invalid_max_batch_exits_nonzero_without_exporting(self, tmp_path: Path) -> None:
        rc = pe_export.main(['--config-only', '--max-batch', '0', '--models-dir', str(tmp_path)])
        assert rc == 2
        assert not (tmp_path / PE_IMAGE_MODEL).exists()

    def test_invalid_opset_exits_nonzero(self, tmp_path: Path) -> None:
        rc = pe_export.main(['--config-only', '--opset', '0', '--models-dir', str(tmp_path)])
        assert rc == 2

    def test_custom_triton_name_is_respected(self, tmp_path: Path) -> None:
        rc = pe_export.main(
            ['--config-only', '--triton-name', 'pe_image_encoder_v2', '--models-dir', str(tmp_path)]
        )
        assert rc == 0
        assert (tmp_path / 'pe_image_encoder_v2' / 'config.pbtxt').is_file()


# =============================================================================
# ONNX contract checking (the exporter's own QA gate)
# =============================================================================


def _good_report() -> pe_export.OnnxReport:
    return pe_export.OnnxReport(
        input_name='images',
        input_shape=['batch', 3, PE_SIZE, PE_SIZE],
        outputs=[('image_embeddings', ['batch', PE_EMBEDDING_DIM])],
        embedding_name='image_embeddings',
        embedding_dim=PE_EMBEDDING_DIM,
        dynamic_batch=True,
    )


class TestCheckClientContract:
    def test_a_conforming_graph_reports_no_problems(self) -> None:
        assert pe_export.check_client_contract(_good_report(), pe_export.PETritonConfig()) == []

    def test_optimum_style_tensor_names_are_flagged(self) -> None:
        report = _good_report()
        report.input_name = 'pixel_values'
        report.embedding_name = 'image_embeds'
        report.outputs = [('image_embeds', ['batch', PE_EMBEDDING_DIM])]
        problems = pe_export.check_client_contract(report, pe_export.PETritonConfig())
        assert len(problems) == 2
        assert any('pixel_values' in p for p in problems)
        assert any('image_embeds' in p for p in problems)

    def test_static_leading_axis_is_flagged(self) -> None:
        """The batch-1 trace bug: loads fine, then rejects every batch > 1."""
        report = _good_report()
        report.dynamic_batch = False
        problems = pe_export.check_client_contract(report, pe_export.PETritonConfig())
        assert any('static' in p for p in problems)

    def test_wrong_embedding_width_is_flagged(self) -> None:
        report = _good_report()
        report.embedding_dim = 512
        report.outputs = [('image_embeddings', ['batch', 512])]
        problems = pe_export.check_client_contract(report, pe_export.PETritonConfig())
        assert any('512' in p for p in problems)

    def test_missing_pooled_output_is_flagged(self) -> None:
        report = _good_report()
        report.embedding_name = None
        report.embedding_dim = None
        problems = pe_export.check_client_contract(report, pe_export.PETritonConfig())
        assert any('pooled-embedding' in p for p in problems)

    def test_a_skipped_probe_asserts_nothing(self) -> None:
        """No onnxruntime installed must not fabricate a passing verdict."""
        report = pe_export.OnnxReport(skipped=True)
        assert pe_export.check_client_contract(report, pe_export.PETritonConfig()) == []


class TestValidateOnnxAgainstARealGraph:
    """Drive :func:`validate_onnx` over genuine, executable ONNX graphs.

    A two-node graph stands in for the PE vision tower: same tensor names,
    same 336x336 input, same 1024-wide pooled output — no weights download,
    no GPU, no PyTorch. What is under test is the exporter's QA gate, not PE
    itself, and the gate only ever reads ONNX metadata plus one ORT forward.
    """

    @staticmethod
    def _stub_onnx(path: Path, *, batch_dim: str | int) -> Path:
        """Write ``images[B,3,336,336] -> image_embeddings[B,1024]``.

        ``batch_dim`` is either the symbolic name ``'batch'`` (a correct,
        batchable export) or the literal ``1`` (what a batch-1 trace bakes
        in).
        """
        pytest.importorskip('onnxruntime')
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        images = helper.make_tensor_value_info(
            'images', TensorProto.FLOAT, [batch_dim, 3, PE_SIZE, PE_SIZE]
        )
        embeddings = helper.make_tensor_value_info(
            'image_embeddings', TensorProto.FLOAT, [batch_dim, PE_EMBEDDING_DIM]
        )
        weight = numpy_helper.from_array(
            np.zeros((3, PE_EMBEDDING_DIM), dtype=np.float32), name='proj'
        )
        graph = helper.make_graph(
            [
                helper.make_node('ReduceMean', ['images'], ['pooled'], axes=[2, 3], keepdims=0),
                helper.make_node('MatMul', ['pooled', 'proj'], ['image_embeddings']),
            ],
            'pe_stub',
            [images],
            [embeddings],
            initializer=[weight],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid('', pe_export.ONNX_OPSET_VERSION)],
            # Pin the IR version to the one paired with opset 17. The
            # installed onnx package defaults to a newer IR than the
            # installed onnxruntime accepts ("Unsupported model IR version").
            ir_version=8,
        )
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
        return path

    def test_dynamic_batch_export_satisfies_the_contract(self, tmp_path: Path) -> None:
        onnx_path = self._stub_onnx(tmp_path / 'dyn.onnx', batch_dim='batch')
        report = pe_export.validate_onnx(onnx_path)
        assert report.skipped is False
        assert report.input_name == 'images'
        assert report.embedding_name == 'image_embeddings'
        assert report.embedding_dim == PE_EMBEDDING_DIM
        assert report.dynamic_batch is True
        assert pe_export.check_client_contract(report, pe_export.PETritonConfig()) == []

    def test_batch_one_trace_is_caught_as_a_static_leading_axis(self, tmp_path: Path) -> None:
        """The exact bug the batch-2 trace dummy exists to prevent."""
        onnx_path = self._stub_onnx(tmp_path / 'static.onnx', batch_dim=1)
        report = pe_export.validate_onnx(onnx_path)
        assert report.dynamic_batch is False
        problems = pe_export.check_client_contract(report, pe_export.PETritonConfig())
        assert any('static' in p for p in problems)


class TestDynamicDimDetection:
    @pytest.mark.parametrize('dim', ['batch', None, -1, 0])
    def test_symbolic_dims_are_dynamic(self, dim: object) -> None:
        assert pe_export._is_dynamic(dim) is True

    @pytest.mark.parametrize('dim', [1, 2, 32, PE_SIZE])
    def test_concrete_dims_are_static(self, dim: int) -> None:
        assert pe_export._is_dynamic(dim) is False
