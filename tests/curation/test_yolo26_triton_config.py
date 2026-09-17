"""Unit tests for the YOLO26 Triton config writer (Phase 4).

Pure rendering tests — no file I/O, no Triton, no docker. The file
system + HTTP behaviors are covered in test_triton_promote.py.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.services.training.yolo_triton_config import (
    DEFAULT_INPUT_SIZE,
    DEFAULT_MAX_BATCH,
    YOLO26_DET_DIMS,
    YOLO26_TOPK,
    Yolo26TritonConfig,
    render_config,
    render_labels_file,
)


# =============================================================================
# render_config
# =============================================================================


class TestRenderConfig:
    def test_defaults_emit_yolo26_shape(self) -> None:
        cfg = Yolo26TritonConfig(model_name='yolo26m_v7_test')
        out = render_config(cfg)
        # name + backend
        assert 'name: "yolo26m_v7_test"' in out
        assert 'backend: "onnxruntime"' in out
        # shape: [3, 640, 640] in / [300, 6] out
        assert f'dims: [ 3, {DEFAULT_INPUT_SIZE}, {DEFAULT_INPUT_SIZE} ]' in out
        assert f'dims: [ {YOLO26_TOPK}, {YOLO26_DET_DIMS} ]' in out
        assert 'name: "output0"' in out
        # No EfficientNMS plugin reference (the YOLO11 path has this; YOLO26
        # must NOT — NMS is internal to the forward pass).
        assert 'EfficientNMS' not in out
        # FP16 is the default
        assert 'precision_mode' in out
        assert '"FP16"' in out
        # Dynamic batching block present
        assert 'dynamic_batching' in out
        # TensorRT execution accelerator block present
        assert 'tensorrt' in out
        assert 'trt_engine_cache_enable' in out

    def test_fp32_mode(self) -> None:
        cfg = Yolo26TritonConfig(model_name='m1', fp16=False)
        out = render_config(cfg)
        assert '"FP32"' in out
        assert '"FP16"' not in out

    def test_custom_max_batch_appears_in_output(self) -> None:
        cfg = Yolo26TritonConfig(model_name='m1', max_batch_size=16)
        out = render_config(cfg)
        assert 'max_batch_size: 16' in out
        assert 'preferred_batch_size: [ 4, 16 ]' in out

    def test_input_size_override(self) -> None:
        cfg = Yolo26TritonConfig(model_name='m1', input_size=1280)
        out = render_config(cfg)
        assert 'dims: [ 3, 1280, 1280 ]' in out

    def test_multi_gpu_id_list(self) -> None:
        cfg = Yolo26TritonConfig(model_name='m1', gpu_ids=(0, 1))
        out = render_config(cfg)
        # Triton's syntax is `gpus: [ 0, 1 ]`
        assert 'gpus: [ 0, 1 ]' in out

    @pytest.mark.parametrize(
        ('field', 'value'),
        [
            ('max_batch_size', 0),
            ('max_batch_size', -1),
            ('input_size', 0),
            ('input_size', -32),
        ],
    )
    def test_rejects_invalid_dims(self, field: str, value: int) -> None:
        # mypy can't reason about **{field: value} matching the dataclass
        # field types; cast to Any to silence the type-arg check while
        # keeping the runtime parametrize behavior.
        kwargs: dict[str, Any] = {field: value}
        cfg = Yolo26TritonConfig(model_name='m1', **kwargs)
        with pytest.raises(ValueError, match=field.split('_', maxsplit=1)[0]):
            render_config(cfg)

    def test_rejects_empty_gpu_list(self) -> None:
        cfg = Yolo26TritonConfig(model_name='m1', gpu_ids=())
        with pytest.raises(ValueError, match='gpu'):
            render_config(cfg)

    def test_workspace_bytes_round_trip(self) -> None:
        cfg = Yolo26TritonConfig(model_name='m1', trt_workspace_bytes=2 * 1024 * 1024 * 1024)
        out = render_config(cfg)
        assert '"max_workspace_size_bytes" value: "2147483648"' in out


# =============================================================================
# render_labels_file
# =============================================================================


class TestRenderLabelsFile:
    def test_simple_contiguous(self) -> None:
        out = render_labels_file({0: 'pickup', 1: 'sedan', 2: 'license_plate'})
        assert out.splitlines() == ['pickup', 'sedan', 'license_plate']

    def test_gaps_emit_unknown_placeholder(self) -> None:
        # Subset training rebases ids contiguous, but a misconfigured
        # caller could pass a gappy mapping. Guard against blank lines.
        out = render_labels_file({0: 'pickup', 2: 'license_plate'})
        lines = out.splitlines()
        assert lines == ['pickup', 'unknown_1', 'license_plate']

    def test_empty_mapping(self) -> None:
        assert render_labels_file({}) == ''

    def test_trailing_newline(self) -> None:
        out = render_labels_file({0: 'pickup'})
        assert out.endswith('\n')


# =============================================================================
# Constants sanity
# =============================================================================


def test_defaults_are_sane() -> None:
    assert DEFAULT_INPUT_SIZE == 640
    assert DEFAULT_MAX_BATCH == 8
    assert YOLO26_TOPK == 300
    assert YOLO26_DET_DIMS == 6
