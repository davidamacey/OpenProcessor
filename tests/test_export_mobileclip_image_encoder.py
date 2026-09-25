"""Tests for the F-13c fix in ``export/export_mobileclip_image_encoder.py``.

Fresh-start E2E findings (2026-09-25), F-13(c): the MobileCLIP2-S2 image
encoder's FP16 TensorRT build fails ("Could not find any implementation for
node …stages.1/downsample/proj/proj.0/reparam_conv/Conv + PWN…") with no
FP32 fallback, and the exporter still printed "Export Complete" and exited
0. Before the fix, ``convert_to_tensorrt`` caught the FP16 build failure and
returned ``None`` -- it never retried at FP32, and callers had no way to
tell success from failure short of checking for a missing file.

These tests fake ``tensorrt`` (and the ``trt_utils`` precision helpers)
entirely -- no GPU, no TensorRT package, no real MobileCLIP checkpoint --
and assert:
  1. an FP16 build failure automatically falls back to an FP32 build that
     still produces a usable engine (was: silently produced nothing);
  2. when *every* precision fails, ``convert_to_tensorrt`` raises instead
     of returning ``None`` silently;
  3. ``main()`` propagates that failure as a non-zero exit code (was:
     always printed "Export Complete" and exited 0, F-13c's "fail-open"
     symptom).
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


def _install_fake_tensorrt(monkeypatch: pytest.MonkeyPatch, build_results: list):
    """Fake ``tensorrt`` whose builder returns ``build_results`` in order.

    ``None`` mimics a failed build (TensorRT's real signal for "no engine
    produced"); bytes mimics a successful serialized engine.
    """
    calls = {'i': 0}

    def build_serialized_network(_network, _config):
        result = build_results[calls['i']]
        calls['i'] += 1
        return result

    builder = MagicMock()
    builder.build_serialized_network.side_effect = build_serialized_network
    builder.create_builder_config.return_value = MagicMock()
    builder.create_optimization_profile.return_value = MagicMock()

    parser = MagicMock()
    parser.parse.return_value = True

    fake_trt = types.SimpleNamespace(
        Logger=lambda _level=None: MagicMock(),
        Builder=lambda _logger: builder,
        OnnxParser=lambda _network, _logger: parser,
        MemoryPoolType=types.SimpleNamespace(WORKSPACE=0),
    )
    fake_trt.Logger.WARNING = 0
    monkeypatch.setitem(sys.modules, 'tensorrt', fake_trt)
    return builder


@pytest.fixture
def mce(monkeypatch: pytest.MonkeyPatch):
    """Import the module fresh with trt_utils' precision helpers stubbed out.

    Stubbing ``trt_utils.bake_fp16_onnx``/``enable_fp16``/
    ``create_explicit_network`` isolates this test to the fallback
    control-flow in ``export_mobileclip_image_encoder.py`` itself, rather
    than re-testing ``trt_utils`` (covered by ``tests/test_trt_utils.py``)
    or requiring a real ONNX graph + onnxconverter-common.
    """
    monkeypatch.delitem(sys.modules, 'export_mobileclip_image_encoder', raising=False)
    monkeypatch.delitem(sys.modules, 'trt_utils', raising=False)
    import trt_utils

    monkeypatch.setattr(trt_utils, 'bake_fp16_onnx', lambda onnx_path: onnx_path)
    monkeypatch.setattr(trt_utils, 'enable_fp16', lambda _builder, _config: True)
    monkeypatch.setattr(trt_utils, 'create_explicit_network', lambda _builder: MagicMock())

    import export_mobileclip_image_encoder as module

    return module


class TestConvertToTensorrtFp32Fallback:
    def test_fp16_failure_falls_back_to_a_working_fp32_engine(
        self, mce, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        onnx_path = tmp_path / 'image_encoder.onnx'
        onnx_path.write_bytes(b'fake-onnx-graph')
        plan_path = tmp_path / 'models' / 'mobileclip2_s2_image_encoder' / '1' / 'model.plan'

        # First build_serialized_network call (FP16) fails; second (FP32
        # fallback) succeeds.
        _install_fake_tensorrt(monkeypatch, build_results=[None, b'fp32-engine-bytes'])

        result = mce.convert_to_tensorrt(onnx_path, plan_path, fp16=True, max_batch_size=8)

        assert result == plan_path
        assert plan_path.read_bytes() == b'fp32-engine-bytes'

    def test_both_precisions_failing_raises_instead_of_returning_none(
        self, mce, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        onnx_path = tmp_path / 'image_encoder.onnx'
        onnx_path.write_bytes(b'fake-onnx-graph')
        plan_path = tmp_path / 'model.plan'

        _install_fake_tensorrt(monkeypatch, build_results=[None, None])

        with pytest.raises(RuntimeError, match='every precision'):
            mce.convert_to_tensorrt(onnx_path, plan_path, fp16=True, max_batch_size=8)

        assert not plan_path.exists()

    def test_fp16_success_never_attempts_fp32(
        self, mce, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        onnx_path = tmp_path / 'image_encoder.onnx'
        onnx_path.write_bytes(b'fake-onnx-graph')
        plan_path = tmp_path / 'model.plan'

        builder = _install_fake_tensorrt(monkeypatch, build_results=[b'fp16-engine-bytes'])

        result = mce.convert_to_tensorrt(onnx_path, plan_path, fp16=True, max_batch_size=8)

        assert result == plan_path
        assert plan_path.read_bytes() == b'fp16-engine-bytes'
        assert builder.build_serialized_network.call_count == 1


class TestMainExitsNonZeroOnTotalFailure:
    def test_main_exits_1_when_every_precision_fails(
        self, mce, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reproduces F-13c's fail-open symptom: previously ``main()`` always
        printed "Export Complete" and returned normally (exit 0) even when
        no TensorRT engine was ever written."""
        monkeypatch.setattr(sys, 'argv', ['export_mobileclip_image_encoder.py', '--model', 'S2'])
        monkeypatch.setattr(
            mce, 'load_mobileclip_model', lambda _model_name, _checkpoint_path: MagicMock()
        )
        monkeypatch.setattr(mce, 'export_to_onnx', lambda _model, _output_path: tmp_path / 'x.onnx')
        (tmp_path / 'x.onnx').write_bytes(b'fake')
        monkeypatch.setattr(mce, 'validate_onnx', lambda _encoder, _onnx_path: True)
        monkeypatch.setattr(mce, 'benchmark_onnx', lambda *_a, **_k: None)

        def _always_fails(*_args, **_kwargs):
            raise RuntimeError('TensorRT conversion failed at every precision attempted: boom')

        monkeypatch.setattr(mce, 'convert_to_tensorrt', _always_fails)

        # Only the hardcoded checkpoint-existence check needs to be faked;
        # everything downstream of it is already monkeypatched above.
        real_exists = Path.exists

        def _fake_exists(self):
            if self.name == 'mobileclip2_s2.pt':
                return True
            return real_exists(self)

        monkeypatch.setattr(Path, 'exists', _fake_exists)

        with pytest.raises(SystemExit) as exc_info:
            mce.main()

        assert exc_info.value.code == 1
