"""Tests for ``export/export_detector_dual_head.py`` and its ``.sh`` companion.

The exporter's job is to produce a detector ONNX with **two** named
outputs — the detection tensor plus the backbone feature map that
:func:`src.services.detection.geometry.roi_pool_sppf` pools into the
curation ``backbone_embedding`` field. Two things are checkable without a GPU
and both are checked here:

1. **CLI/argument validation** — every combination that cannot produce a
   loadable model is rejected before a multi-minute checkpoint load.
2. **ONNX graph structure** — a real ``torch.onnx.export`` runs on CPU
   against a synthetic YOLO-shaped model *and* against a real (randomly
   initialized, offline-constructed) Ultralytics YOLO11n, asserting the
   graph has exactly the two expected output names in order and that the
   reported feature-map shape is the ``[C, H, W]`` the RoI pooler needs.

What is **not** covered here: the TensorRT engine build and real Triton
inference — both need GPU infrastructure.

The export scripts are not a package, so the module is loaded via
``importlib`` (same convention as ``tests/curation/test_backfill_scores_cli.py``).
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn


if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PY = REPO_ROOT / 'export' / 'export_detector_dual_head.py'
SCRIPT_SH = REPO_ROOT / 'export' / 'export_detector_dual_head.sh'


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location('export_detector_dual_head_test', SCRIPT_PY)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def mod() -> ModuleType:
    return _load_module()


def _args(mod: ModuleType, argv: list[str]):
    return mod.build_parser().parse_args(argv)


BASE_ARGV = ['--weights', 'model.pt', '--triton-name', 'my_detector']


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_defaults_are_valid(mod: ModuleType) -> None:
    mod.validate_args(_args(mod, BASE_ARGV))


@pytest.mark.parametrize(
    ('extra', 'expected'),
    [
        (['--triton-name', 'has/slash'], 'not a valid Triton model'),
        (['--triton-name', '.hidden'], 'not a valid Triton model'),
        (['--triton-name', ''], 'not a valid Triton model'),
        (['--imgsz', '100'], 'multiple of 32'),
        (['--imgsz', '0'], 'multiple of 32'),
        (['--max-batch', '0'], '--max-batch must be >= 1'),
        (['--opset', '9'], '--opset must be >= 12'),
        (['--feature-index', '-1'], '--feature-index must be >= 0'),
        (['--feature-output', 'output0'], 'must differ'),
    ],
)
def test_invalid_arguments_are_rejected(mod: ModuleType, extra: list[str], expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        mod.validate_args(_args(mod, [*BASE_ARGV, *extra]))


def test_yolov5_loader_requires_a_fork_checkout(mod: ModuleType, tmp_path: Path) -> None:
    argv = [*BASE_ARGV, '--loader', 'yolov5', '--yolov5-fork', str(tmp_path)]
    with pytest.raises(ValueError, match=r'models/yolo\.py'):
        mod.validate_args(_args(mod, argv))

    (tmp_path / 'models').mkdir()
    (tmp_path / 'models' / 'yolo.py').write_text('# fork stub\n')
    mod.validate_args(_args(mod, argv))


def test_yolov5_fork_default_ignores_the_removed_env_var(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The serving path's NMS is native, so no shared fork env var remains."""
    monkeypatch.setenv('DETECTION_YOLOV5_FORK', '/opt/forks/yolov5')
    args = _args(mod, BASE_ARGV)
    assert args.yolov5_fork == Path(mod.DEFAULT_YOLOV5_FORK)


def test_main_exits_2_on_invalid_arguments(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A validation failure must not reach ``run()``."""

    def _explode(_args):
        raise AssertionError('run() must not be reached for invalid arguments')

    monkeypatch.setattr(mod, 'run', _explode)
    assert mod.main([*BASE_ARGV, '--imgsz', '77']) == 2


def test_main_requires_weights_and_name(mod: ModuleType) -> None:
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(['--weights', 'model.pt'])
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(['--triton-name', 'x'])


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------


class SPPF(nn.Module):
    """Stand-in for the backbone bottleneck the exporter taps by class name."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.cv = nn.Conv2d(c1, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv(x)


class Head(nn.Module):
    """Detection head: returns ``(predictions, aux_features)`` like YOLO in eval mode."""

    def __init__(self, c: int, nc: int) -> None:
        super().__init__()
        self.cv = nn.Conv2d(c, 5 + nc, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        y = self.cv(x)
        b, ch, h, w = y.shape
        return y.view(b, ch, h * w).permute(0, 2, 1), [y]


class TinyDetector(nn.Module):
    """A minimal YOLO-shaped model: ``Sequential`` backbone with an ``SPPF``."""

    def __init__(self, nc: int = 3) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=32, padding=1),
            SPPF(8, 16),
            Head(16, nc),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


@pytest.fixture
def tiny_model() -> TinyDetector:
    return TinyDetector()


def test_find_feature_layer_by_class_name(mod: ModuleType, tiny_model) -> None:
    path, layer = mod.find_feature_layer(tiny_model, module_name='SPPF')
    assert path == 'model[1]'
    assert type(layer).__name__ == 'SPPF'


def test_find_feature_layer_by_explicit_index(mod: ModuleType, tiny_model) -> None:
    path, layer = mod.find_feature_layer(tiny_model, module_name='SPPF', index=0)
    assert path == 'model[0]'
    assert type(layer).__name__ == 'Conv2d'


def test_find_feature_layer_unknown_module_raises(mod: ModuleType, tiny_model) -> None:
    with pytest.raises(ValueError, match='No module of type'):
        mod.find_feature_layer(tiny_model, module_name='NotARealBlock')


def test_find_feature_layer_out_of_range_index_raises(mod: ModuleType, tiny_model) -> None:
    with pytest.raises(ValueError, match='not addressable'):
        mod.find_feature_layer(tiny_model, module_name='SPPF', index=99)


# ---------------------------------------------------------------------------
# ONNX graph structure (CPU)
# ---------------------------------------------------------------------------


def test_export_emits_two_named_outputs(mod: ModuleType, tiny_model, tmp_path: Path) -> None:
    pytest.importorskip('onnx')
    onnx_path = tmp_path / 'tiny_dual.onnx'

    report = mod.export_dual_head_onnx(tiny_model, onnx_path, imgsz=64, opset=17)

    assert onnx_path.exists()
    outputs = mod.inspect_onnx_outputs(onnx_path)
    assert [name for name, _ in outputs] == ['output0', 'sppf_feat']
    # 64 / stride 32 -> a 2x2 grid of 16-channel cells, the [C, H, W] shape
    # roi_pool_sppf expects (H == W == input_size // stride).
    assert report['feature_channels'] == 16
    assert report['feature_spatial'] == (2, 2)
    assert report['feature_layer'] == 'model[1]'
    assert report['detect_dims'] == [4, 8]


def test_export_honours_custom_output_names(mod: ModuleType, tiny_model, tmp_path: Path) -> None:
    pytest.importorskip('onnx')
    onnx_path = tmp_path / 'renamed.onnx'

    mod.export_dual_head_onnx(
        tiny_model,
        onnx_path,
        imgsz=64,
        detect_output='detections',
        feature_output='backbone_feat',
    )

    assert [name for name, _ in mod.inspect_onnx_outputs(onnx_path)] == [
        'detections',
        'backbone_feat',
    ]


def test_export_round_trip_validation_runs(mod: ModuleType, tiny_model, tmp_path: Path) -> None:
    """onnxruntime must agree with the PyTorch reference on both tensors."""
    pytest.importorskip('onnxruntime')
    report = mod.export_dual_head_onnx(tiny_model, tmp_path / 'rt.onnx', imgsz=64)
    assert report['round_trip'] == 'ok'


def test_export_rejects_a_non_feature_map_tap(mod: ModuleType, tiny_model, tmp_path: Path) -> None:
    """Tapping the detection head captures a tuple, not a ``[B, C, H, W]`` map."""
    with pytest.raises(RuntimeError, match='not a tensor'):
        mod.export_dual_head_onnx(tiny_model, tmp_path / 'bad.onnx', imgsz=64, feature_index=2)


class _Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2)


class Rank3Detector(nn.Module):
    """Its tapped layer emits ``[B, C, N]`` — no spatial grid to pool over."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 8, 3, stride=32, padding=1), _Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def test_export_rejects_a_wrong_rank_tap(mod: ModuleType, tmp_path: Path) -> None:
    """A rank-3 tensor cannot be RoI-pooled over a spatial grid."""
    with pytest.raises(RuntimeError, match=r'expected \[B,C,H,W\]'):
        mod.export_dual_head_onnx(
            Rank3Detector(), tmp_path / 'rank.onnx', imgsz=64, feature_index=1
        )


def test_export_of_a_real_yolo_architecture(mod: ModuleType, tmp_path: Path) -> None:
    """End-to-end on a real Ultralytics YOLO11n graph (random weights, offline).

    Built from the bundled architecture YAML rather than a downloaded
    checkpoint so the test needs no network; the graph — and therefore
    the SPPF tap point — is the real one.
    """
    pytest.importorskip('onnx')
    ultralytics = pytest.importorskip('ultralytics')

    model = ultralytics.YOLO('yolo11n.yaml').model
    onnx_path = tmp_path / 'yolo11n_dual.onnx'

    report = mod.export_dual_head_onnx(model, onnx_path, imgsz=160, opset=17)

    assert [name for name, _ in mod.inspect_onnx_outputs(onnx_path)] == ['output0', 'sppf_feat']
    assert report['feature_layer'] == 'model[9]'
    # SPPF sits at stride 32 in the YOLO11 backbone.
    assert report['feature_spatial'] == (160 // 32, 160 // 32)
    assert report['feature_channels'] > 0
    # YOLO11 head: [4 + nc, anchors] per item, 80 COCO classes.
    assert report['detect_dims'][0] == 84


# ---------------------------------------------------------------------------
# Triton artifacts
# ---------------------------------------------------------------------------


def test_render_triton_config_declares_both_outputs(mod: ModuleType) -> None:
    config = mod.render_triton_config(
        triton_name='my_detector',
        imgsz=1280,
        max_batch=16,
        detect_output='output0',
        detect_dims=[100800, 85],
        feature_output='sppf_feat',
        feature_channels=768,
        feature_spatial=(40, 40),
    )

    assert 'name: "my_detector"' in config
    assert 'platform: "tensorrt_plan"' in config
    assert 'max_batch_size: 16' in config
    assert 'dims: [ 3, 1280, 1280 ]' in config
    assert 'name: "output0"' in config
    assert 'dims: [ 100800, 85 ]' in config
    assert 'name: "sppf_feat"' in config
    assert 'dims: [ 768, 40, 40 ]' in config
    # Preferred sizes never exceed max_batch.
    assert 'preferred_batch_size: [ 8, 16 ]' in config


def test_render_triton_config_small_batch_has_a_preferred_size(mod: ModuleType) -> None:
    config = mod.render_triton_config(
        triton_name='m',
        imgsz=640,
        max_batch=4,
        detect_output='output0',
        detect_dims=[84, 8400],
        feature_output='sppf_feat',
        feature_channels=256,
        feature_spatial=(20, 20),
    )
    assert 'preferred_batch_size: [ 1 ]' in config


def test_render_labels_fills_gaps(mod: ModuleType) -> None:
    assert mod.render_labels({}) == ''
    assert mod.render_labels({0: 'car', 2: 'truck'}) == 'car\nunknown_1\ntruck\n'


# ---------------------------------------------------------------------------
# Shell companion (trtexec build + install)
# ---------------------------------------------------------------------------


def _run_sh(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['bash', str(SCRIPT_SH), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_sh_script_is_syntactically_valid() -> None:
    assert (
        subprocess.run(['bash', '-n', str(SCRIPT_SH)], capture_output=True, check=False).returncode
        == 0
    )


def test_sh_help_exits_zero_and_documents_usage() -> None:
    result = _run_sh('--help')
    assert result.returncode == 0
    assert '--onnx' in result.stdout
    assert '--name' in result.stdout


@pytest.mark.parametrize(
    ('argv', 'expected_rc', 'expected_err'),
    [
        ([], 1, '--onnx is required'),
        (['--onnx', '/etc/hostname'], 1, '--name is required'),
        (['--onnx', '/etc/hostname', '--name', 'bad name'], 1, 'not a valid Triton'),
        (['--onnx', '/etc/hostname', '--name', 'ok', '--input-size', '100'], 1, 'multiple of 32'),
        (['--onnx', '/etc/hostname', '--name', 'ok', '--max-batch', 'x'], 1, 'positive integer'),
        (
            ['--onnx', '/etc/hostname', '--name', 'ok', '--min-batch', '8', '--opt-batch', '2'],
            1,
            'min <= opt <= max',
        ),
        (['--onnx', '/nonexistent/model.onnx', '--name', 'ok'], 3, 'ONNX not found'),
        (['--unknown-flag'], 1, 'unknown argument'),
    ],
)
def test_sh_argument_validation(argv: list[str], expected_rc: int, expected_err: str) -> None:
    result = _run_sh(*argv)
    assert result.returncode == expected_rc, result.stderr
    assert expected_err in result.stderr


@pytest.mark.skipif(shutil.which('trtexec') is not None, reason='trtexec is installed')
def test_sh_reports_missing_trtexec_distinctly() -> None:
    """A valid invocation without TensorRT must say so, not fail obscurely."""
    result = _run_sh('--onnx', '/etc/hostname', '--name', 'ok')
    assert result.returncode == 2
    assert 'trtexec' in result.stderr
