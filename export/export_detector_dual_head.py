#!/usr/bin/env python3
"""
Dual-Head Detector Export (detections + backbone feature map)
=============================================================

Re-export a YOLO-family detector so its ONNX graph exposes **two** named
outputs instead of one:

    output0    : the detector head's raw tensor (layout is the family's own —
                 YOLOv5 ``[B, N, 5 + nc]``, Ultralytics v8+ ``[B, 4 + nc, N]``)
    sppf_feat  : the backbone bottleneck feature map, ``[B, C, H, W]``

Why this exists
---------------
The curation subsystem stores a per-item backbone embedding
(``v6_embedding`` on the items index, dimension
``CurationConfig.backbone_embedding_dim``) and consumes it in residual
clustering, the 2-D embedding visualization, item scores and the OCC
conflict handler. That embedding is produced by RoI-pooling a detector's
backbone feature map over each detection box
(:func:`src.services.detection.geometry.roi_pool_sppf`), which needs the
feature map on the wire — a stock detector export only emits the
detection tensor. This script is the missing producer side.

Nothing here is tied to a particular model, class count or deployment:
the checkpoint, Triton model name, network input size, tapped module and
both output names are CLI arguments, mirroring how model identity is
parameterized elsewhere in this repo (``DetectionProfile.detector_model``
/ ``input_size``, ``export_yolo26.py --custom-model``).

Loaders
-------
``--loader ultralytics``  Ultralytics checkpoints/YAMLs (YOLOv8/11/26, ...).
``--loader yolov5``       A YOLOv5 fork checkout (``models/yolo.py`` on
                          disk); the same fork
                          :mod:`src.services.detection.ensemble_nms` uses
                          for client-side NMS, so its ``DETECTION_YOLOV5_FORK``
                          env var is honoured as the default path.
``--loader auto``         (default) Ultralytics first, fork as fallback.

Usage
-----
    # ONNX only (no GPU needed)
    docker compose exec yolo-api python /app/export/export_detector_dual_head.py \\
        --weights /app/pytorch_models/my_detector.pt \\
        --triton-name my_detector_dual_head

    # ONNX + TensorRT engine + config.pbtxt written into the model repo
    docker compose exec yolo-api python /app/export/export_detector_dual_head.py \\
        --weights /app/pytorch_models/my_detector.pt \\
        --triton-name my_detector_dual_head \\
        --imgsz 1280 --max-batch 16 --formats onnx trt

    # A YOLOv5-fork checkpoint
    docker compose exec yolo-api python /app/export/export_detector_dual_head.py \\
        --weights /app/pytorch_models/legacy_v5.pt --loader yolov5 \\
        --yolov5-fork /app/external/yolov5 --imgsz 1280 \\
        --triton-name legacy_v5_dual_head

``export_detector_dual_head.sh`` is the trtexec equivalent of the ``trt``
format, for environments that have ``trtexec`` but not the TensorRT Python
bindings (e.g. inside the triton-server container).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from torch import nn


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('export_detector_dual_head')


# ============================================================================
# Configuration
# ============================================================================

# Class name of the module whose output is tapped. SPPF is the backbone
# bottleneck in every YOLOv5/v8/v11-lineage architecture; a different
# family passes its own via --feature-module (or pins --feature-index).
DEFAULT_FEATURE_MODULE = 'SPPF'
# Tensor names. `output0` is what src/services/curation/ingest_detect.py
# requests from a raw-output detector; `sppf_feat` is the new one.
DEFAULT_DETECT_OUTPUT = 'output0'
DEFAULT_FEATURE_OUTPUT = 'sppf_feat'
DEFAULT_INPUT_NAME = 'images'

DEFAULT_OPSET = 17
DEFAULT_IMG_SIZE = 640
DEFAULT_MAX_BATCH = 8
DEFAULT_MODELS_DIR = Path('/app/models')
DEFAULT_ONNX_DIR = Path('/app/pytorch_models')
# Same default + env var as src/services/detection/ensemble_nms.py, so a
# deployment points at its fork once.
DEFAULT_YOLOV5_FORK = './external/yolov5'
YOLOV5_FORK_ENV = 'DETECTION_YOLOV5_FORK'

WORKSPACE_GB = 4
# Triton model names are directory names in the model repository.
TRITON_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
ROUND_TRIP_ATOL = 1e-3


@dataclass
class DualHeadExport:
    """Report for one dual-head export run."""

    weights: str
    triton_name: str
    loader: str
    onnx: str
    imgsz: int
    opset: int
    feature_layer: str
    feature_channels: int
    feature_spatial: tuple[int, int]
    detect_dims: list[int]
    round_trip: str = 'skipped'
    plan: str | None = None
    config: str | None = None
    labels: str | None = None
    class_names: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            'weights': self.weights,
            'triton_name': self.triton_name,
            'loader': self.loader,
            'onnx': self.onnx,
            'imgsz': self.imgsz,
            'opset': self.opset,
            'feature_layer': self.feature_layer,
            'feature_channels': self.feature_channels,
            'feature_spatial': list(self.feature_spatial),
            'detect_dims': self.detect_dims,
            'round_trip': self.round_trip,
            'plan': self.plan,
            'config': self.config,
            'labels': self.labels,
            'num_classes': len(self.class_names),
        }


# ============================================================================
# Argument validation (pure — unit-tested without a GPU)
# ============================================================================


def default_yolov5_fork() -> str:
    """Fork checkout path, honouring the shared ``DETECTION_YOLOV5_FORK`` env var."""
    return os.environ.get(YOLOV5_FORK_ENV, DEFAULT_YOLOV5_FORK)


def validate_args(args: argparse.Namespace) -> None:
    """Reject argument combinations that cannot produce a loadable model.

    Raises:
        ValueError: With a message naming the offending flag. Checked here
            rather than after a multi-minute checkpoint load so a typo
            fails in milliseconds.
    """
    if not TRITON_NAME_RE.match(args.triton_name):
        msg = (
            f'--triton-name {args.triton_name!r} is not a valid Triton model '
            'name (it becomes a directory in the model repository): use '
            '[A-Za-z0-9._-] and do not start with punctuation.'
        )
        raise ValueError(msg)
    if args.imgsz <= 0 or args.imgsz % 32 != 0:
        msg = f'--imgsz must be a positive multiple of 32, got {args.imgsz}'
        raise ValueError(msg)
    if args.max_batch < 1:
        msg = f'--max-batch must be >= 1, got {args.max_batch}'
        raise ValueError(msg)
    if args.opset < 12:
        msg = f'--opset must be >= 12 for a dual-output dynamic-batch export, got {args.opset}'
        raise ValueError(msg)
    if args.feature_index is not None and args.feature_index < 0:
        msg = f'--feature-index must be >= 0, got {args.feature_index}'
        raise ValueError(msg)
    if args.detect_output == args.feature_output:
        msg = f'--detect-output and --feature-output must differ (both are {args.detect_output!r})'
        raise ValueError(msg)
    if args.loader == 'yolov5' and not (Path(args.yolov5_fork) / 'models' / 'yolo.py').exists():
        msg = (
            f'--loader yolov5 needs a fork checkout; {args.yolov5_fork}/models/yolo.py '
            f'not found. Pass --yolov5-fork or set {YOLOV5_FORK_ENV}.'
        )
        raise ValueError(msg)


# ============================================================================
# Model loading
# ============================================================================


def _load_ultralytics(weights: Path) -> tuple[nn.Module, dict[int, str]]:
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    names = dict(getattr(yolo, 'names', {}) or {})
    # ultralytics annotates ``YOLO.model`` loosely; it is the nn.Module.
    return cast('nn.Module', yolo.model), names


def _load_yolov5_fork(weights: Path, fork: Path) -> tuple[nn.Module, dict[int, str]]:
    """Load a YOLOv5-fork checkpoint via the fork's own ``attempt_load``.

    The fork is put on ``sys.path`` first because its checkpoints pickle
    ``models.yolo.DetectionModel`` by module path — unpickling fails
    without it. ``torch.load``'s ``weights_only`` default flipped to True
    in torch 2.6 and blocks those pickles, so it is forced back off for
    the duration of the load: the operator supplied this checkpoint
    explicitly, the same trust assumption every YOLOv5 export makes.
    """
    import torch

    fork = fork.resolve()
    if not (fork / 'models' / 'yolo.py').exists():
        msg = f'Expected models/yolo.py under {fork}; pass --yolov5-fork at the fork checkout.'
        raise FileNotFoundError(msg)
    if str(fork) not in sys.path:
        sys.path.insert(0, str(fork))

    from models.experimental import attempt_load  # type: ignore[import-not-found]

    original_load = torch.load

    def _trusting_load(*a: Any, **kw: Any) -> Any:
        return original_load(*a, **{**kw, 'weights_only': False})

    torch.load = _trusting_load  # type: ignore[assignment]
    try:
        model = attempt_load(str(weights), device='cpu', inplace=True, fuse=True)
    finally:
        torch.load = original_load  # type: ignore[assignment]
    raw_names = getattr(model, 'names', {}) or {}
    names = dict(enumerate(raw_names)) if isinstance(raw_names, list) else dict(raw_names)
    return model, names


def load_detector(weights: Path, loader: str, fork: Path) -> tuple[nn.Module, dict[int, str], str]:
    """Load ``weights`` and return ``(module, class_names, loader_used)``."""
    if loader == 'ultralytics':
        model, names = _load_ultralytics(weights)
        return model, names, 'ultralytics'
    if loader == 'yolov5':
        model, names = _load_yolov5_fork(weights, fork)
        return model, names, 'yolov5'

    try:
        model, names = _load_ultralytics(weights)
    except Exception as exc:
        logger.info('Ultralytics could not load %s (%s); trying the YOLOv5 fork', weights, exc)
        model, names = _load_yolov5_fork(weights, fork)
        return model, names, 'yolov5'
    return model, names, 'ultralytics'


# ============================================================================
# Graph surgery
# ============================================================================


def find_feature_layer(
    model: nn.Module, *, module_name: str, index: int | None = None
) -> tuple[str, nn.Module]:
    """Locate the backbone module whose output becomes ``sppf_feat``.

    Searched by class *name* rather than a fixed index so a fork's
    architecture drift (or a different YOLO generation) does not silently
    tap the wrong layer. ``index`` pins a layer explicitly when a model
    has an unusual graph.

    Returns:
        ``(human_readable_path, module)``.
    """
    layers: Any = getattr(model, 'model', model)
    if index is not None:
        try:
            return f'model[{index}]', layers[index]
        except (IndexError, KeyError, TypeError) as exc:
            msg = f'--feature-index {index} is not addressable on this model ({exc})'
            raise ValueError(msg) from exc

    matches: list[tuple[str, nn.Module]] = []
    try:
        for i, layer in enumerate(layers):
            if type(layer).__name__ == module_name:
                matches.append((f'model[{i}]', layer))
    except TypeError:
        matches = []
    if not matches:
        matches = [
            (name, module)
            for name, module in model.named_modules()
            if type(module).__name__ == module_name
        ]
    if not matches:
        msg = (
            f'No module of type {module_name!r} found in this model. Pass '
            '--feature-module with the backbone bottleneck class name, or '
            '--feature-index with its position.'
        )
        raise ValueError(msg)
    if len(matches) > 1:
        logger.warning(
            'Multiple %r modules found (%s); tapping the first.',
            module_name,
            [name for name, _ in matches],
        )
    return matches[0]


def build_dual_head(model: nn.Module, layer: nn.Module) -> nn.Module:
    """Wrap ``model`` so ``forward`` returns ``(detections, feature_map)``.

    A forward hook captures the tapped layer's output; the detection path
    is untouched, so the exported ``output0`` is bit-identical to a
    single-head export of the same checkpoint.
    """
    import torch
    from torch import nn as torch_nn

    class DualHeadDetector(torch_nn.Module):
        def __init__(self, base: nn.Module, tapped: nn.Module) -> None:
            super().__init__()
            self.base = base
            self._captured: torch.Tensor | None = None

            def _hook(_module: Any, _inputs: Any, output: Any) -> None:
                self._captured = output

            tapped.register_forward_hook(_hook)

        def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self._captured = None
            detect = self.base(images)
            # YOLO detection models return (pred, aux_feature_list) in eval
            # mode; only the concatenated prediction tensor is output0.
            while isinstance(detect, (tuple, list)):
                detect = detect[0]
            captured = self._captured
            if captured is None:
                msg = 'The tapped module never fired — the forward pass does not traverse it.'
                raise RuntimeError(msg)
            return detect, captured

    wrapper = DualHeadDetector(model, layer)
    wrapper.eval()
    return wrapper


# ============================================================================
# ONNX export
# ============================================================================


def export_dual_head_onnx(
    model: nn.Module,
    out_path: Path,
    *,
    imgsz: int = DEFAULT_IMG_SIZE,
    opset: int = DEFAULT_OPSET,
    feature_module: str = DEFAULT_FEATURE_MODULE,
    feature_index: int | None = None,
    detect_output: str = DEFAULT_DETECT_OUTPUT,
    feature_output: str = DEFAULT_FEATURE_OUTPUT,
    input_name: str = DEFAULT_INPUT_NAME,
    validate: bool = True,
) -> dict[str, Any]:
    """Export ``model`` to a two-output ONNX and report the tensor shapes.

    Returns a dict with ``feature_layer``, ``feature_channels``,
    ``feature_spatial``, ``detect_dims`` (per-item, batch axis dropped)
    and ``round_trip``. ``detect_dims``/``feature_*`` are what the Triton
    ``config.pbtxt`` must declare, so they are measured from a real
    forward pass rather than assumed.
    """
    import torch

    layer_path, layer = find_feature_layer(model, module_name=feature_module, index=feature_index)
    logger.info('Tapping %s (%s) for %s', layer_path, type(layer).__name__, feature_output)

    model.eval()
    wrapper = build_dual_head(model, layer)
    dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32)
    with torch.no_grad():
        detect, feature = wrapper(dummy)

    if not isinstance(feature, torch.Tensor):
        msg = (
            f'{layer_path} ({type(layer).__name__}) returned '
            f'{type(feature).__name__}, not a tensor — tap a module whose '
            'output is the [B, C, H, W] feature map.'
        )
        raise RuntimeError(msg)
    feature_shape = tuple(int(d) for d in feature.shape)
    if len(feature_shape) != 4:
        msg = f'Tapped tensor has rank {len(feature_shape)} ({feature_shape}); expected [B,C,H,W].'
        raise RuntimeError(msg)
    _, channels, fh, fw = feature_shape
    stride = imgsz / float(fh) if fh else 0.0
    logger.info(
        'Forward OK: %s=%s, %s=%s (stride %.1f)',
        detect_output,
        tuple(int(d) for d in detect.shape),
        feature_output,
        feature_shape,
        stride,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Exporting ONNX -> %s (opset %d)', out_path, opset)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        opset_version=opset,
        input_names=[input_name],
        output_names=[detect_output, feature_output],
        dynamic_axes={
            input_name: {0: 'batch'},
            detect_output: {0: 'batch'},
            feature_output: {0: 'batch'},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    graph_outputs = inspect_onnx_outputs(out_path)
    names = [name for name, _ in graph_outputs]
    if names != [detect_output, feature_output]:
        msg = f'Exported ONNX has outputs {names}; expected [{detect_output!r}, {feature_output!r}]'
        raise RuntimeError(msg)

    report: dict[str, Any] = {
        'feature_layer': layer_path,
        'feature_channels': int(channels),
        'feature_spatial': (int(fh), int(fw)),
        'detect_dims': [int(d) for d in detect.shape[1:]],
        'round_trip': 'skipped',
    }
    if validate:
        report['round_trip'] = _round_trip(out_path, dummy, detect, feature, input_name)
    return report


def inspect_onnx_outputs(onnx_path: Path) -> list[tuple[str, list[int]]]:
    """Return ``[(output_name, per_item_dims), ...]`` from an ONNX graph.

    Dynamic axes come back as ``-1``. The batch axis is dropped because
    Triton's ``dims`` are per-item when ``max_batch_size > 0``.
    """
    import onnx

    graph = onnx.load(str(onnx_path)).graph
    outputs: list[tuple[str, list[int]]] = []
    for out in graph.output:
        dims = [d.dim_value if d.dim_value > 0 else -1 for d in out.type.tensor_type.shape.dim[1:]]
        outputs.append((out.name, dims))
    return outputs


def _round_trip(
    onnx_path: Path,
    dummy: Any,
    detect: Any,
    feature: Any,
    input_name: str,
) -> str:
    """Compare onnxruntime outputs against the PyTorch reference."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ModuleNotFoundError:
        logger.warning('onnxruntime not installed; skipping round-trip validation.')
        return 'skipped'

    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    ort_detect, ort_feature = session.run(None, {input_name: dummy.numpy()})[:2]
    detect_ok = np.allclose(detect.numpy(), ort_detect, atol=ROUND_TRIP_ATOL)
    feature_ok = np.allclose(feature.numpy(), ort_feature, atol=ROUND_TRIP_ATOL)
    logger.info('ORT round-trip: detect=%s feature=%s', detect_ok, feature_ok)
    if not (detect_ok and feature_ok):
        msg = 'ONNX round-trip differs from the PyTorch reference; export rejected.'
        raise RuntimeError(msg)
    return 'ok'


# ============================================================================
# TensorRT engine + Triton model repository
# ============================================================================


def build_engine(onnx_path: Path, plan_path: Path, *, imgsz: int, max_batch: int) -> bool:
    """Build a TensorRT engine with a bounded dynamic-batch profile.

    Every dynamic axis is bounded: an unbounded spatial axis makes TRT
    budget for huge activations and fail on consumer GPUs (same reasoning
    as ``export_yolo26.build_engine``).
    """
    try:
        import tensorrt as trt
    except ModuleNotFoundError:
        logger.error(
            'TensorRT Python bindings not available. Build the engine with '
            'export/export_detector_dual_head.sh (trtexec) instead.'
        )
        return False

    sys.path.insert(0, str(Path(__file__).parent))
    from trt_utils import create_explicit_network, enable_fp16

    trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, '')

    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB << 30)
    if not enable_fp16(builder, config):
        logger.info('FP16 builder flag unavailable (strongly-typed TRT); following ONNX dtypes.')

    network = create_explicit_network(builder)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            logger.error('ONNX parse error [%d]: %s', i, parser.get_error(i))
        return False

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        profile.set_shape(
            network.get_input(i).name,
            min=(1, 3, imgsz, imgsz),
            opt=(max(1, max_batch // 2), 3, imgsz, imgsz),
            max=(max_batch, 3, imgsz, imgsz),
        )
    config.add_optimization_profile(profile)

    logger.info('Building TensorRT engine (this may take several minutes)...')
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        logger.error('Engine build failed — builder returned None')
        return False

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(serialized)
    logger.info('Engine saved: %s (%.1f MB)', plan_path, plan_path.stat().st_size / 1e6)
    return True


def render_triton_config(
    *,
    triton_name: str,
    imgsz: int,
    max_batch: int,
    detect_output: str,
    detect_dims: list[int],
    feature_output: str,
    feature_channels: int,
    feature_spatial: tuple[int, int],
    input_name: str = DEFAULT_INPUT_NAME,
) -> str:
    """Render ``config.pbtxt`` for a dual-output TensorRT plan.

    Both outputs are declared: a Triton model only serves tensors its
    config names, so omitting ``sppf_feat`` here silently drops the
    feature map even though the engine produces it.
    """
    preferred = [size for size in (8, 16, 32, 64) if size <= max_batch] or [1]
    preferred_str = ', '.join(str(b) for b in preferred)
    detect_dims_str = ', '.join(str(d) for d in detect_dims)
    fh, fw = feature_spatial
    return f"""# Auto-generated by export/export_detector_dual_head.py.
# Dual-head detector: detections + backbone feature map. The feature map
# feeds src.services.detection.geometry.roi_pool_sppf, whose pooled,
# L2-normalized vector is stored as the curation `v6_embedding` field.

name: "{triton_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch}

input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [ 3, {imgsz}, {imgsz} ]
  }}
]

output [
  {{
    name: "{detect_output}"
    data_type: TYPE_FP32
    dims: [ {detect_dims_str} ]
  }},
  {{
    name: "{feature_output}"
    data_type: TYPE_FP32
    dims: [ {feature_channels}, {fh}, {fw} ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ {preferred_str} ]
  max_queue_delay_microseconds: 5000
}}

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""


def render_labels(class_names: dict[int, str]) -> str:
    """One class name per line in id order (Triton ``labels.txt``)."""
    if not class_names:
        return ''
    return '\n'.join(class_names.get(i, f'unknown_{i}') for i in range(max(class_names) + 1)) + '\n'


# ============================================================================
# Main
# ============================================================================


def run(args: argparse.Namespace) -> DualHeadExport:
    """Execute the export described by ``args`` (already validated)."""
    weights = args.weights.resolve()
    if not weights.exists():
        msg = f'Checkpoint not found: {weights}'
        raise FileNotFoundError(msg)

    onnx_path = args.onnx or (args.onnx_dir / f'{args.triton_name}.onnx')
    model, class_names, loader_used = load_detector(weights, args.loader, args.yolov5_fork)
    onnx_report = export_dual_head_onnx(
        model,
        onnx_path,
        imgsz=args.imgsz,
        opset=args.opset,
        feature_module=args.feature_module,
        feature_index=args.feature_index,
        detect_output=args.detect_output,
        feature_output=args.feature_output,
        validate=not args.no_validate,
    )

    result = DualHeadExport(
        weights=str(weights),
        triton_name=args.triton_name,
        loader=loader_used,
        onnx=str(onnx_path),
        imgsz=args.imgsz,
        opset=args.opset,
        feature_layer=onnx_report['feature_layer'],
        feature_channels=onnx_report['feature_channels'],
        feature_spatial=onnx_report['feature_spatial'],
        detect_dims=onnx_report['detect_dims'],
        round_trip=onnx_report['round_trip'],
        class_names=[class_names[i] for i in sorted(class_names)],
    )

    if 'trt' not in args.formats:
        return result

    model_dir = args.models_dir / args.triton_name
    plan_path = model_dir / '1' / 'model.plan'
    if not build_engine(onnx_path, plan_path, imgsz=args.imgsz, max_batch=args.max_batch):
        msg = f'TensorRT engine build failed for {args.triton_name}'
        raise RuntimeError(msg)
    result.plan = str(plan_path)

    config_path = model_dir / 'config.pbtxt'
    config_path.write_text(
        render_triton_config(
            triton_name=args.triton_name,
            imgsz=args.imgsz,
            max_batch=args.max_batch,
            detect_output=args.detect_output,
            detect_dims=result.detect_dims,
            feature_output=args.feature_output,
            feature_channels=result.feature_channels,
            feature_spatial=result.feature_spatial,
        )
    )
    result.config = str(config_path)
    logger.info('Wrote %s', config_path)

    labels = render_labels(class_names)
    if labels:
        labels_path = model_dir / 'labels.txt'
        labels_path.write_text(labels)
        result.labels = str(labels_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Export a YOLO-family detector with a backbone feature-map output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--weights', type=Path, required=True, help='Detector checkpoint (.pt).')
    parser.add_argument(
        '--triton-name',
        required=True,
        help='Triton model name (also the model-repository directory name).',
    )
    parser.add_argument(
        '--loader',
        choices=['auto', 'ultralytics', 'yolov5'],
        default='auto',
        help='Checkpoint loader (default: auto — ultralytics, then the YOLOv5 fork).',
    )
    parser.add_argument(
        '--yolov5-fork',
        type=Path,
        default=Path(default_yolov5_fork()),
        help=f'YOLOv5 fork checkout (default: ${YOLOV5_FORK_ENV} or {DEFAULT_YOLOV5_FORK}).',
    )
    parser.add_argument(
        '--feature-module',
        default=DEFAULT_FEATURE_MODULE,
        help=f'Class name of the module to tap (default: {DEFAULT_FEATURE_MODULE}).',
    )
    parser.add_argument(
        '--feature-index',
        type=int,
        default=None,
        help='Tap this layer index instead of searching by class name.',
    )
    parser.add_argument('--imgsz', type=int, default=DEFAULT_IMG_SIZE, help='Network input size.')
    parser.add_argument('--opset', type=int, default=DEFAULT_OPSET, help='ONNX opset version.')
    parser.add_argument(
        '--max-batch', type=int, default=DEFAULT_MAX_BATCH, help='Engine/Triton max batch size.'
    )
    parser.add_argument(
        '--detect-output',
        default=DEFAULT_DETECT_OUTPUT,
        help=f'Name of the detection output tensor (default: {DEFAULT_DETECT_OUTPUT}).',
    )
    parser.add_argument(
        '--feature-output',
        default=DEFAULT_FEATURE_OUTPUT,
        help=f'Name of the feature-map output tensor (default: {DEFAULT_FEATURE_OUTPUT}).',
    )
    parser.add_argument('--onnx', type=Path, default=None, help='Explicit ONNX output path.')
    parser.add_argument(
        '--onnx-dir',
        type=Path,
        default=DEFAULT_ONNX_DIR,
        help=f'Directory for the intermediate ONNX (default: {DEFAULT_ONNX_DIR}).',
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f'Triton model repository (default: {DEFAULT_MODELS_DIR}).',
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['onnx', 'trt'],
        default=['onnx'],
        help='Artifacts to produce (default: onnx; onnx is always produced).',
    )
    parser.add_argument(
        '--no-validate', action='store_true', help='Skip the onnxruntime round-trip check.'
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    try:
        validate_args(args)
    except ValueError as exc:
        logger.error('%s', exc)
        return 2

    result = run(args)
    logger.info('Export complete')
    for key, value in result.as_dict().items():
        logger.info('  %s: %s', key, value)
    logger.info(
        'Declare both outputs in config.pbtxt and RoI-pool %s per detection '
        '(src.services.detection.geometry.roi_pool_sppf, target_dim = '
        'CurationConfig.backbone_embedding_dim).',
        args.feature_output,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
