"""Dual-head detector model: weight loading, feature-layer tap, ONNX export and graph checks."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger('export_detector_dual_head')


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


ROUND_TRIP_ATOL = 1e-3


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

    from models.experimental import attempt_load

    original_load = torch.load

    def _trusting_load(*a: Any, **kw: Any) -> Any:
        return original_load(*a, **{**kw, 'weights_only': False})

    torch.load = _trusting_load
    try:
        model = attempt_load(str(weights), device='cpu', inplace=True, fuse=True)
    finally:
        torch.load = original_load
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
    # After the float round trip: the served engine takes uint8 pixels and scales them itself.
    sys.path.insert(0, str(Path(__file__).parent))
    from uint8_input import add_uint8_input

    add_uint8_input(out_path, out_path)
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
