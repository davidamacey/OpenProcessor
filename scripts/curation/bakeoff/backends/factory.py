"""Built-in detector backends and the system-under-test builder.

Registers the five generic backends (``ultralytics``, ``triton``,
``two-stage``, ``onnxruntime``, ``coreml``) into ``registry.py`` on import,
and builds the detector a ``run`` scores from parsed CLI args, honoring
``--mode crop``. Backends import lazily: each pulls in its own runtime
(Ultralytics, ONNX Runtime, tritonclient, coremltools) only when built.
Backend-specific options arrive in ``args.backend_options``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import get_backend, register_backend


if TYPE_CHECKING:
    import argparse

    from .base import Detector


def _ultralytics(args: argparse.Namespace) -> Detector:
    from .ultralytics_pt import UltralyticsDetector

    return UltralyticsDetector(
        args.weights,
        name=args.name,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf_floor,
        iou=args.nms_iou,
    )


def _triton(args: argparse.Namespace) -> Detector:
    from .triton_trt import TritonYoloDetector

    return TritonYoloDetector(
        url=args.triton_url,
        model=args.triton_model,
        name=args.name,
        input_size=args.imgsz,
        conf=args.conf_floor,
        iou=args.nms_iou,
    )


def _onnxruntime(args: argparse.Namespace) -> Detector:
    from .onnxruntime_ort import OnnxRuntimeDetector

    return OnnxRuntimeDetector(
        args.weights,
        name=args.name,
        imgsz=args.imgsz,
        conf=args.conf_floor,
        iou=args.nms_iou,
        providers=args.ort_providers,
        coords_normalized=args.coords_normalized,
    )


def _coreml(args: argparse.Namespace) -> Detector:
    from .coreml import CoreMLDetector

    return CoreMLDetector(
        args.weights,
        name=args.name,
        imgsz=args.imgsz,
        conf=args.conf_floor,
        iou=args.nms_iou,
        compute_units=args.coreml_compute_units,
        coords_normalized=args.coords_normalized,
    )


def parse_class_ids(value: str | None) -> tuple[int, ...]:
    """``'2,3'`` -> ``(2, 3)``; empty/None -> ``()`` (keep every class)."""
    return tuple(int(c) for c in str(value or '').split(',') if c.strip())


def _primary_detector(args: argparse.Namespace) -> Detector:
    """Coarse-stage detector for crop mode / two-stage.

    Keeps only the context (parent) classes from the profile or
    ``--primary-classes``; an empty list keeps every class.
    """
    from .ultralytics_pt import UltralyticsDetector

    keep = set(parse_class_ids(args.primary_classes))
    return UltralyticsDetector(
        args.primary_weights,
        name='primary',
        imgsz=args.primary_imgsz,
        device=args.device,
        conf=args.primary_conf,
        iou=args.nms_iou,
        keep_classes=keep or None,
    )


def _two_stage(args: argparse.Namespace) -> Detector:
    """A fixed coarse+fine cascade as the system under test itself.

    Unlike ``--mode crop`` (which wraps any backend), the pair here is the
    model being scored: an Ultralytics coarse stage and an Ultralytics or
    Triton fine stage.
    """
    from .two_stage import TwoStageDetector
    from .ultralytics_pt import UltralyticsDetector

    primary = _primary_detector(args)
    if args.secondary_backend == 'triton':
        from .triton_trt import TritonYoloDetector

        secondary: Detector = TritonYoloDetector(
            url=args.triton_url,
            model=args.triton_model,
            input_size=640,
            conf=args.conf_floor,
            iou=args.nms_iou,
        )
    else:
        secondary = UltralyticsDetector(
            args.weights,
            name='secondary',
            imgsz=args.secondary_imgsz,
            device=args.device,
            conf=args.conf_floor,
            iou=args.nms_iou,
        )
    return TwoStageDetector(primary, secondary, name=args.name or 'two-stage', nms_iou=args.nms_iou)


register_backend('ultralytics', _ultralytics)
register_backend('triton', _triton)
register_backend('two-stage', _two_stage)
register_backend('onnxruntime', _onnxruntime)
register_backend('coreml', _coreml)


def _build_backend(args: argparse.Namespace) -> Detector:
    """Build the system under test, honoring --mode (full vs crop).

    ``--mode crop`` wraps ANY backend in a coarse-detector->crop->detector
    pipeline (parent object -> part, any two-stage cascade);
    ``--mode full`` runs the detector directly on the source frame.
    """
    inner = get_backend(args.backend)(args)
    if args.mode == 'crop' and args.backend != 'two-stage':
        from .two_stage import TwoStageDetector

        return TwoStageDetector(
            _primary_detector(args),
            inner,
            name=args.name or inner.name,
            primary_conf=args.primary_conf,
            nms_iou=args.nms_iou,
        )
    return inner
