"""Build the detector under test from parsed ``run`` CLI args.

Moved out of ``run.py`` so the CLI module stays about the scoring flow.
Backends import lazily: each pulls in its own runtime (Ultralytics, ONNX
Runtime, tritonclient, coremltools) only when selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import argparse

    from .base import Detector


def _build_detector(args: argparse.Namespace, backend: str) -> Detector:
    """Build ONE model detector (the model under test) for a backend."""
    if backend == 'ultralytics':
        from .ultralytics_pt import UltralyticsDetector

        return UltralyticsDetector(
            args.weights,
            name=args.name,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf_floor,
            iou=args.nms_iou,
        )
    if backend == 'triton':
        from .triton_trt import TritonYoloDetector

        return TritonYoloDetector(
            url=args.triton_url,
            model=args.triton_model,
            name=args.name,
            input_size=args.imgsz,
            conf=args.conf_floor,
            iou=args.nms_iou,
        )
    if backend == 'open-image-models':
        from .open_image_models import OpenImageModelsDetector

        return OpenImageModelsDetector(
            name=args.name,
            conf=args.conf_floor,
            device='cpu' if str(args.device) == 'cpu' else 'cuda',
        )
    if backend == 'lpdnet':
        from .lpdnet import LpdnetDetector

        return LpdnetDetector(
            args.weights,
            name=args.name,
            variant=str(args.backend_options.get('variant', 'usa')),
            conf=args.conf_floor,
            iou=args.nms_iou,
            device='cpu' if str(args.device) == 'cpu' else 'cuda',
        )
    if backend == 'onnxruntime':
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
    if backend == 'coreml':
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
    raise SystemExit(f'unknown / not-yet-wired backend: {backend!r}')


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


def _build_backend(args: argparse.Namespace) -> Detector:
    """Build the system under test, honoring --mode (full vs crop).

    ``--mode crop`` wraps ANY backend in a coarse-detector->crop->detector
    pipeline (parent object -> part, any two-stage cascade);
    ``--mode full`` runs the detector directly on the source frame. The
    ``two-stage`` backend is a fixed, non-wrapped variant of that same
    cascade for when the coarse+fine pair is the system under test itself
    (not a wrapper around one of the other single backends above).
    """
    if args.backend == 'two-stage':
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
        return TwoStageDetector(
            primary, secondary, name=args.name or 'two-stage', nms_iou=args.nms_iou
        )

    inner = _build_detector(args, args.backend)
    if args.mode == 'crop':
        from .two_stage import TwoStageDetector

        return TwoStageDetector(
            _primary_detector(args),
            inner,
            name=args.name or inner.name,
            primary_conf=args.primary_conf,
            nms_iou=args.nms_iou,
        )
    return inner
