"""Detector backends for the license_plate example (plate-only public models).

Imported via the example profile's ``backend_modules``; registers ``lpdnet``
and ``open-image-models`` into the harness backend registry. The detector
modules themselves load lazily, so importing this package needs neither
onnxruntime nor the ``open-image-models`` package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.curation.bakeoff.backends.registry import register_backend


if TYPE_CHECKING:
    import argparse

    from scripts.curation.bakeoff.backends.base import Detector


def _device(args: argparse.Namespace) -> str:
    return 'cpu' if str(args.device) == 'cpu' else 'cuda'


def build_lpdnet(args: argparse.Namespace) -> Detector:
    """NVIDIA TAO LPDNet; ``backend_options.variant`` = ``usa`` (default) or ``ccpd``."""
    from .lpdnet import LpdnetDetector

    return LpdnetDetector(
        args.weights,
        name=args.name,
        variant=str(args.backend_options.get('variant', 'usa')),
        conf=args.conf_floor,
        iou=args.nms_iou,
        device=_device(args),
    )


def build_open_image_models(args: argparse.Namespace) -> Detector:
    """ankandrew/open-image-models plate detector (``backend_options.model_name`` optional)."""
    from .open_image_models import OpenImageModelsDetector

    extra = (
        {'model_name': args.backend_options['model_name']}
        if 'model_name' in args.backend_options
        else {}
    )
    return OpenImageModelsDetector(
        name=args.name, conf=args.conf_floor, device=_device(args), **extra
    )


register_backend('lpdnet', build_lpdnet)
register_backend('open-image-models', build_open_image_models)
