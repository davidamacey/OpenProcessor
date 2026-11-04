"""Uploads are routed by the checkpoint's head kind, not by ``end2end`` (U-C2).

A YOLO26 checkpoint loads with ``end2end=False`` on Ultralytics 8.4.197, so the old flag sent it
to the YOLO11 EfficientNMS toolchain. The one-to-one head is identified by its ``one2one_cv2``
branch (YOLO26 and YOLOv10); every other Ultralytics head is exported with external NMS.
Untrained checkpoints built from the packaged YAMLs carry the same head modules as the official
weights, so the routing tests run offline; the ``families`` test uses the real downloads.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest
import torch

from src.services.model_export import validate_pytorch_model


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ('yaml_name', 'head_kind'),
    [
        ('yolo26n.yaml', 'one2one'),
        ('yolov10n.yaml', 'one2one'),
        ('yolo11n.yaml', 'nms'),
        ('yolov5nu.yaml', 'nms'),
    ],
)
def test_head_kind_follows_the_head_module(tmp_path: Path, yaml_name: str, head_kind: str) -> None:
    from ultralytics import YOLO

    ckpt = tmp_path / 'm.pt'
    YOLO(yaml_name).save(str(ckpt))

    ok, msg, info = validate_pytorch_model(ckpt)

    assert ok, msg
    assert info['head_kind'] == head_kind
    assert 'end2end' not in info
    assert 'ultralytics_version' in info


class DetectionModel(torch.nn.Module):
    """Stands in for the original yolov5 repo's class; pickled under its module name."""

    __module__ = 'models.yolo'


def _original_yolov5_checkpoint(path: Path) -> None:
    """A checkpoint that pickles a ``models.yolo`` class, as the original yolov5 repo writes."""
    pkg, mod = ModuleType('models'), ModuleType('models.yolo')
    mod.DetectionModel = DetectionModel  # type: ignore[attr-defined]
    pkg.yolo = mod  # type: ignore[attr-defined]
    sys.modules.update({'models': pkg, 'models.yolo': mod})
    try:
        torch.save({'model': DetectionModel(), 'pad': torch.zeros(2000)}, path)
    finally:
        del sys.modules['models'], sys.modules['models.yolo']


def test_original_yolov5_checkpoint_is_refused_with_the_yolov5u_message(tmp_path: Path) -> None:
    ckpt = tmp_path / 'yolov5n.pt'
    _original_yolov5_checkpoint(ckpt)

    ok, msg, info = validate_pytorch_model(ckpt)

    assert not ok
    assert info == {}
    assert 'YOLOv5u' in msg
    assert 'yolov5su.pt' in msg


@pytest.mark.families
@pytest.mark.parametrize(
    ('weight', 'head_kind'),
    [
        ('yolo26n.pt', 'one2one'),
        ('yolov10n.pt', 'one2one'),
        ('yolo11n.pt', 'nms'),
        ('yolov5nu.pt', 'nms'),
    ],
)
def test_official_weights_route_by_head_kind(op_weights, weight: str, head_kind: str) -> None:
    ok, msg, info = validate_pytorch_model(op_weights(weight))

    assert ok, msg
    assert info['head_kind'] == head_kind


@pytest.mark.families
def test_official_original_yolov5_weights_are_refused(op_weights) -> None:
    ok, msg, _ = validate_pytorch_model(op_weights('yolov5n.pt'))

    assert not ok
    assert 'YOLOv5u' in msg
