"""CPU NMS for a YOLOv5-style raw-output ensemble/backbone detector.

Some backbone detectors are served by Triton with NMS applied
client-side rather than baked into the exported graph — Triton returns
the raw output ``[batch, N, 5 + nc]`` (4 cx/cy/w/h + 1 objectness + nc
class scores, in pixel coordinates relative to the network input) and
this module applies NMS.

To guarantee bit-exact behavior with a given trained model, this
imports a YOLOv5-fork's ``non_max_suppression`` directly rather than
reimplementing it — some forks diverge subtly from upstream Ultralytics
in tie-breaking / clamping. The fork location is configurable via the
``DETECTION_YOLOV5_FORK`` env var; the default assumes a vendored
checkout at ``./external/yolov5`` relative to the process's working
directory.

Default thresholds follow common YOLOv5 defaults
(``utils/general.py``): ``conf_thres=0.25``, ``iou_thres=0.45``,
``max_det=300``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
import torch


_DEFAULT_FORK_PATH: Final[str] = './external/yolov5'

_DEFAULT_CONF: Final[float] = 0.25
_DEFAULT_IOU: Final[float] = 0.45
_DEFAULT_MAX_DET: Final[int] = 300


def _ensure_fork_on_path() -> Path:
    fork_dir = Path(os.environ.get('DETECTION_YOLOV5_FORK', _DEFAULT_FORK_PATH))
    if not fork_dir.is_dir():
        raise RuntimeError(
            f'YOLOv5 fork not found at {fork_dir}. Set DETECTION_YOLOV5_FORK to its location.'
        )
    if str(fork_dir) not in sys.path:
        sys.path.insert(0, str(fork_dir))
    return fork_dir


_ensure_fork_on_path()
# Imported lazily after path injection so the rest of the module loads even
# when the fork is missing (allows unit tests to mock).
from utils.general import non_max_suppression as _fork_nms  # noqa: E402


def apply_ensemble_nms(
    raw_output: np.ndarray | torch.Tensor,
    conf_thres: float = _DEFAULT_CONF,
    iou_thres: float = _DEFAULT_IOU,
    max_det: int = _DEFAULT_MAX_DET,
    classes: list[int] | None = None,
    agnostic: bool = False,
) -> list[list[dict]]:
    """Apply YOLOv5-fork NMS to a raw ensemble-detector output.

    Args:
        raw_output: shape `[B, N, 5 + nc]` (cx, cy, w, h, obj_conf, +nc class
            scores). PIXEL space coordinates relative to the network input
            (e.g. 1280x1280). NumPy or torch.
        conf_thres, iou_thres, max_det, classes, agnostic: passed through to
            the fork's `non_max_suppression`.

    Returns:
        List of per-image detection lists. Each detection is a dict with
        keys `box` ([x1, y1, x2, y2] in pixel space), `score`, `class_id`.
        Caller normalizes coords vs. input size for storage.
    """
    if isinstance(raw_output, np.ndarray):
        prediction = torch.from_numpy(raw_output)
    else:
        prediction = raw_output

    fork_out = _fork_nms(
        prediction,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=classes,
        agnostic=agnostic,
        max_det=max_det,
    )

    results: list[list[dict]] = []
    for det in fork_out:
        per_image: list[dict] = []
        if det is None or det.numel() == 0:
            results.append(per_image)
            continue
        for row in det.tolist():
            x1, y1, x2, y2, score, cls = row
            per_image.append(
                {
                    'box': [x1, y1, x2, y2],
                    'score': float(score),
                    'class_id': int(cls),
                }
            )
        results.append(per_image)
    return results


__all__ = ['apply_ensemble_nms']
