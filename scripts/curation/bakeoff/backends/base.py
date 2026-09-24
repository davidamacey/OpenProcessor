"""Backend abstraction: a detector maps an RGB frame to target-class boxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """One predicted target-class box in absolute pixel coords (xyxy) + score.

    Coordinates are in the original frame's pixel space (not normalized,
    not letterboxed) so the harness can score every backend the same way.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def coco_bbox(self) -> list[float]:
        """``[x, y, w, h]`` in absolute pixels (COCO convention)."""
        return [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]


@runtime_checkable
class Detector(Protocol):
    """A single-target-class detector that runs in its own native runtime.

    Implementations load weights at construction and expose a stable
    ``name`` (used in reports) and a ``runtime`` tag (e.g. ``ultralytics``,
    ``onnxruntime``, ``triton-trt``) so latency can be read in context.
    Detection must run at a LOW score threshold (e.g. 0.001) so the full
    precision/recall curve is available to COCOeval; the operating-point
    filter (conf=0.25) is applied later by the harness, not the backend.
    """

    name: str
    runtime: str

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        """Return all target-class boxes for one HxWx3 uint8 RGB frame."""
        ...
