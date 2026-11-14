"""One request to the engine and its answer, as the front and the pipeline exchange them."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import numpy as np

    from src.domain import frame_engine_wire as wire


@dataclass
class EngineRequest:
    data: bytes
    plan: wire.EnginePlan
    boxes: np.ndarray | None = None
    arrived: float = field(default_factory=time.perf_counter)
    #: ``time.perf_counter()`` value by which the caller has its answer; no model call outlives it.
    deadline: float = math.inf
    #: The stage the request last entered (``frame_engine.stage_metrics.STAGES``): logged with a
    #: request the front answers at its deadline.
    stage: str = 'queue'

    def enter(self, stage: str) -> None:
        self.stage = stage


@dataclass
class EngineResult:
    status: str
    decode_route: str
    branches: dict[str, tuple[wire.BranchState, str]]
    #: Output tensor name to array (``crop_jpegs``: a list of bytes), the plan's branches only.
    outputs: dict[str, Any] = field(default_factory=dict)
