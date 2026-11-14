"""One request to the engine and its answer, as the front and the pipeline exchange them."""

from __future__ import annotations

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


@dataclass
class EngineResult:
    status: str
    decode_route: str
    branches: dict[str, tuple[wire.BranchState, str]]
    #: Output tensor name to array (``crop_jpegs``: a list of bytes), the plan's branches only.
    outputs: dict[str, Any] = field(default_factory=dict)
