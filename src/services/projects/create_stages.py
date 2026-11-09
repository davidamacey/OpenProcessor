"""Per-stage wall time of a project create (#439, 4D-0).

Each stage feeds ``op_pipeline_stage_seconds`` through :func:`~src.utils.stage_timing.stage_timer`
(the #414 pattern) and is also kept by name, so the ``project_create`` job can return the table in
its result: the job runs in a worker process, whose Prometheus registry the API does not serve.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

from src.utils.stage_timing import stage_timer


if TYPE_CHECKING:
    from collections.abc import Iterator


class CreateStages:
    """Seconds per stage of one create, in the order the stages ran."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            with stage_timer(name):
                yield
        finally:
            self.seconds[name] = round(time.perf_counter() - start, 4)
