"""Task routes of the `gpu_eval` family."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {'op.gpu_eval.bakeoff': TaskQueues.GPU_EVAL}
