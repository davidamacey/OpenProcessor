"""Task routes of the `gpu_export` family."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {'op.export.model_export': TaskQueues.GPU_EXPORT}
