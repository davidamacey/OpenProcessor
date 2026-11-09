"""Task routes of the `gpu_train` family."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {'op.gpu_train.train': TaskQueues.GPU_TRAIN}
