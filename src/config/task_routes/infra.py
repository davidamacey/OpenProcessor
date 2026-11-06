"""Routes of the infrastructure tasks."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {
    'op.infra.noop': TaskQueues.JOBS,
    'op.infra.sleep_noop': TaskQueues.JOBS,
}
