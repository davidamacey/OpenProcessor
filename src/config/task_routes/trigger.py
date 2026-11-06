"""Task routes of the `trigger` family."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {
    'op.autolabel.run': TaskQueues.AUTOLABEL,
    'op.assist_loop.run': TaskQueues.ASSIST_LOOP,
    'op.vlm.label_batch': TaskQueues.JOBS,
    'op.vlm.verify_region_batch': TaskQueues.JOBS,
}
