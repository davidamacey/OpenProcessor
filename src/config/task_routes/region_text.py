"""Task routes of the registration-plate text family."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {
    'op.region_text.reread': TaskQueues.TEXT_READ,
    'op.region_text.consensus': TaskQueues.JOBS,
}
