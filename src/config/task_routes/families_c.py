"""Task routes of lane 4C."""

from __future__ import annotations

from src.config.queue import TaskQueues


ROUTES: dict[str, str] = {
    'op.families_c.project_delete': TaskQueues.JOBS,
    'op.families_c.open_vocab_drain': TaskQueues.OPEN_VOCAB,
    'op.families_c.directory_ingest': TaskQueues.JOBS,
}
