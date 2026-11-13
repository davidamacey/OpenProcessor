"""Task routes of lane 4A: the kinds run on the queue their job kind names (label operations a
person waits on are on ``interactive``, the rest on ``jobs``)."""

from __future__ import annotations

from src.config.job_kinds.families_a import KINDS


ROUTES: dict[str, str] = {kind.task_name: kind.queue for kind in KINDS}
