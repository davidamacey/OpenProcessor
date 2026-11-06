"""The beat schedule: every periodic task with its interval (02b section 4.1).

Each entry carries ``expires`` equal to its interval, so a backlog behind a stalled consumer is
dropped instead of replayed (Celery periodic tasks guide). ``src/tasks/app.py`` applies the
schedule when the app's configuration loads, never at import. The reaper is the only entry of
lane 3A; the periodic lane adds the rest to ``BEAT_SCHEDULE``.
"""

from __future__ import annotations

from typing import Any


REAPER_INTERVAL_S = 30

BEAT_SCHEDULE: dict[str, dict[str, Any]] = {
    'reaper': {
        'task': 'op.periodic.reaper',
        'schedule': float(REAPER_INTERVAL_S),
        'options': {'expires': REAPER_INTERVAL_S},
    },
}


def apply_schedule(source: Any, **_: Any) -> None:
    """``on_after_configure`` receiver: put the schedule into the loading configuration."""
    source['beat_schedule'] = BEAT_SCHEDULE
