"""The job state machine: statuses, allowed transitions, terminal and active sets.

Pure data. The database ``CHECK`` on ``jobs.status``, the transition trigger and the partial
unique index that allows one active job per exclusive group are all generated from here, so a
status added in one place cannot be forgotten in another.
"""

from __future__ import annotations


QUEUED = 'queued'
RUNNING = 'running'
PAUSED = 'paused'
CANCELLING = 'cancelling'
COMPLETED = 'completed'
COMPLETED_WITH_ERRORS = 'completed_with_errors'
FAILED = 'failed'
CANCELLED = 'cancelled'
INTERRUPTED = 'interrupted'

STATUSES: tuple[str, ...] = (
    QUEUED,
    RUNNING,
    PAUSED,
    CANCELLING,
    COMPLETED,
    COMPLETED_WITH_ERRORS,
    FAILED,
    CANCELLED,
    INTERRUPTED,
)

# No further transition; ``finished_at`` is set exactly for these.
TERMINAL: frozenset[str] = frozenset({COMPLETED, COMPLETED_WITH_ERRORS, FAILED, CANCELLED})

# Counted by the singleton index. ``interrupted`` is deliberately not active: a new start is
# allowed while an interrupted job waits to be resumed.
ACTIVE: frozenset[str] = frozenset({QUEUED, RUNNING, PAUSED, CANCELLING})

TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {
        (QUEUED, RUNNING),
        (QUEUED, CANCELLED),
        (QUEUED, FAILED),
        (QUEUED, PAUSED),
        (RUNNING, COMPLETED),
        (RUNNING, COMPLETED_WITH_ERRORS),
        (RUNNING, FAILED),
        (RUNNING, CANCELLING),
        (RUNNING, PAUSED),
        (RUNNING, QUEUED),
        (RUNNING, INTERRUPTED),
        (CANCELLING, INTERRUPTED),
        (CANCELLING, CANCELLED),
        (PAUSED, QUEUED),
        (INTERRUPTED, QUEUED),
        (PAUSED, CANCELLED),
        (INTERRUPTED, CANCELLED),
        (INTERRUPTED, FAILED),
    }
)


__all__ = [
    'ACTIVE',
    'CANCELLED',
    'CANCELLING',
    'COMPLETED',
    'COMPLETED_WITH_ERRORS',
    'FAILED',
    'INTERRUPTED',
    'PAUSED',
    'QUEUED',
    'RUNNING',
    'STATUSES',
    'TERMINAL',
    'TRANSITIONS',
]
