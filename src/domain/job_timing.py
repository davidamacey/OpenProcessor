"""Timing of the job lease, in seconds (02b section 3.1).

A running job holds a lease on its ``jobs`` row. The heartbeat thread renews it every
``HEARTBEAT_TICK_S``; a holder whose last good renew is older than ``LEASE_TTL_S - FENCE_MARGIN_S``
stops writing, so it is silent before a reaper can take the job over at ``LEASE_TTL_S``.
"""

from __future__ import annotations

from src.config.env import env_int


# Each value is an operator setting so a test stack can shorten the timers; the defaults are the
# production values (pinned by tests/jobs/test_timing_constants.py).
LEASE_TTL_S = env_int('OP_JOB_LEASE_TTL_S', 120, minimum=10)
"""A lease whose ``lease_expires_at`` is past is expired; the reaper may act on its row."""

HEARTBEAT_TICK_S = env_int('OP_JOB_HEARTBEAT_TICK_S', 15, minimum=1)
"""Seconds between renews (eight ticks fit in one TTL)."""

FENCE_MARGIN_S = env_int('OP_JOB_FENCE_MARGIN_S', 30, minimum=1)
"""A holder stops writing once its last good renew is older than ``LEASE_TTL_S - FENCE_MARGIN_S``."""

PROGRESS_MIN_INTERVAL_S = 1
"""At most one progress write per job per interval, except on a stage change."""
