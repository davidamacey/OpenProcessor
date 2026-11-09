"""Timing of the job lease, in seconds (02b section 3.1).

A running job holds a lease on its ``jobs`` row. The heartbeat thread renews it every
``HEARTBEAT_TICK_S``; a holder whose last good renew is older than ``LEASE_TTL_S - FENCE_MARGIN_S``
stops writing, so it is silent before a reaper can take the job over at ``LEASE_TTL_S``.
"""

from __future__ import annotations


LEASE_TTL_S = 120
"""A lease whose ``lease_expires_at`` is past is expired; the reaper may act on its row."""

HEARTBEAT_TICK_S = 15
"""Seconds between renews (eight ticks fit in one TTL)."""

FENCE_MARGIN_S = 30
"""A holder stops writing once its last good renew is older than ``LEASE_TTL_S - FENCE_MARGIN_S``."""

PROGRESS_MIN_INTERVAL_S = 1
"""At most one progress write per job per interval, except on a stage change."""
