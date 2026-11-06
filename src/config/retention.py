"""How long rows live before the reaper deletes them (02a section 9).

Constants, not environment, until a deployment needs to tune one. Each lane that owns a table
adds its own line here and its delete to the reaper's retention rule.
"""

from __future__ import annotations

from datetime import timedelta


EVENTS_MAX_AGE = timedelta(hours=24)
TERMINAL_JOBS_MAX_AGE = timedelta(days=90)
WORKER_CAPABILITIES_MAX_AGE = timedelta(days=1)
GPU_DEVICES_MAX_AGE = timedelta(days=1)
DELETE_BATCH = 10_000
