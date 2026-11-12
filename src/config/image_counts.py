"""Settings of the stored per-image counts."""

from __future__ import annotations

from typing import Final

from src.config.env import env_int


SWEEP_PER_TICK_ENV: Final = 'OP_IMAGE_COUNTS_SWEEP_PER_TICK'
SWEEP_PER_TICK_DEFAULT: Final = 500


def sweep_per_tick() -> int:
    """Images without ``counts_at`` that one drain tick counts besides the marked ones (the
    default is one items aggregation page); ``0`` leaves them to the backfill job."""
    return env_int(SWEEP_PER_TICK_ENV, SWEEP_PER_TICK_DEFAULT, minimum=0)
