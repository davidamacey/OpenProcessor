"""How many ids a batch route writes inside the request before it queues a job instead."""

from __future__ import annotations

from src.config.env import env_int


DEFAULT_BATCH_SYNC_MAX = 500
"""One page of the bulk OCC loop (``OCC_BULK_PAGE_SIZE``): the largest write that is one ``_mget`` and one ``_bulk``."""


def batch_sync_max() -> int:
    """``OP_BATCH_SYNC_MAX``: a batch of at most this many ids is written in the request (200);
    a larger one is queued as a job (202)."""
    return env_int('OP_BATCH_SYNC_MAX', DEFAULT_BATCH_SYNC_MAX, minimum=1)
