"""``op.periodic.cache_prune``: keep the tmpfs crop cache under its size cap.

Ingest writes crops to ``<crop_cache_dir>/<slug>/<crop_id>.jpg`` and never evicts; this tick evicts
the oldest files across every project's directory once the whole cache exceeds
``crop_cache_max_bytes``. The deployment-wide ``CACHE_PRUNE`` advisory lock keeps two ticks from
racing each other's deletes (it replaces the ``.prune.lock`` file lock). The tick must run in a
container that mounts the same crop cache directory as ingest.
"""

from __future__ import annotations

import asyncio

from src.config.advisory_locks import CACHE_PRUNE
from src.config.curation_accessors import base_curation_config
from src.services.curation.source_image_cache import prune_crop_cache
from src.services.periodic.runner import tick_lock


async def tick() -> tuple[int, int]:
    """``(files removed, bytes removed)``; ``(0, 0)`` when another tick holds the lock."""
    config = base_curation_config()
    async with tick_lock(CACHE_PRUNE) as held:
        if not held:
            return (0, 0)
        return await asyncio.to_thread(
            prune_crop_cache, config.crop_cache_dir, config.crop_cache_max_bytes
        )
