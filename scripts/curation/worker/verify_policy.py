"""The bound project's ``region_verify`` policy as the region worker reads it."""

from __future__ import annotations

import time
from typing import Any

from opensearchpy.exceptions import OpenSearchException

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.curation.vlm_policy_store import get_vlm_policy


logger = get_logger('curation_worker')

POLICY_TTL_S = 30.0
_cache: dict[str, tuple[float, str]] = {}


def reset_cache() -> None:
    _cache.clear()


async def project_region_verify(opensearch: Any) -> str:
    """The bound project's ``region_verify`` policy (``vlm`` when never set), re-read
    at most every :data:`POLICY_TTL_S`. A read that fails keeps the last value."""
    slug = get_curation_config().project_slug
    now = time.monotonic()
    cached = _cache.get(slug)
    if cached is not None and now - cached[0] < POLICY_TTL_S:
        return cached[1]
    try:
        value = (await get_vlm_policy(opensearch)).region_verify
    except (OpenSearchException, OSError) as exc:
        logger.warning('region_verify_policy_read_failed', project=slug, error=str(exc))
        return cached[1] if cached is not None else 'vlm'
    _cache[slug] = (now, value)
    return value


__all__ = ['project_region_verify', 'reset_cache']
