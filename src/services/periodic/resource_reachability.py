"""``op.periodic.resource_reachability``: probe the monitoring services once for every API process.

Every 30 s the tick probes the enabled services of :mod:`src.services.resource_links` and writes the
answer to Valkey (``op:resource_reachability``, 90 s expiry); the API only reads it.
"""

from __future__ import annotations

from src.services.resource_links import probe_enabled, publish_reachability


async def tick() -> None:
    await publish_reachability(await probe_enabled())
