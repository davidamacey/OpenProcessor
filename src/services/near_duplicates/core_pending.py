"""Mark core images stale for the single-writer ``near_dup_core_group`` job, in the writer's own
transaction.

Core ingest and the core delete path call this before the OpenSearch write they describe (see
``pending.py`` for why the row goes first). The write is one short ``core_tx`` transaction; the
rows belong to no project.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.db.session import async_session_factory, core_tx
from src.services.near_duplicates.pending import enqueue_core_pending


if TYPE_CHECKING:
    from collections.abc import Iterable


async def record_pending(image_ids: Iterable[str], reason: str) -> int:
    """Insert (or refresh) one core pending row per image; returns the number of images."""
    ids = list(image_ids)
    if not ids:
        return 0
    async with (
        async_session_factory()() as session,
        core_tx(session, 'core near-duplicate pending'),
    ):
        return await enqueue_core_pending(session, ids, reason)
