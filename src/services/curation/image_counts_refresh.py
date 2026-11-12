"""Keep the stored per-image counts (``counts``, ``counts_at``) equal to the items (R-13).

A writer that changed an item calls :func:`enqueue_image_counts` (the OCC helpers do it for every
count-relevant item update they commit); the ``op.periodic.image_counts_drain`` tick calls
:func:`drain` every few seconds. Both sides are idempotent: an image's counts are always
recomputed from its items with the same aggregation the cards use
(:func:`~src.services.curation.image_counts.item_facts`), never incremented, so a lost or repeated
mark costs a recount, not a wrong number.

Images that were never counted (no ``counts_at``: just ingested, or from before this field) are
picked up by :func:`sweep_uncounted`, a bounded number per tick, so no ingest path has to remember
to mark its images. The ``image_counts_backfill`` job runs the same recount over the whole index.

* bulk update: https://docs.opensearch.org/latest/api-reference/document-apis/bulk/
* refresh: https://docs.opensearch.org/latest/api-reference/index-apis/refresh/
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from src.clients.image_counts_dirty import DirtyImages, mark_dirty
from src.config.curation_accessors import images_index, items_index
from src.config.image_counts import sweep_per_tick
from src.core.logging import get_logger
from src.services.curation.image_counts import item_facts
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)

#: Images recounted per items aggregation and per bulk write.
BATCH: Final = 500
#: A drain stops starting batches after this long (the tick's soft limit is 120 s).
DRAIN_BUDGET_S: Final = 90.0


class CountsWriteError(RuntimeError):
    """The images index refused some count writes; the images stay marked for the next drain."""


@dataclass
class DrainReport:
    recounted: int = 0
    swept: int = 0


async def enqueue_image_counts(image_ids: Iterable[str]) -> int:
    """Mark ``image_ids`` for a recount (the entry point of a writer that changed their items)."""
    return await mark_dirty(image_ids)


async def recount_images(opensearch: Any, image_ids: list[str]) -> int:
    """Recompute the counts of ``image_ids`` from their items and store them; returns how many
    image documents were updated (an image deleted meanwhile is skipped).

    Raises :class:`CountsWriteError` when the images index refuses a write for any other reason.
    """
    if not image_ids:
        return 0
    counted_at = now_iso()
    facts = await item_facts(opensearch, image_ids, include_test=False, box_limit=0)
    body: list[dict[str, Any]] = []
    for image_id in image_ids:
        body.append({'update': {'_index': images_index(), '_id': image_id, 'retry_on_conflict': 3}})
        body.append(
            {'doc': {'counts': facts[image_id].counts.model_dump(), 'counts_at': counted_at}}
        )
    resp = await opensearch.bulk(body=body, refresh=False)
    updated = 0
    failed: list[str] = []
    for image_id, item in zip(image_ids, resp.get('items') or [], strict=True):
        status = int((item.get('update') or {}).get('status', 0))
        if status in (200, 201):
            updated += 1
        elif status != 404:
            failed.append(image_id)
    if failed:
        msg = f'{len(failed)} of {len(image_ids)} image count writes were refused'
        raise CountsWriteError(msg)
    return updated


async def sweep_uncounted(opensearch: Any, limit: int | None = None) -> int:
    """Recount up to ``limit`` images that have no ``counts_at`` (default: ``sweep_per_tick()``);
    returns how many were found."""
    limit = sweep_per_tick() if limit is None else limit
    if limit == 0:
        return 0
    resp = await opensearch.search(
        index=images_index(),
        body={
            'size': limit,
            'sort': [{'image_id': 'asc'}],
            '_source': False,
            'query': {'bool': {'must_not': {'exists': {'field': 'counts_at'}}}},
        },
    )
    ids = [str(h['_id']) for h in resp['hits']['hits']]
    await recount_images(opensearch, ids)
    return len(ids)


async def drain(
    opensearch: Any, dirty: DirtyImages, *, budget_s: float = DRAIN_BUDGET_S
) -> DrainReport:
    """Recount every marked image of the bound project, then sweep the never-counted ones."""
    report = DrainReport()
    deadline = time.monotonic() + budget_s
    if await dirty.claim():
        # A write is marked after it committed, so a refresh now makes every marked write visible
        # to the aggregation (the items index refreshes once a second on its own).
        await opensearch.indices.refresh(index=items_index())
        while time.monotonic() < deadline:
            batch = await dirty.batch(BATCH)
            if not batch:
                break
            await recount_images(opensearch, batch)
            await dirty.done(batch)
            report.recounted += len(batch)
    report.swept = await sweep_uncounted(opensearch)
    if report.recounted or report.swept:
        logger.info('image_counts_drained', recounted=report.recounted, swept=report.swept)
    return report
