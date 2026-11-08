"""Chain the item level after an image-level run (research plan 5.2 and 7.1).

The item level groups only inside the image level's `leak` groups, so a new image run can change
which items may group at all. When one finishes:

* a full image build or a regroup (`touched` is None) starts a full item build, because any leak
  group may have moved;
* an incremental image run marks the items of the images it regrouped pending (`group_changed`)
  and starts the item rescore, or a full item build when the item level was never built.

The start is refused with 409 `job_busy` while an item job is active; the pending rows then wait
for the next rescore, so nothing is lost. The idempotency key is the image run id, so a redelivered
image job starts one item job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config.curation_accessors import items_index
from src.core.exceptions import RequestRejectedError
from src.db.repos import near_dup as repo
from src.db.session import async_session_factory
from src.services.curation.clustering.pit_pages import iter_pages
from src.services.jobs.service import start_job
from src.services.near_duplicates import pending


if TYPE_CHECKING:
    import uuid
    from collections.abc import AsyncIterator, Collection

    from src.services.near_duplicates.runs import Env

REQUESTER = 'near-dup-chain'
IMAGE_CHUNK = 1000


async def _items_of(env: Env, image_ids: Collection[str]) -> AsyncIterator[list[str]]:
    wanted = sorted(image_ids)
    for start in range(0, len(wanted), IMAGE_CHUNK):
        query = {'terms': {'image_id': wanted[start : start + IMAGE_CHUNK]}}
        async for hits in iter_pages(
            env.client, index=items_index(), query=query, source=['image_id']
        ):
            yield [h['_id'] for h in hits]


async def mark_items_of(env: Env, image_ids: Collection[str]) -> int:
    """Mark every item standing on one of ``image_ids`` pending at the item level."""
    marked = 0
    async for item_ids in _items_of(env, image_ids):
        async with env.scope() as session:
            marked += await pending.enqueue_pending(
                session, env.project_id, 'item', item_ids, 'group_changed'
            )
    return marked


async def after_image_run(
    env: Env, run_id: uuid.UUID, *, touched: Collection[str] | None
) -> uuid.UUID | None:
    """Start the item level that follows an image-level run; the job id, or None when an item
    job is already active (its pending rows are picked up by the next rescore)."""
    if env.plan.level != 'image':
        return None
    kind = 'near_dup_score'
    if touched is not None:
        async with env.scope() as session:
            built = await repo.current_run(session, env.project_id, 'item') is not None
        if built:
            await mark_items_of(env, touched)
            kind = 'near_dup_rescore'
    try:
        async with async_session_factory()() as session:
            started = await start_job(
                session,
                kind,
                env.project_id,
                params={'level': 'item'},
                parent_job_id=env.job_id,
                requested_by=REQUESTER,
                idempotency_key=f'near_dup_chain:{run_id}',
            )
    except RequestRejectedError as exc:
        if isinstance(exc.detail, dict) and exc.detail.get('error') == 'job_busy':
            return None
        raise
    return started.job.id
