"""The ``image_counts_backfill`` job: store the counts of every image of the bound project.

Walks the images in id order, a page at a time, and recounts each page from its items with
:func:`~src.services.curation.image_counts_refresh.recount_images`, the same recount the drain
runs. It exists so a project that already has images gets ``counts_at`` everywhere in one
pausable pass; once every image has it, the gallery sort and ``label_state`` filter read the
stored counts. The cursor (the last image id done) is checkpointed after each page, so a resumed
delivery continues after it, and re-running the job is harmless (a recount never increments).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from src.config.curation_accessors import images_index
from src.services.curation.image_counts_refresh import BATCH, recount_images
from src.services.jobs.bound import bound_job_project
from src.tasks.runtime.context import JobResult, Paused, Yield


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext

STAGE: Final = 'counting images'


async def _page(client: Any, after: str | None) -> list[str]:
    body: dict[str, Any] = {
        'size': BATCH,
        'sort': [{'image_id': 'asc'}],
        '_source': False,
        'query': {'match_all': {}},
    }
    if after is not None:
        body['search_after'] = [after]
    resp = await client.search(index=images_index(), body=body)
    return [str(h['_id']) for h in resp['hits']['hits']]


async def run(ctx: JobContext) -> JobResult | Yield | Paused:
    """``image_counts_backfill``: recount every image, resumable and pausable."""
    saved = dict(ctx.resume_from or {})
    after: str | None = saved.get('after')
    counted = int(saved.get('counted', 0))
    async with bound_job_project(ctx):
        client = await ctx.client('opensearch')
        total = int((await client.count(index=images_index()))['count'])
        while True:
            if ctx.cancelled():
                return JobResult({'cancelled': True, 'counted': counted})
            if ctx.paused():
                return Paused()
            if ctx.slice_due():
                return Yield()
            ids = await _page(client, after)
            if not ids:
                return JobResult({'counted': counted})
            await recount_images(client, ids)
            after = ids[-1]
            counted += len(ids)
            await ctx.checkpoint({'after': after, 'counted': counted})
            await ctx.progress(counted, total, STAGE, {'counted': counted})
