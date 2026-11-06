"""The ``source_backfill`` job: walk the bound project's images in id order, in slices.

Pausable and sliced (``slice_s``): the handler returns :class:`Yield` at a page boundary once
its slice ran, and :class:`Paused` when a pause was requested; the cursor (the last image id
done) is checkpointed after each page, so a resumed delivery continues after it and the counts
carry over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.curation_accessors import images_index
from src.services.jobs.bound import bound_job_project
from src.services.source_backfill.content import ContentCounts, record_contents
from src.tasks.runtime.context import JobResult, Paused, Yield


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.tasks.runtime.context import JobContext

PAGE = 200
STEP_CONTENT = 'content'


async def _page(client: AsyncOpenSearch, index: str, after: str | None) -> list[tuple[str, str]]:
    body: dict[str, Any] = {
        'size': PAGE,
        'sort': [{'image_id': 'asc'}],
        '_source': ['image_path'],
        'query': {'match_all': {}},
    }
    if after is not None:
        body['search_after'] = [after]
    resp = await client.search(index=index, body=body)
    return [
        (h['_id'], str((h.get('_source') or {}).get('image_path') or ''))
        for h in resp['hits']['hits']
    ]


async def run(ctx: JobContext) -> JobResult | Yield | Paused:
    """``source_backfill``: record the content key of every stored image that lacks one."""
    saved = ctx.resume_from or {}
    after: str | None = saved.get('after')
    counts = ContentCounts(**saved.get('counts', {}))
    scanned = int(saved.get('scanned', 0))
    async with bound_job_project(ctx) as project_id:
        client = await ctx.client('opensearch')
        index = images_index()
        while True:
            if ctx.cancelled():
                return JobResult({'cancelled': True, 'scanned': scanned, **counts.as_dict()})
            if ctx.paused():
                return Paused()
            if ctx.slice_due():
                return Yield()
            page = await _page(client, index, after)
            if not page:
                break
            async with ctx.session() as session:
                counts.add(await record_contents(session, project_id, page))
            after = page[-1][0]
            scanned += len(page)
            await ctx.checkpoint({'after': after, 'counts': counts.as_dict(), 'scanned': scanned})
            await ctx.progress(scanned, None, STEP_CONTENT, counts.as_dict())
    result = {'scanned': scanned, **counts.as_dict()}
    return JobResult(result, with_errors=counts.unreadable + counts.changed > 0)
