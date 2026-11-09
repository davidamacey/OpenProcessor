"""The ``source_backfill`` job: walk the bound project's images in id order, in slices.

Three steps, chosen by ``params['steps']`` (default all, in this order): ``content`` and
``capture`` run on every page of the walk (the image's content key; its capture metadata), then
``location_purge`` brings the stored coordinates within the project's location policy (a switch
to a stricter policy starts the job with that step alone, ``services/curation/location_policy``).

Pausable and sliced (``slice_s``): the handler returns :class:`Yield` at a page boundary once
its slice ran, and :class:`Paused` when a pause was requested; the cursor (the last image id
done) is checkpointed after each page, so a resumed delivery continues after it and the counts
carry over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from src.config.curation_accessors import images_index
from src.services.curation.location_policy_store import get_location_policy
from src.services.jobs.bound import bound_job_project
from src.services.source_backfill.capture import CaptureCounts, record_captures
from src.services.source_backfill.content import ContentCounts, record_contents
from src.services.source_backfill.location_purge import purge_pass, settle
from src.tasks.runtime.context import JobResult, Paused, Yield


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.tasks.runtime.context import JobContext

PAGE = 200
STEP_CONTENT: Final = 'content'
STEP_CAPTURE: Final = 'capture'
STEP_LOCATION_PURGE: Final = 'location_purge'
STEPS: Final = (STEP_CONTENT, STEP_CAPTURE, STEP_LOCATION_PURGE)


class UnknownStepError(ValueError):
    """``params['steps']`` names a step this job does not have."""


async def _page(
    client: AsyncOpenSearch, index: str, after: str | None
) -> list[tuple[str, str, bool]]:
    """``(image_id, image_path, has a capture parser stamp)`` of the next page."""
    body: dict[str, Any] = {
        'size': PAGE,
        'sort': [{'image_id': 'asc'}],
        '_source': ['image_path', 'capture.parser'],
        'query': {'match_all': {}},
    }
    if after is not None:
        body['search_after'] = [after]
    resp = await client.search(index=index, body=body)
    out: list[tuple[str, str, bool]] = []
    for hit in resp['hits']['hits']:
        src = hit.get('_source') or {}
        stamped = bool((src.get('capture') or {}).get('parser'))
        out.append((hit['_id'], str(src.get('image_path') or ''), stamped))
    return out


def _steps(ctx: JobContext) -> tuple[str, ...]:
    asked = ctx.params.get('steps')
    if asked is None:
        return STEPS
    unknown = sorted(set(asked) - set(STEPS))
    if unknown:
        msg = f'unknown source_backfill steps: {unknown}'
        raise UnknownStepError(msg)
    return tuple(step for step in STEPS if step in asked)


@dataclass
class _State:
    """The walk's cursor and counts: what a checkpoint carries and a resumed delivery restores."""

    after: str | None = None
    counts: ContentCounts = field(default_factory=ContentCounts)
    captured: CaptureCounts = field(default_factory=CaptureCounts)
    scanned: int = 0
    purged: int = 0
    settled: bool = False
    walked: bool = False

    @classmethod
    def restore(cls, saved: dict[str, Any]) -> _State:
        return cls(
            after=saved.get('after'),
            counts=ContentCounts(**saved.get('counts', {})),
            captured=CaptureCounts(**saved.get('capture', {})),
            scanned=int(saved.get('scanned', 0)),
            purged=int(saved.get('purged', 0)),
            settled=bool(saved.get('settled', False)),
            walked=bool(saved.get('walked', False)),
        )

    def cursor(self) -> dict[str, Any]:
        return {
            'after': self.after,
            'counts': self.counts.as_dict(),
            'capture': {k.removeprefix('capture_'): v for k, v in self.captured.as_dict().items()},
            'scanned': self.scanned,
            'purged': self.purged,
            'settled': self.settled,
            'walked': self.walked,
        }

    def tally(self) -> dict[str, Any]:
        return {
            'scanned': self.scanned,
            **self.counts.as_dict(),
            **self.captured.as_dict(),
            'location_purged': self.purged,
        }


async def _interrupted(ctx: JobContext, state: _State) -> JobResult | Yield | Paused | None:
    if ctx.cancelled():
        return JobResult({'cancelled': True, **state.tally()})
    if ctx.paused():
        return Paused()
    if ctx.slice_due():
        return Yield()
    return None


async def _walk(
    ctx: JobContext, state: _State, client: AsyncOpenSearch, project_id: Any, steps: tuple[str, ...]
) -> JobResult | Yield | Paused | None:
    """The page walk (content and capture); ``None`` once every page is done."""
    policy = await get_location_policy() if STEP_CAPTURE in steps else 'store'
    shown = STEP_CAPTURE if STEP_CAPTURE in steps else STEP_CONTENT
    while True:
        if (stop := await _interrupted(ctx, state)) is not None:
            return stop
        page = await _page(client, images_index(), state.after)
        if not page:
            return None
        if STEP_CONTENT in steps:
            async with ctx.session() as session:
                state.counts.add(await record_contents(session, project_id, [p[:2] for p in page]))
        if STEP_CAPTURE in steps:
            state.captured.add(await record_captures(client, policy, page))
        state.after = page[-1][0]
        state.scanned += len(page)
        await ctx.checkpoint(state.cursor())
        await ctx.progress(state.scanned, None, shown, state.tally())


async def _purge(
    ctx: JobContext, state: _State, client: AsyncOpenSearch
) -> JobResult | Yield | Paused | None:
    """The location-purge passes; ``None`` once a pass changes nothing."""
    if not state.settled and await get_location_policy() != 'store':
        await settle()
        state.settled = True
    while True:
        if (stop := await _interrupted(ctx, state)) is not None:
            return stop
        changed = await purge_pass(client, await get_location_policy())
        state.purged += changed
        state.walked = True
        await ctx.checkpoint(state.cursor())
        await ctx.progress(state.purged, None, STEP_LOCATION_PURGE, state.tally())
        if changed == 0:
            return None


async def run(ctx: JobContext) -> JobResult | Yield | Paused:
    """``source_backfill``: content keys, capture metadata, then the location policy."""
    steps = _steps(ctx)
    state = _State.restore(dict(ctx.resume_from or {}))
    async with bound_job_project(ctx) as project_id:
        client = await ctx.client('opensearch')
        walking = bool({STEP_CONTENT, STEP_CAPTURE} & set(steps)) and not state.walked
        if walking and (stop := await _walk(ctx, state, client, project_id, steps)) is not None:
            return stop
        if STEP_LOCATION_PURGE in steps and (stop := await _purge(ctx, state, client)) is not None:
            return stop
    return JobResult(
        state.tally(),
        with_errors=state.counts.unreadable + state.counts.changed + state.captured.unreadable > 0,
    )
