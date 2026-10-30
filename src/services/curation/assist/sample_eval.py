"""Ask the VLM about a set of crops: the chunked execution an assist run and the
auto-label loop share.

The ids are split by images per call; each chunk runs the documented stages (verify
the boxes, then the item's class and attributes, or the chosen boxes' attributes)
under a semaphore the caller owns, so concurrent callers can share one cap on
in-flight requests. Writes are the chunk's own (:mod:`~src.services.curation.assist.chunk`).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.services.curation.assist.box_attributes import box_attributes_chunk
from src.services.curation.assist.chunk import ChunkOutcome, process_chunk
from src.services.curation.assist.verify import verify_chunk


if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from opensearchpy import AsyncOpenSearch

    from src.services.curation.assist.slot_write import RunMeta
    from src.services.curation.vlm_class_batch import VlmContext


def _sum(outcomes: list[ChunkOutcome]) -> ChunkOutcome:
    total = ChunkOutcome()
    for o in outcomes:
        for name in vars(total):
            setattr(total, name, getattr(total, name) + getattr(o, name))
    return total


async def _chunk(
    opensearch: AsyncOpenSearch,
    ctx: VlmContext,
    chunk: list[str],
    *,
    actions: Collection[str],
    box_ids: dict[str, list[str]] | None,
    meta: RunMeta,
) -> ChunkOutcome:
    """One chunk, in the documented order: verify the boxes, then the item's class and
    attributes (or the chosen boxes' attributes)."""
    parts: list[ChunkOutcome] = []
    if 'verify' in actions:
        parts.append(
            await verify_chunk(
                opensearch, ctx.labeler, chunk, box_ids=box_ids, pack_stamp=ctx.pack_stamp
            )
        )
    if box_ids is not None:
        if 'attributes' in actions:
            parts.append(
                await box_attributes_chunk(
                    opensearch, ctx.labeler, chunk, box_ids=box_ids, meta=meta
                )
            )
    elif 'classify' in actions or 'attributes' in actions:
        parts.append(
            await process_chunk(
                opensearch,
                ctx,
                chunk,
                classify='classify' in actions,
                attributes='attributes' in actions,
                meta=meta,
            )
        )
    return _sum(parts)


async def ask_sample(
    opensearch: AsyncOpenSearch,
    ctx: VlmContext,
    ids: list[str],
    *,
    actions: Collection[str],
    box_ids: dict[str, list[str]] | None,
    images_per_call: int,
    meta: RunMeta,
    semaphore: asyncio.Semaphore,
    should_stop: Callable[[], bool] = lambda: False,
    on_chunk: Callable[[int], None] | None = None,
) -> ChunkOutcome:
    """Run ``actions`` over ``ids`` and return the summed outcome.

    ``should_stop`` is checked as each chunk gets its turn on ``semaphore``; a chunk
    that has not started then is skipped, one in flight finishes. ``on_chunk`` gets the
    size of each chunk that completed."""
    size = max(1, images_per_call)
    chunks = [ids[i : i + size] for i in range(0, len(ids), size)]

    async def gated(chunk: list[str]) -> ChunkOutcome | None:
        async with semaphore:
            if should_stop():
                return None
            out = await _chunk(opensearch, ctx, chunk, actions=actions, box_ids=box_ids, meta=meta)
            if on_chunk is not None:
                on_chunk(len(chunk))
            return out

    results = await asyncio.gather(*[gated(c) for c in chunks])
    return _sum([r for r in results if r is not None])


__all__ = ['ask_sample']
