"""The core exact-duplicate lookup: which stored core image holds these exact bytes.

``imohash`` finds the candidate and SHA-256 confirms (``content_key``); the rows are in
``image_contents`` with no project, so two concurrent ingests of one file meet at the advisory
lock instead of racing past an index that has not refreshed (#358). A row whose images document
is gone (a crash between the claim and the index write) is released and the file treated as new,
as the curation ingest does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.clients.opensearch.names import IndexName
from src.services.near_duplicates import content_key


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opensearchpy import AsyncOpenSearch


@dataclass(frozen=True)
class Holder:
    """The stored core image that holds a file's exact bytes."""

    image_id: str
    image_path: str


async def resolve_holders(
    client: AsyncOpenSearch, entries: Sequence[tuple[str, bytes]]
) -> list[Holder | None]:
    """One answer per ``(imohash, bytes)``: the holding image, or ``None`` for a new file."""
    verdicts = await content_key.resolve_many_core(entries)
    ids = sorted({v.holder for v in verdicts if v.holder is not None})
    live: set[str] = set()
    if ids:
        found = await client.mget(index=IndexName.GLOBAL.value, body={'ids': ids}, _source=False)
        live = {d['_id'] for d in found.get('docs') or [] if d.get('found')}
        orphans = [i for i in ids if i not in live]
        if orphans:
            await content_key.release_core(orphans)
    return [
        Holder(v.holder, v.holder_path or v.holder)
        if v.holder is not None and v.holder in live
        else None
        for v in verdicts
    ]
