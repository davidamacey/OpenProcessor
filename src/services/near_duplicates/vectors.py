"""Read the vectors a near-duplicate run groups: by id (mget) or the whole level (PIT pages)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.services.curation.clustering.pit_pages import iter_pages
from src.services.near_duplicates.candidates import unit_rows


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from opensearchpy import AsyncOpenSearch

MGET_CHUNK = 500
ITEM_FIELDS = ('image_id', 'detector_class_name')
TERMS_CHUNK = 1000


@dataclass(frozen=True)
class MemberDoc:
    """One group member: its id, unit vector, and for an item its image and detector class."""

    member_id: str
    vector: np.ndarray
    image_id: str | None = None
    detector_class: str | None = None


def _member(doc_id: str, source: dict[str, Any], field: str, *, item: bool) -> MemberDoc | None:
    raw = source.get(field)
    if not raw:
        return None
    return MemberDoc(
        doc_id,
        unit_rows([raw])[0],
        source.get('image_id') if item else None,
        source.get('detector_class_name') if item else None,
    )


async def fetch_members(
    client: AsyncOpenSearch, *, index: str, field: str, ids: Iterable[str], item: bool
) -> dict[str, MemberDoc]:
    """The members of ``ids`` that exist and carry a vector; the others are simply absent."""
    wanted = sorted(set(ids))
    found: dict[str, MemberDoc] = {}
    includes = [field, *ITEM_FIELDS] if item else [field]
    for start in range(0, len(wanted), MGET_CHUNK):
        response = await client.mget(
            index=index,
            body={'ids': wanted[start : start + MGET_CHUNK]},
            _source_includes=includes,
        )
        for doc in response.get('docs') or []:
            if not doc.get('found'):
                continue
            member = _member(doc['_id'], doc.get('_source') or {}, field, item=item)
            if member is not None:
                found[member.member_id] = member
    return found


async def scan_members(
    client: AsyncOpenSearch, *, index: str, field: str, item: bool
) -> AsyncIterator[list[MemberDoc]]:
    """Pages of every member of the index that has a vector."""
    source = [field, *ITEM_FIELDS] if item else [field]
    async for hits in iter_pages(
        client, index=index, query={'exists': {'field': field}}, source=source
    ):
        page = [_member(h['_id'], h.get('_source') or {}, field, item=item) for h in hits]
        yield [m for m in page if m is not None]


def stack(members: Iterable[MemberDoc]) -> tuple[list[str], np.ndarray]:
    """Members sorted by id, with their vectors as one matrix."""
    ordered = sorted(members, key=lambda m: m.member_id)
    if not ordered:
        return [], np.zeros((0, 0), dtype=np.float32)
    return [m.member_id for m in ordered], np.stack([m.vector for m in ordered])


async def scan_items_of_images(
    client: AsyncOpenSearch, *, index: str, field: str, image_ids: Iterable[str]
) -> list[MemberDoc]:
    """Every item with a vector that stands on one of ``image_ids``."""
    wanted = sorted(set(image_ids))
    found: list[MemberDoc] = []
    for start in range(0, len(wanted), TERMS_CHUNK):
        query = {
            'bool': {
                'filter': [
                    {'terms': {'image_id': wanted[start : start + TERMS_CHUNK]}},
                    {'exists': {'field': field}},
                ]
            }
        }
        async for hits in iter_pages(
            client, index=index, query=query, source=[field, *ITEM_FIELDS]
        ):
            page = [_member(h['_id'], h.get('_source') or {}, field, item=True) for h in hits]
            found.extend(m for m in page if m is not None)
    return found
