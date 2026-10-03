"""A selection of items to act on: explicit ids or a filter, optionally capped.

The "run on selection" routes (embed, label, cluster, export, reprocess) take
an :class:`ItemSelection`. The caller either names crop ids or sends the same
:class:`~src.services.curation.item_filter.ItemFilter` the list routes accept,
plus an optional cap: the ``limit`` largest boxes, or a ``limit``-sized random
sample (deterministic for a given ``seed``). A filter selection is resolved
here, once, so the action runs on exactly the ids a dry run counted.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.services.curation.item_filter import ItemFilter, item_filter_clauses, visibility_clauses


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

MAX_SELECTION_IDS = 5000
# A filter that matches more than this cannot be capped or sampled here: narrow it.
MAX_SELECTION_SCAN = 500_000
_PAGE = 5000

Sample = Literal['random', 'largest']


class SelectionError(ValueError):
    """The selection is malformed or too large to resolve (a 4xx, never a silent truncation)."""


class ItemSelection(BaseModel):
    """Exactly one of ``crop_ids`` / ``filter``; ``limit`` / ``sample`` apply to a filter."""

    model_config = ConfigDict(extra='forbid')

    crop_ids: list[str] | None = Field(default=None, max_length=MAX_SELECTION_IDS)
    filter: ItemFilter | None = None
    limit: int | None = Field(default=None, ge=1)
    sample: Sample | None = None
    seed: int = 0
    # A filter selection includes hold-out and human-ignored items only when asked.
    include_test: bool = False
    include_excluded: bool = False

    @model_validator(mode='after')
    def _one_form(self) -> Self:
        if (self.crop_ids is None) == (self.filter is None):
            msg = 'exactly one of crop_ids or filter is required'
            raise ValueError(msg)
        if self.crop_ids is not None and (self.limit is not None or self.sample is not None):
            msg = 'limit and sample apply to a filter selection, not to explicit crop_ids'
            raise ValueError(msg)
        if self.sample is not None and self.limit is None:
            msg = 'sample needs a limit'
            raise ValueError(msg)
        if self.filter is not None and self.filter.is_empty() and self.limit is None:
            msg = 'an empty filter selects every item; set a filter field or a limit'
            raise ValueError(msg)
        return self


def selection_query(sel: ItemSelection) -> dict[str, Any]:
    """The items query for a filter selection (``ValueError`` from a bad band)."""
    assert sel.filter is not None
    clauses = [
        *item_filter_clauses(sel.filter),
        *visibility_clauses(
            include_test=sel.include_test,
            include_excluded=sel.include_excluded or 'excluded' in sel.filter.review_status,
        ),
    ]
    return {'bool': {'filter': clauses}} if clauses else {'match_all': {}}


async def _scan(
    opensearch: AsyncOpenSearch, query: dict[str, Any], index: str
) -> list[tuple[str, float]]:
    """``(crop_id, area)`` of every match, ``crop_id`` order, bounded."""
    out: list[tuple[str, float]] = []
    cursor: list[Any] | None = None
    while True:
        body: dict[str, Any] = {
            'size': _PAGE,
            '_source': ['crop_area_norm'],
            'query': query,
            'sort': [{'crop_id': 'asc'}],
        }
        if cursor is not None:
            body['search_after'] = cursor
        hits = ((await opensearch.search(index=index, body=body)).get('hits') or {}).get(
            'hits'
        ) or []
        if not hits:
            break
        out.extend(
            (h['_id'], float((h.get('_source') or {}).get('crop_area_norm') or 0.0)) for h in hits
        )
        if len(out) > MAX_SELECTION_SCAN:
            msg = f'the filter matches more than {MAX_SELECTION_SCAN} items; narrow it'
            raise SelectionError(msg)
        cursor = hits[-1].get('sort')
        if cursor is None or len(hits) < _PAGE:
            break
    return out


async def resolve_ids(
    opensearch: AsyncOpenSearch,
    query: dict[str, Any],
    *,
    index: str,
    limit: int | None = None,
    sample: Sample | None = None,
    seed: int = 0,
) -> list[str]:
    """The ids ``query`` matches, capped by ``limit`` (``largest`` boxes, a
    seeded ``random`` draw, else the first in ``crop_id`` order). A draw is
    returned sorted; ``largest`` keeps its largest-first order."""
    found = await _scan(opensearch, query, index)
    if limit is not None and len(found) > limit:
        if sample == 'random':
            found = sorted(random.Random(seed).sample(found, limit))
        elif sample == 'largest':
            found = sorted(found, key=lambda r: (-r[1], r[0]))[:limit]
        else:
            found = found[:limit]
    return [crop_id for crop_id, _ in found]


async def resolve_selection(
    opensearch: AsyncOpenSearch, sel: ItemSelection, *, index: str
) -> list[str]:
    """The crop ids the selection names, in a stable order (explicit ids keep
    the caller's order, deduplicated)."""
    if sel.crop_ids is not None:
        return list(dict.fromkeys(sel.crop_ids))
    try:
        query = selection_query(sel)
    except ValueError as exc:
        raise SelectionError(str(exc)) from exc
    return await resolve_ids(
        opensearch, query, index=index, limit=sel.limit, sample=sel.sample, seed=sel.seed
    )


__all__ = [
    'MAX_SELECTION_IDS',
    'MAX_SELECTION_SCAN',
    'ItemSelection',
    'SelectionError',
    'resolve_ids',
    'resolve_selection',
    'selection_query',
]
