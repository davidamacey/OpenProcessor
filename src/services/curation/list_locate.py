"""Where an item sits in a sorted list, and the cursors that resume the list around it.

``before_query`` turns a sort and one document into the query for the items that sort strictly
before it, so the position is one ``count`` whatever the depth. The cursor of the page that starts
at the item is the sort values of the item before it, found by reading the same query once in the
reversed order. Every cursored list that offers a locate (`/crops`, `/review/{tab}`, `/images`)
goes through :func:`locate_item`.

* search_after: https://docs.opensearch.org/latest/search-plugins/searching-data/paginate/
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from src.services.curation import review_sorts
from src.services.curation.list_cursor import encode_cursor


class UnlocatableSortError(ValueError):
    """A sort clause this module can't express as a range query."""


def _sort_key(entry: dict[str, Any] | str) -> tuple[str, str]:
    if isinstance(entry, str):
        return entry, 'asc'
    ((field, spec),) = entry.items()
    if field.startswith('_'):
        raise UnlocatableSortError(f'cannot locate under a {field} sort')
    if isinstance(spec, dict) and spec.get('missing', '_last') != '_last':
        raise UnlocatableSortError(f'cannot locate under missing={spec["missing"]!r}')
    order = spec.get('order', 'asc') if isinstance(spec, dict) else str(spec)
    return field, order


def _equal(field: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {'bool': {'must_not': {'exists': {'field': field}}}}
    if isinstance(value, int | float) and not isinstance(value, bool):
        return {'range': {field: {'gte': value, 'lte': value}}}
    return {'term': {field: value}}


def _strictly_before(field: str, order: str, value: Any) -> dict[str, Any] | None:
    """Items that sort before ``value`` on ``field`` (missing values sort last)."""
    if value is None:
        return {'exists': {'field': field}}
    return {'range': {field: {'lt' if order == 'asc' else 'gt': value}}}


def _rank_clause(ranks: set[int]) -> dict[str, Any]:
    """Items whose ``vlm_confidence`` ranks (see ``VLM_CONFIDENCE_RANK_SORT``) are in ``ranks``."""
    known = list(review_sorts.VLM_CONFIDENCE_ORDER)
    should: list[dict[str, Any]] = []
    if 0 in ranks:
        should.append({'bool': {'must_not': [{'terms': {'vlm_confidence': known}}]}})
    present = [known[r - 1] for r in sorted(ranks) if r > 0]
    if present:
        should.append({'terms': {'vlm_confidence': present}})
    return {'bool': {'should': should, 'minimum_should_match': 1}}


def _source_rank(source: dict[str, Any]) -> int:
    value = source.get('vlm_confidence')
    order = review_sorts.VLM_CONFIDENCE_ORDER
    return order.index(value) + 1 if value in order else 0


def before_query(sort: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    """Query for items sorting strictly before the doc ``source`` under ``sort``."""
    # Per sort key: (items equal to source, items strictly before source).
    keys: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for entry in sort:
        if entry == review_sorts.VLM_CONFIDENCE_RANK_SORT:
            rank = _source_rank(source)
            keys.append((_rank_clause({rank}), _rank_clause(set(range(rank))) if rank else None))
            continue
        field, order = _sort_key(entry)
        value = source.get(field)
        if isinstance(value, list):
            raise UnlocatableSortError(f'cannot locate by multi-valued field {field}')
        keys.append((_equal(field, value), _strictly_before(field, order, value)))
    alternatives: list[dict[str, Any]] = [
        {'bool': {'filter': [*(eq for eq, _b in keys[:i]), before]}}
        for i, (_eq, before) in enumerate(keys)
        if before is not None
    ]
    return {'bool': {'should': alternatives, 'minimum_should_match': 1}}


def reverse_sort(sort: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``sort`` read backwards: every order flipped and ``missing`` moved to the other end."""
    flip = {'asc': 'desc', 'desc': 'asc', '_last': '_first', '_first': '_last'}
    out: list[dict[str, Any]] = []
    for entry in sort:
        ((key, spec),) = entry.items()
        spec = copy.deepcopy(spec)
        if isinstance(spec, str):
            out.append({key: flip[spec]})
            continue
        spec['order'] = flip[spec.get('order', 'asc')]
        if 'missing' in spec:
            spec['missing'] = flip.get(spec['missing'], spec['missing'])
        out.append({key: spec})
    return out


@dataclass(frozen=True)
class ListLocation:
    """``index`` is 0-based. ``cursor_before`` starts a page at the item (``None`` for the first
    row); ``cursor_after`` resumes right after it."""

    index: int
    total: int
    cursor_before: str | None
    cursor_after: str

    def as_wire(self) -> dict[str, Any]:
        return {
            'index': self.index,
            'total': self.total,
            'cursor_before': self.cursor_before,
            'cursor_after': self.cursor_after,
        }


async def locate_item(
    opensearch: Any,
    *,
    index: str,
    query: dict[str, Any],
    sort: list[dict[str, Any]],
    sort_id: str,
    source: dict[str, Any],
    id_field: str = 'crop_id',
) -> ListLocation | None:
    """The position of the document ``source`` in ``query`` ordered by ``sort`` (which ends on the
    unique ``id_field``), or ``None`` when ``query`` does not match it.

    Raises :class:`UnlocatableSortError` for a sort a range query cannot express."""
    member = {'bool': {'filter': [query, {'term': {id_field: source[id_field]}}]}}
    own = await opensearch.search(
        index=index, body={'query': member, 'size': 1, 'sort': sort, '_source': False}
    )
    hits = own['hits']['hits']
    if not hits:
        return None
    before = {'bool': {'filter': [query, before_query(sort, source)]}}
    rank = int((await opensearch.count(index=index, body={'query': before}))['count'])
    total = int((await opensearch.count(index=index, body={'query': query}))['count'])
    cursor_before = None
    if rank:
        prev = await opensearch.search(
            index=index,
            body={'query': before, 'size': 1, 'sort': reverse_sort(sort), '_source': False},
        )
        cursor_before = encode_cursor(sort_id, prev['hits']['hits'][0]['sort'])
    return ListLocation(
        index=rank,
        total=total,
        cursor_before=cursor_before,
        cursor_after=encode_cursor(sort_id, hits[0]['sort']),
    )


__all__ = ['ListLocation', 'UnlocatableSortError', 'before_query', 'locate_item', 'reverse_sort']
