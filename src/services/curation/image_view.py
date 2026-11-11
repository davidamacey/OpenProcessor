"""The image gallery: `GET .../images` as a view model.

The images index is queried directly (source, position, near-duplicate group, sort, cursor). A
predicate about an image's *items* (the shared item filter, a label state, the hold-out) cannot be
asked of the images index, so it is resolved to the images that have such items first, at most
`JOIN_MAX_IMAGES` of them (`join_truncated` says when it was more), the same capped join the map
uses. Counts and card boxes for the page then come from one items aggregation (`image_counts`).

* search_after: https://docs.opensearch.org/latest/search-plugins/searching-data/paginate/
* terms query: https://docs.opensearch.org/latest/query-dsl/term/terms/
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from src.config.curation_accessors import images_index, items_index
from src.config.project_context import project_api_base
from src.schemas.curation.images import HiddenCounts, ImageCard, ImagePage, SortOption
from src.services.curation.image_counts import item_facts
from src.services.curation.image_facts import (
    IMAGE_SOURCE_FIELDS,
    file_name,
    near_dup_of,
    served_capture,
    taken_at,
    thumbnail_url,
)
from src.services.curation.item_filter import ItemFilter, item_filter_clauses, visibility_clauses
from src.services.curation.map_cells import JOIN_MAX_IMAGES, LOCATION_FIELD, geo_box_query


if TYPE_CHECKING:
    from src.schemas.curation.map import MapBounds
    from src.services.embedding_spaces.readers import WriteFields

NearDupFilter = Literal['member', 'representative', 'none']
LabelState = Literal['has_unlabeled', 'all_labeled', 'no_items']

DEFAULT_SORT: Final = 'indexed_at:desc'
LIMIT_MAX: Final = 200
LIMIT_DEFAULT: Final = 60

_TAKEN = 'capture.taken_local'
_SORTS: Final[dict[str, tuple[str, list[dict[str, Any]]]]] = {
    'indexed_at:desc': ('Newest indexed', [{'indexed_at': {'order': 'desc'}}]),
    'indexed_at:asc': ('Oldest indexed', [{'indexed_at': {'order': 'asc'}}]),
    'taken_at:desc': ('Newest captured', [{_TAKEN: {'order': 'desc', 'missing': '_last'}}]),
    'taken_at:asc': ('Oldest captured', [{_TAKEN: {'order': 'asc', 'missing': '_last'}}]),
    'file_name:asc': ('Path (A to Z)', [{'image_path': {'order': 'asc'}}]),
}
SORT_OPTIONS: Final = [SortOption(id=k, label=label) for k, (label, _) in _SORTS.items()]

_HOLDOUT = {'term': {'test_holdout': True}}
_UNLABELED = {'bool': {'must_not': {'exists': {'field': 'class_id'}}}}


class ImageQueryError(ValueError):
    """The request names a sort, cursor or combination the route does not serve."""


@dataclass(frozen=True)
class ImageQuery:
    item_filter: ItemFilter
    source: str | None = None
    has_location: bool | None = None
    geo_box: MapBounds | None = None
    in_holdout: bool | None = None
    near_dup: NearDupFilter | None = None
    label_state: LabelState | None = None
    sort: str = DEFAULT_SORT
    cursor: str | None = None
    limit: int = LIMIT_DEFAULT
    include_test: bool = False

    def narrowed(self) -> bool:
        return not self.item_filter.is_empty() or any(
            v is not None
            for v in (
                self.source,
                self.has_location,
                self.geo_box,
                self.in_holdout,
                self.near_dup,
                self.label_state,
            )
        )


def _encode_cursor(sort: str, after: list[Any]) -> str:
    raw = json.dumps({'sort': sort, 'after': after}, separators=(',', ':')).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _decode_cursor(cursor: str, sort: str) -> list[Any]:
    try:
        body = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        after = body['after']
        cursor_sort = body['sort']
    except (binascii.Error, ValueError, KeyError, TypeError) as exc:
        msg = 'cursor is not one this route served'
        raise ImageQueryError(msg) from exc
    if cursor_sort != sort or not isinstance(after, list):
        msg = 'cursor belongs to a different sort'
        raise ImageQueryError(msg)
    return after


async def _item_image_ids(opensearch: Any, clauses: list[dict[str, Any]]) -> tuple[list[str], bool]:
    """The images with an item matching ``clauses`` and whether there were more than we use."""
    resp = await opensearch.search(
        index=items_index(),
        body={
            'size': 0,
            'query': {'bool': {'filter': clauses}} if clauses else {'match_all': {}},
            'aggs': {'images': {'terms': {'field': 'image_id', 'size': JOIN_MAX_IMAGES}}},
        },
    )
    agg = resp['aggregations']['images']
    return [b['key'] for b in agg['buckets']], int(agg.get('sum_other_doc_count', 0)) > 0


def _image_level_clauses(q: ImageQuery, policy: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if q.source is not None:
        out.append({'term': {'source': q.source}})
    located = {'exists': {'field': LOCATION_FIELD}}
    if q.has_location is not None:
        if policy == 'drop':
            out.append({'match_all': {}} if not q.has_location else {'match_none': {}})
        else:
            out.append(located if q.has_location else {'bool': {'must_not': located}})
    if q.geo_box is not None:
        out.append({'match_none': {}} if policy == 'drop' else geo_box_query(q.geo_box))
    group = {'exists': {'field': 'near_dup_group_id'}}
    representative = {'term': {'near_dup_is_representative': True}}
    if q.near_dup == 'member':
        out.append({'bool': {'filter': [group], 'must_not': [representative]}})
    elif q.near_dup == 'representative':
        out.append(representative)
    elif q.near_dup == 'none':
        out.append({'bool': {'must_not': [group]}})
    return out


async def _image_query(
    opensearch: Any, q: ImageQuery, embedding: WriteFields, policy: str
) -> tuple[dict[str, Any], bool]:
    """The images-index query for ``q`` and whether an item join was truncated."""
    f = q.item_filter
    if q.label_state == 'no_items' and not f.is_empty():
        msg = 'label_state=no_items cannot be combined with an item filter'
        raise ImageQueryError(msg)
    item_clauses = [
        *item_filter_clauses(f, embedding=embedding),
        *visibility_clauses(
            include_test=q.include_test, include_excluded='excluded' in f.review_status
        ),
    ]
    positive: list[list[dict[str, Any]]] = []
    negative: list[list[dict[str, Any]]] = []
    if q.label_state == 'has_unlabeled':
        positive.append([*item_clauses, _UNLABELED])
    elif not f.is_empty() or q.label_state == 'all_labeled':
        positive.append(item_clauses)
    if q.label_state == 'all_labeled':
        negative.append([*item_clauses, _UNLABELED])
    if q.label_state == 'no_items':
        negative.append(item_clauses)
    if q.in_holdout is True:
        positive.append([_HOLDOUT])
    elif q.in_holdout is False:
        negative.append([_HOLDOUT])
    filters = _image_level_clauses(q, policy)
    must_not: list[dict[str, Any]] = []
    truncated = False
    for clauses in positive:
        ids, more = await _item_image_ids(opensearch, clauses)
        filters.append({'terms': {'image_id': ids}})
        truncated = truncated or more
    for clauses in negative:
        ids, more = await _item_image_ids(opensearch, clauses)
        must_not.append({'terms': {'image_id': ids}})
        truncated = truncated or more
    body: dict[str, Any] = {'filter': filters, 'must_not': must_not}
    return {'bool': {k: v for k, v in body.items() if v}} if filters or must_not else {
        'match_all': {}
    }, truncated


def _card(src: dict[str, Any], prefix: str, policy: str, facts: Any) -> ImageCard:
    image_id = str(src.get('image_id') or '')
    capture = served_capture(src, policy)
    return ImageCard(
        image_id=image_id,
        file_name=file_name(src),
        thumbnail_url=thumbnail_url(prefix, image_id),
        width=src.get('width'),
        height=src.get('height'),
        source=src.get('source') or '',
        indexed_at=src.get('indexed_at'),
        taken_at=taken_at(capture),
        has_location='location' in capture,
        counts=facts.counts,
        near_dup=near_dup_of(src),
        boxes=facts.boxes,
        boxes_truncated=facts.boxes_truncated,
    )


async def image_page(
    opensearch: Any, q: ImageQuery, embedding: WriteFields, policy: str
) -> ImagePage:
    """One page of images with their counts and boxes."""
    if q.sort not in _SORTS:
        msg = f'sort must be one of {sorted(_SORTS)}'
        raise ImageQueryError(msg)
    query, join_truncated = await _image_query(opensearch, q, embedding, policy)
    body: dict[str, Any] = {
        'size': q.limit + 1,
        'track_total_hits': True,
        'query': query,
        'sort': [*_SORTS[q.sort][1], {'image_id': {'order': 'asc'}}],
        '_source': {'includes': list(IMAGE_SOURCE_FIELDS)},
    }
    if q.cursor is not None:
        body['search_after'] = _decode_cursor(q.cursor, q.sort)
    resp = await opensearch.search(index=images_index(), body=body)
    hits = resp['hits']['hits']
    kept = hits[: q.limit]
    next_cursor = _encode_cursor(q.sort, kept[-1]['sort']) if len(hits) > q.limit and kept else None
    ids = [str(h['_source']['image_id']) for h in kept]
    facts = await item_facts(opensearch, ids, include_test=q.include_test)
    prefix = project_api_base()
    cards = [_card(h['_source'], prefix, policy, facts[i]) for h, i in zip(kept, ids, strict=True)]
    empty = None
    if not cards:
        empty = 'no_match' if q.narrowed() else 'no_images'
    return ImagePage(
        items=cards,
        next_cursor=next_cursor,
        total=int(resp['hits']['total']['value']),
        join_truncated=join_truncated,
        sort_applied=q.sort,
        sort_options=SORT_OPTIONS,
        hidden=HiddenCounts(
            test_holdout=0 if q.include_test else sum(f.counts.in_holdout for f in facts.values()),
            excluded=sum(f.excluded for f in facts.values()),
        ),
        empty_reason=empty,
    )
