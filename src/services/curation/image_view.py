"""The image gallery: `GET .../images` as a view model.

The images index is queried directly (source, position, near-duplicate group, sort, cursor). A
predicate about an image's *items* (the shared item filter, a label state, the hold-out) cannot be
asked of the images index, so it is resolved to the images that have such items first, at most
`JOIN_MAX_IMAGES` of them (`join_truncated` says when it was more), the same capped join the map
uses. Counts and card boxes for the page then come from one items aggregation (`image_counts`).

When every image in a list's scope carries its stored counts (`counts_at`, set by the periodic
drain or the `image_counts_backfill` job), `label_state` is a range query on `counts.*` and the
"most unlabelled first" sort is served, so neither needs the capped join. That holds only without
item narrowing (an item filter, held-out items shown): those still resolve through the items.
The stored counts trail the items by at most the drain interval.

* search_after: https://docs.opensearch.org/latest/search-plugins/searching-data/paginate/
* range query: https://docs.opensearch.org/latest/query-dsl/term/range/
* filter aggregation: https://docs.opensearch.org/latest/aggregations/bucket/filter/
* terms query: https://docs.opensearch.org/latest/query-dsl/term/terms/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from src.config.curation_accessors import images_index
from src.config.project_context import project_api_base
from src.schemas.curation.images import ImageCard, ImageLocate, ImagePage
from src.schemas.curation.views import HiddenCounts
from src.services.curation.hidden_counts import hidden_counts
from src.services.curation.image_counts import item_facts
from src.services.curation.image_facts import (
    IMAGE_SOURCE_FIELDS,
    file_name,
    near_dup_of,
    served_capture,
    taken_at,
    thumbnail_url,
)
from src.services.curation.image_join import (
    NearDupFilter,
    image_predicate_clauses,
    item_image_ids,
    near_dup_clauses,
    without_image_predicates,
)
from src.services.curation.item_filter import ItemFilter, item_filter_clauses, visibility_clauses
from src.services.curation.list_cursor import page_body, split_page, with_tiebreak
from src.services.curation.list_locate import locate_item


if TYPE_CHECKING:
    from src.services.embedding_spaces.readers import WriteFields

LabelState = Literal['has_unlabeled', 'all_labeled', 'no_items']

DEFAULT_SORT: Final = 'indexed_at:desc'
LIMIT_MAX: Final = 200
LIMIT_DEFAULT: Final = 60
EMPTY_MESSAGES: Final = {
    'no_images': 'No images yet. Ingest images to see them here.',
    'no_match': 'No images match this filter.',
}

_TAKEN = 'capture.taken_local'
_SORTS: Final[dict[str, tuple[str, list[dict[str, Any]]]]] = {
    'indexed_at:desc': ('Newest indexed', [{'indexed_at': {'order': 'desc'}}]),
    'indexed_at:asc': ('Oldest indexed', [{'indexed_at': {'order': 'asc'}}]),
    'taken_at:desc': ('Newest captured', [{_TAKEN: {'order': 'desc', 'missing': '_last'}}]),
    'taken_at:asc': ('Oldest captured', [{_TAKEN: {'order': 'asc', 'missing': '_last'}}]),
    'file_name:asc': ('Path (A to Z)', [{'image_path': {'order': 'asc'}}]),
    'file_name:desc': ('Path (Z to A)', [{'image_path': {'order': 'desc'}}]),
}
SORT_OPTIONS: Final = [{'id': k, 'label': label} for k, (label, _) in _SORTS.items()]

#: Sorts on the stored counts; served (and accepted) only while every image in scope has them.
_STORED_SORTS: Final[dict[str, tuple[str, list[dict[str, Any]]]]] = {
    'unlabeled_count:desc': ('Most unlabelled first', [{'counts.unlabeled': {'order': 'desc'}}]),
    'unlabeled_count:asc': ('Fewest unlabelled first', [{'counts.unlabeled': {'order': 'asc'}}]),
}
STORED_SORT_OPTIONS: Final = [{'id': k, 'label': label} for k, (label, _) in _STORED_SORTS.items()]
_STORED_LABEL_STATE: Final[dict[str, list[dict[str, Any]]]] = {
    'has_unlabeled': [{'range': {'counts.unlabeled': {'gt': 0}}}],
    'all_labeled': [
        {'range': {'counts.items': {'gt': 0}}},
        {'term': {'counts.unlabeled': 0}},
    ],
    'no_items': [{'term': {'counts.items': 0}}],
}
_UNCOUNTED = {'bool': {'must_not': {'exists': {'field': 'counts_at'}}}}

_HOLDOUT = {'term': {'test_holdout': True}}
_UNLABELED = {'bool': {'must_not': {'exists': {'field': 'class_id'}}}}


class ImageQueryError(ValueError):
    """The request names a sort or combination the route does not serve."""


class ImageCountsNotReadyError(ImageQueryError):
    """A sort on the stored counts was asked while some image in scope has none yet."""


@dataclass(frozen=True)
class ImageViewParams:
    """The image-only list parameters (what ``ImageViewQuery`` collects)."""

    source: str | None = None
    in_holdout: bool | None = None
    near_dup: NearDupFilter | None = None
    label_state: LabelState | None = None
    sort: str | None = None
    include_test: bool = False

    @property
    def sort_id(self) -> str:
        return self.sort or DEFAULT_SORT

    def narrowed(self) -> bool:
        return any(
            v is not None
            for v in (
                self.source,
                self.in_holdout,
                self.near_dup,
                self.label_state,
            )
        )


@dataclass(frozen=True)
class ImageQuery:
    item_filter: ItemFilter
    view: ImageViewParams = ImageViewParams()
    cursor: str | None = None
    limit: int = LIMIT_DEFAULT

    def narrowed(self) -> bool:
        return not self.item_filter.is_empty() or self.view.narrowed()


def _image_level_clauses(q: ImageQuery, policy: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if q.view.source is not None:
        out.append({'term': {'source': q.view.source}})
    out.extend(image_predicate_clauses(q.item_filter, policy=policy))
    out.extend(near_dup_clauses(q.view.near_dup))
    return out


async def counts_ready(opensearch: Any, query: dict[str, Any] | None = None) -> bool:
    """True when every image ``query`` (default: all of them) carries its stored counts."""
    resp = await opensearch.count(
        index=images_index(),
        body={'query': {'bool': {'filter': [query or {'match_all': {}}, _UNCOUNTED]}}},
    )
    return int(resp['count']) == 0


async def _use_stored_label_state(
    opensearch: Any, q: ImageQuery, f: ItemFilter, image_clauses: list[dict[str, Any]]
) -> bool:
    """Whether ``label_state`` can be asked of the stored counts: they describe the images' own
    visible items, so any item narrowing (a filter, held-out items shown) still needs the join,
    and so does a scope that has an uncounted image."""
    if q.view.label_state is None or not f.is_empty() or q.view.include_test:
        return False
    return await counts_ready(opensearch, {'bool': {'filter': image_clauses}})


async def _image_query(
    opensearch: Any, q: ImageQuery, embedding: WriteFields, policy: str
) -> tuple[dict[str, Any], bool]:
    """The images-index query for ``q`` and whether an item join was truncated."""
    f = without_image_predicates(q.item_filter)
    if q.view.label_state == 'no_items' and not f.is_empty():
        msg = 'label_state=no_items cannot be combined with an item filter'
        raise ImageQueryError(msg)
    item_clauses = [
        *item_filter_clauses(f, embedding=embedding),
        *visibility_clauses(
            include_test=q.view.include_test, include_excluded='excluded' in f.review_status
        ),
    ]
    filters = _image_level_clauses(q, policy)
    state = q.view.label_state
    if state is not None and await _use_stored_label_state(opensearch, q, f, filters):
        filters.extend(_STORED_LABEL_STATE[state])
        state = None
    positive: list[list[dict[str, Any]]] = []
    negative: list[list[dict[str, Any]]] = []
    if state == 'has_unlabeled':
        positive.append([*item_clauses, _UNLABELED])
    elif not f.is_empty() or state == 'all_labeled':
        positive.append(item_clauses)
    if state == 'all_labeled':
        negative.append([*item_clauses, _UNLABELED])
    if state == 'no_items':
        negative.append(item_clauses)
    if q.view.in_holdout is True:
        positive.append([_HOLDOUT])
    elif q.view.in_holdout is False:
        negative.append([_HOLDOUT])
    must_not: list[dict[str, Any]] = []
    truncated = False
    for clauses in positive:
        ids, more = await item_image_ids(opensearch, {'bool': {'filter': clauses}})
        filters.append({'terms': {'image_id': ids}})
        truncated = truncated or more
    for clauses in negative:
        ids, more = await item_image_ids(opensearch, {'bool': {'filter': clauses}})
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


def _sort_clause(sort_id: str) -> list[dict[str, Any]]:
    """The sort clauses of ``sort_id`` (refuses one the route does not know)."""
    known = {**_SORTS, **_STORED_SORTS}
    if sort_id not in known:
        msg = f'sort must be one of {sorted(known)}'
        raise ImageQueryError(msg)
    return known[sort_id][1]


def _sort_options(counts_ready: bool) -> list[dict[str, str]]:
    return [*SORT_OPTIONS, *(STORED_SORT_OPTIONS if counts_ready else [])]


def _not_ready() -> ImageCountsNotReadyError:
    return ImageCountsNotReadyError(
        'sorting by unlabelled count needs every image counted; start the image-counts backfill '
        '(POST .../image_counts/backfill). Images ingested since are counted within seconds.'
    )


async def image_page(
    opensearch: Any, q: ImageQuery, embedding: WriteFields, policy: str
) -> ImagePage:
    """One page of images with their counts and boxes."""
    return (await image_page_counted(opensearch, q, embedding, policy))[0]


async def image_page_counted(
    opensearch: Any, q: ImageQuery, embedding: WriteFields, policy: str
) -> tuple[ImagePage, bool]:
    """:func:`image_page` and whether every image in the list's scope carries its stored counts
    (the fact ``ImagePage.sort_options`` is built from), so a caller needs no second count."""
    sort = _sort_clause(q.view.sort_id)
    query, join_truncated = await _image_query(opensearch, q, embedding, policy)
    body = page_body(
        {
            'track_total_hits': True,
            'query': query,
            '_source': {'includes': list(IMAGE_SOURCE_FIELDS)},
            'aggs': {'uncounted': {'filter': _UNCOUNTED}},
        },
        sort=sort,
        tiebreak='image_id',
        cursor=q.cursor,
        sort_id=q.view.sort_id,
        limit=q.limit,
    )
    resp = await opensearch.search(index=images_index(), body=body)
    ready = int(resp['aggregations']['uncounted']['doc_count']) == 0
    if q.view.sort_id in _STORED_SORTS and not ready:
        raise _not_ready()
    kept, next_cursor = split_page(resp['hits']['hits'], sort_id=q.view.sort_id, limit=q.limit)
    ids = [str(h['_source']['image_id']) for h in kept]
    facts = await item_facts(opensearch, ids, include_test=q.view.include_test)
    prefix = project_api_base()
    cards = [_card(h['_source'], prefix, policy, facts[i]) for h, i in zip(kept, ids, strict=True)]
    empty = None
    if not cards:
        empty = 'no_match' if q.narrowed() else 'no_images'
    page = ImagePage(
        items=cards,
        next_cursor=next_cursor,
        total=int(resp['hits']['total']['value']),
        limit=q.limit,
        join_truncated=join_truncated,
        sort_applied=q.view.sort_id,
        sort_options=_sort_options(ready),
        hidden=HiddenCounts(
            **await hidden_counts(
                opensearch,
                item_filter_clauses(without_image_predicates(q.item_filter), embedding=embedding),
                include_test=q.view.include_test,
                include_excluded='excluded' in q.item_filter.review_status,
            )
        ),
        empty_reason=empty,
        empty_message=EMPTY_MESSAGES[empty] if empty else None,
    )
    return page, ready


async def locate_image(
    opensearch: Any,
    q: ImageQuery,
    embedding: WriteFields,
    policy: str,
    image_id: str,
    image_source: dict[str, Any] | None,
) -> ImageLocate:
    """Where ``image_id`` (whose stored document is ``image_source``, ``None`` when there is none)
    sits in the list ``q`` would serve; its cursor and limit are ignored."""
    sort = _sort_clause(q.view.sort_id)
    query, _truncated = await _image_query(opensearch, q, embedding, policy)
    if q.view.sort_id in _STORED_SORTS and not await counts_ready(opensearch, query):
        raise _not_ready()
    out = ImageLocate(
        image_id=image_id,
        in_view=False,
        index=None,
        total=int((await opensearch.count(index=images_index(), body={'query': query}))['count']),
        cursor_before=None,
        cursor_after=None,
        previous_image_id=None,
        reason='not_found',
        reason_message='No such image.',
    )
    if image_source is None:
        return out
    source = {**image_source, 'image_id': image_id}
    source[_TAKEN] = (source.get('capture') or {}).get('taken_local')
    source['counts.unlabeled'] = (source.get('counts') or {}).get('unlabeled')
    where = await locate_item(
        opensearch,
        index=images_index(),
        query=query,
        sort=with_tiebreak(sort, 'image_id'),
        sort_id=q.view.sort_id,
        source=source,
        id_field='image_id',
    )
    if where is None:
        return out.model_copy(
            update={
                'reason': 'filtered_out',
                'reason_message': 'The image is not in the current filtered view.',
            }
        )
    return out.model_copy(
        update={
            'in_view': True,
            'reason': None,
            'reason_message': None,
            'previous_image_id': where.previous_id,
            **where.as_wire(),
        }
    )
