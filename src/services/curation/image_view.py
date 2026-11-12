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

_HOLDOUT = {'term': {'test_holdout': True}}
_UNLABELED = {'bool': {'must_not': {'exists': {'field': 'class_id'}}}}


class ImageQueryError(ValueError):
    """The request names a sort or combination the route does not serve."""


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
    positive: list[list[dict[str, Any]]] = []
    negative: list[list[dict[str, Any]]] = []
    if q.view.label_state == 'has_unlabeled':
        positive.append([*item_clauses, _UNLABELED])
    elif not f.is_empty() or q.view.label_state == 'all_labeled':
        positive.append(item_clauses)
    if q.view.label_state == 'all_labeled':
        negative.append([*item_clauses, _UNLABELED])
    if q.view.label_state == 'no_items':
        negative.append(item_clauses)
    if q.view.in_holdout is True:
        positive.append([_HOLDOUT])
    elif q.view.in_holdout is False:
        negative.append([_HOLDOUT])
    filters = _image_level_clauses(q, policy)
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


async def image_page(
    opensearch: Any, q: ImageQuery, embedding: WriteFields, policy: str
) -> ImagePage:
    """One page of images with their counts and boxes."""
    if q.view.sort_id not in _SORTS:
        msg = f'sort must be one of {sorted(_SORTS)}'
        raise ImageQueryError(msg)
    query, join_truncated = await _image_query(opensearch, q, embedding, policy)
    body = page_body(
        {
            'track_total_hits': True,
            'query': query,
            '_source': {'includes': list(IMAGE_SOURCE_FIELDS)},
        },
        sort=_SORTS[q.view.sort_id][1],
        tiebreak='image_id',
        cursor=q.cursor,
        sort_id=q.view.sort_id,
        limit=q.limit,
    )
    resp = await opensearch.search(index=images_index(), body=body)
    kept, next_cursor = split_page(resp['hits']['hits'], sort_id=q.view.sort_id, limit=q.limit)
    ids = [str(h['_source']['image_id']) for h in kept]
    facts = await item_facts(opensearch, ids, include_test=q.view.include_test)
    prefix = project_api_base()
    cards = [_card(h['_source'], prefix, policy, facts[i]) for h, i in zip(kept, ids, strict=True)]
    empty = None
    if not cards:
        empty = 'no_match' if q.narrowed() else 'no_images'
    return ImagePage(
        items=cards,
        next_cursor=next_cursor,
        total=int(resp['hits']['total']['value']),
        limit=q.limit,
        join_truncated=join_truncated,
        sort_applied=q.view.sort_id,
        sort_options=SORT_OPTIONS,
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
    if q.view.sort_id not in _SORTS:
        msg = f'sort must be one of {sorted(_SORTS)}'
        raise ImageQueryError(msg)
    query, _truncated = await _image_query(opensearch, q, embedding, policy)
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
    where = await locate_item(
        opensearch,
        index=images_index(),
        query=query,
        sort=with_tiebreak(_SORTS[q.view.sort_id][1], 'image_id'),
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
