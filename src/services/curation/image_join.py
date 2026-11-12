"""Predicates about an image, and the join that lets an item route ask them.

An item carries no coordinate and no near-duplicate group of its own image (the position is never
copied onto an item), so a position or group predicate is asked of the images index. A route that
lists items resolves the predicates to the images that match, at most :data:`JOIN_MAX_IMAGES` of
them, and then filters its items on ``image_id``; ``join_truncated`` says when it was more. The map
and the image gallery join the other way (an item predicate resolved to images) with
:func:`item_image_ids`.

* geo_bounding_box: https://docs.opensearch.org/latest/query-dsl/geo-and-xy/geo-bounding-box/
* geo_polygon: https://docs.opensearch.org/latest/query-dsl/geo-and-xy/geopolygon/
* terms aggregation: https://docs.opensearch.org/latest/aggregations/bucket/terms/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from src.config.curation_accessors import images_index, items_index
from src.domain.location_policy import snap_box, snap_polygon
from src.services.curation.location_policy_store import get_location_policy


if TYPE_CHECKING:
    from src.domain.geo_box import GeoBox
    from src.services.curation.item_filter import ItemFilter

LOCATION_FIELD: Final = 'capture.location'
JOIN_MAX_IMAGES: Final = 10_000

NearDupFilter = Literal['member', 'representative', 'none', 'collapsed']
IMAGE_PREDICATE_FIELDS: Final = ('geo_bbox', 'geo_polygon', 'has_location', 'near_dup_group')


@dataclass(frozen=True)
class ImageJoin:
    image_ids: tuple[str, ...]
    truncated: bool


class ImageJoinMissingError(RuntimeError):
    """A filter with image predicates reached the query builder without its join: a programming
    error that must surface, never an empty or unfiltered result."""


def geo_box_query(box: GeoBox) -> dict[str, Any]:
    return {
        'geo_bounding_box': {
            LOCATION_FIELD: {
                'top_left': {'lat': box.north, 'lon': box.west},
                'bottom_right': {'lat': box.south, 'lon': box.east},
            }
        }
    }


def location_clauses(
    has_location: bool | None, box: GeoBox | None, policy: str
) -> list[dict[str, Any]]:
    """Images-index clauses for "has a served position" and "inside ``box``" under ``policy``."""
    out: list[dict[str, Any]] = []
    located = {'exists': {'field': LOCATION_FIELD}}
    if has_location is not None:
        if policy == 'drop':
            out.append({'match_all': {}} if not has_location else {'match_none': {}})
        else:
            out.append(located if has_location else {'bool': {'must_not': located}})
    if box is not None:
        out.append({'match_none': {}} if policy == 'drop' else geo_box_query(box))
    return out


def near_dup_clauses(near_dup: NearDupFilter | None) -> list[dict[str, Any]]:
    group = {'exists': {'field': 'near_dup_group_id'}}
    representative = {'term': {'near_dup_is_representative': True}}
    if near_dup == 'member':
        return [{'bool': {'filter': [group], 'must_not': [representative]}}]
    if near_dup == 'representative':
        return [representative]
    if near_dup == 'none':
        return [{'bool': {'must_not': [group]}}]
    if near_dup == 'collapsed':
        # One image per group (its representative) plus every image in no group.
        return [
            {
                'bool': {
                    'should': [representative, {'bool': {'must_not': [group]}}],
                    'minimum_should_match': 1,
                }
            }
        ]
    return []


def has_image_predicates(f: ItemFilter) -> bool:
    return any(getattr(f, name) is not None for name in IMAGE_PREDICATE_FIELDS)


def without_image_predicates(f: ItemFilter) -> ItemFilter:
    """``f`` minus the image predicates, for a route that asks them of the images index itself."""
    return f.model_copy(update=dict.fromkeys(IMAGE_PREDICATE_FIELDS))


def image_predicate_clauses(f: ItemFilter, *, policy: str) -> list[dict[str, Any]]:
    """The image predicates of ``f`` as images-index clauses, area snapped to ``policy``."""
    out = location_clauses(f.has_location, None, policy)
    if f.geo_bbox is not None:
        box = snap_box(f.geo_bbox, policy)
        out.append({'match_none': {}} if box is None else geo_box_query(box))
    if f.geo_polygon is not None:
        ring = snap_polygon(f.geo_polygon, policy)
        out.append(
            {'match_none': {}}
            if ring is None
            else {
                'geo_polygon': {
                    LOCATION_FIELD: {'points': [{'lat': lat, 'lon': lon} for lon, lat in ring]}
                }
            }
        )
    if f.near_dup_group is not None:
        out.append({'term': {'near_dup_group_id': str(f.near_dup_group)}})
    return out


async def joined_image_ids(
    opensearch: Any, *, index: str, clauses: list[dict[str, Any]]
) -> tuple[list[str], bool]:
    """The ids of the images in ``index`` matching ``clauses`` (at most :data:`JOIN_MAX_IMAGES`)
    and whether there were more."""
    resp = await opensearch.search(
        index=index,
        body={
            'size': JOIN_MAX_IMAGES,
            # One past the cap is enough to know there were more, and from + size may not
            # exceed the index's result window (10,000 by default).
            'track_total_hits': JOIN_MAX_IMAGES + 1,
            'query': {'bool': {'filter': clauses}},
            '_source': False,
            'sort': [{'image_id': 'asc'}],
        },
    )
    ids = [str(h['_id']) for h in resp['hits']['hits']]
    return ids, resp['hits']['total']['value'] > JOIN_MAX_IMAGES


async def resolve_image_join(opensearch: Any, f: ItemFilter, *, policy: str) -> ImageJoin | None:
    """The images matching ``f``'s image predicates; ``None`` (no query) when it has none."""
    if not has_image_predicates(f):
        return None
    ids, truncated = await joined_image_ids(
        opensearch, index=images_index(), clauses=image_predicate_clauses(f, policy=policy)
    )
    return ImageJoin(image_ids=tuple(ids), truncated=truncated)


async def resolve_filter_join(opensearch: Any, f: ItemFilter) -> ImageJoin | None:
    """:func:`resolve_image_join` under the bound project's location policy; a route resolves
    this once, before it builds its query, and passes the result to ``item_filter_clauses``."""
    if not has_image_predicates(f):
        return None
    return await resolve_image_join(opensearch, f, policy=await get_location_policy())


def join_truncated(join: ImageJoin | None) -> bool:
    return join is not None and join.truncated


async def item_image_ids(opensearch: Any, query: dict[str, Any]) -> tuple[list[str], bool]:
    """The images that have an item matching ``query`` and whether there were more than we use."""
    resp = await opensearch.search(
        index=items_index(),
        body={
            'size': 0,
            'query': query,
            'aggs': {'images': {'terms': {'field': 'image_id', 'size': JOIN_MAX_IMAGES}}},
        },
    )
    agg = resp['aggregations']['images']
    return [b['key'] for b in agg['buckets']], int(agg.get('sum_other_doc_count', 0)) > 0


__all__ = [
    'IMAGE_PREDICATE_FIELDS',
    'JOIN_MAX_IMAGES',
    'LOCATION_FIELD',
    'ImageJoin',
    'ImageJoinMissingError',
    'NearDupFilter',
    'geo_box_query',
    'has_image_predicates',
    'image_predicate_clauses',
    'item_image_ids',
    'join_truncated',
    'joined_image_ids',
    'location_clauses',
    'near_dup_clauses',
    'resolve_filter_join',
    'resolve_image_join',
    'without_image_predicates',
]
