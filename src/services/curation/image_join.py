"""Predicates about an image, and the join that lets an item route ask them.

An item carries no coordinate and no near-duplicate group of its own image (the position is never
copied onto an item), so a position or group predicate is asked of the images index. A route that
lists items resolves the predicates to the images that match, at most :data:`JOIN_MAX_IMAGES` of
them, and then filters its items on ``image_id``; ``join_truncated`` says when it was more. The map
and the image gallery join the other way (an item predicate resolved to images) with
:func:`item_image_ids`.

* geo_bounding_box: https://docs.opensearch.org/latest/query-dsl/geo-and-xy/geo-bounding-box/
* terms aggregation: https://docs.opensearch.org/latest/aggregations/bucket/terms/
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

from src.config.curation_accessors import items_index


if TYPE_CHECKING:
    from src.schemas.curation.map import MapBounds

LOCATION_FIELD: Final = 'capture.location'
JOIN_MAX_IMAGES: Final = 10_000

NearDupFilter = Literal['member', 'representative', 'none']


def geo_box_query(box: MapBounds) -> dict[str, Any]:
    return {
        'geo_bounding_box': {
            LOCATION_FIELD: {
                'top_left': {'lat': box.north, 'lon': box.west},
                'bottom_right': {'lat': box.south, 'lon': box.east},
            }
        }
    }


def location_clauses(
    has_location: bool | None, box: MapBounds | None, policy: str
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
    return []


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
    'JOIN_MAX_IMAGES',
    'LOCATION_FIELD',
    'NearDupFilter',
    'geo_box_query',
    'item_image_ids',
    'location_clauses',
    'near_dup_clauses',
]
