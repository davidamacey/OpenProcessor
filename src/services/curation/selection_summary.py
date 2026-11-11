"""The selection summary: one aggregation over the items a selection names.

* filters aggregation: https://docs.opensearch.org/latest/aggregations/bucket/filters/
* terms aggregation: https://docs.opensearch.org/latest/aggregations/bucket/terms/
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from src.config.curation_accessors import images_index, items_index
from src.schemas.curation.selection import ClassCount, SelectionSummary
from src.services.curation.item_selection import (
    MAX_BULK_SELECTION,
    ItemSelection,
    SelectionError,
    selection_query,
)
from src.services.curation.map_cells import JOIN_MAX_IMAGES, LOCATION_FIELD
from src.services.curation.reprocess_locks import class_locked_clause, region_locked_clause


if TYPE_CHECKING:
    from src.services.embedding_spaces.readers import WriteFields

TOP_CLASSES: Final = 10
UNLABELED: Final = 'unlabeled'
OTHER: Final = 'other'
_TAKEN = 'capture_taken_local'


def summary_text(detections: int, images: int, locked: int) -> str:
    noun = 'detection' if detections == 1 else 'detections'
    unit = 'image' if images == 1 else 'images'
    text = f'{detections} {noun} in {images} {unit}'
    return f'{text} · {locked} locked, skipped' if locked else text


def _query(sel: ItemSelection, embedding: WriteFields) -> dict[str, Any]:
    if sel.crop_ids is not None:
        return {'terms': {'crop_id': list(dict.fromkeys(sel.crop_ids))}}
    try:
        return selection_query(sel, embedding)
    except ValueError as exc:
        raise SelectionError(str(exc)) from exc


def _iso(agg: dict[str, Any]) -> str | None:
    value = agg.get('value_as_string')
    return str(value) if value else None


async def _located(opensearch: Any, image_ids: list[str], policy: str) -> int:
    if not image_ids or policy == 'drop':
        return 0
    resp = await opensearch.search(
        index=images_index(),
        body={
            'size': 0,
            'track_total_hits': True,
            'query': {
                'bool': {
                    'filter': [
                        {'terms': {'image_id': image_ids}},
                        {'exists': {'field': LOCATION_FIELD}},
                    ]
                }
            },
        },
    )
    return int(resp['hits']['total']['value'])


async def selection_summary(
    opensearch: Any, sel: ItemSelection, embedding: WriteFields, policy: str
) -> SelectionSummary:
    """Counts for the selection in one query on the items index, plus the images' positions."""
    resp = await opensearch.search(
        index=items_index(),
        body={
            'size': 0,
            'track_total_hits': True,
            'query': _query(sel, embedding),
            'aggs': {
                'image_count': {'cardinality': {'field': 'image_id'}},
                'images': {'terms': {'field': 'image_id', 'size': JOIN_MAX_IMAGES}},
                'clusters': {'cardinality': {'field': 'cluster_uid'}},
                'classes': {'terms': {'field': 'class_name', 'size': TOP_CLASSES}},
                'taken_from': {'min': {'field': _TAKEN}},
                'taken_to': {'max': {'field': _TAKEN}},
                'flags': {
                    'filters': {
                        'filters': {
                            'locked': {
                                'bool': {
                                    'should': [class_locked_clause(), region_locked_clause()],
                                    'minimum_should_match': 1,
                                }
                            },
                            'holdout': {'term': {'test_holdout': True}},
                            'unlabeled': {'bool': {'must_not': {'exists': {'field': 'class_id'}}}},
                        }
                    }
                },
            },
        },
    )
    detections = int(resp['hits']['total']['value'])
    if detections > MAX_BULK_SELECTION:
        msg = f'the selection names {detections} items; the limit is {MAX_BULK_SELECTION}'
        raise SelectionError(msg)
    aggs = resp['aggregations']
    flags = aggs['flags']['buckets']
    image_buckets = aggs['images']
    image_ids = [b['key'] for b in image_buckets['buckets']]
    counts = [
        ClassCount(class_name=b['key'], count=b['doc_count']) for b in aggs['classes']['buckets']
    ]
    unlabeled = flags['unlabeled']['doc_count']
    named = sum(c.count for c in counts)
    other = detections - unlabeled - named
    if unlabeled:
        counts.append(ClassCount(class_name=UNLABELED, count=unlabeled))
    if other > 0:
        counts.append(ClassCount(class_name=OTHER, count=other))
    locked = flags['locked']['doc_count']
    images = int(aggs['image_count']['value'])
    return SelectionSummary(
        detections=detections,
        images=images,
        clusters=int(aggs['clusters']['value']),
        locked=locked,
        in_holdout=flags['holdout']['doc_count'],
        by_class=counts,
        taken_from=_iso(aggs['taken_from']),
        taken_to=_iso(aggs['taken_to']),
        located=await _located(opensearch, image_ids, policy),
        join_truncated=image_buckets.get('sum_other_doc_count', 0) > 0,
        summary_text=summary_text(detections, images, locked),
    )
