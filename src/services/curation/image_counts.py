"""Per-image item counts and card boxes for a set of images: one aggregation on the items index.

A terms bucket per image holds the counts as filter sub-aggregations and the card's boxes as a
``top_hits``, so a page of any size costs one query:

* terms: https://docs.opensearch.org/latest/aggregations/bucket/terms/
* filter: https://docs.opensearch.org/latest/aggregations/bucket/filter/
* top_hits: https://docs.opensearch.org/latest/aggregations/metric/top-hits/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import get_region_fields
from src.config.curation_accessors import items_index
from src.schemas.curation.images import BoxView, ImageCounts
from src.services.curation.image_boxes import boxes_of


#: Items drawn on one card; the card says when there were more.
CARD_ITEMS_MAX = 50

_EXCLUDED = {'term': {'class_excluded': True}}
_HOLDOUT = {'term': {'test_holdout': True}}


@dataclass(frozen=True)
class ImageItemFacts:
    counts: ImageCounts
    excluded: int
    boxes: list[BoxView]
    boxes_truncated: bool


def _bucket_aggs(include_test: bool, box_limit: int) -> dict[str, Any]:
    shown_aggs: dict[str, Any] = {
        'labeled': {'filter': {'exists': {'field': 'class_id'}}},
        'validated': {'filter': {'term': {'class_validated': True}}},
    }
    if box_limit:
        shown_aggs['boxes'] = {
            'top_hits': {
                'size': box_limit,
                'sort': [{'crop_rank_in_image': {'order': 'asc', 'missing': '_last'}}],
                '_source': {
                    'includes': [
                        'crop_id',
                        'bbox_norm',
                        'class_id',
                        'class_name',
                        'class_validated',
                        'test_holdout',
                        get_region_fields().boxes,
                    ]
                },
            }
        }
    shown = {'match_all': {}} if include_test else {'bool': {'must_not': _HOLDOUT}}
    return {
        'excluded': {'filter': _EXCLUDED},
        'live': {
            'filter': {'bool': {'must_not': _EXCLUDED}},
            'aggs': {
                'holdout': {'filter': _HOLDOUT},
                'shown': {'filter': shown, 'aggs': shown_aggs},
            },
        },
    }


def _facts(bucket: dict[str, Any]) -> ImageItemFacts:
    live = bucket['live']
    shown = live['shown']
    items = int(shown['doc_count'])
    labeled = int(shown['labeled']['doc_count'])
    hits = ((shown.get('boxes') or {}).get('hits') or {}).get('hits') or []
    pairs = [(str(h.get('_id') or ''), h.get('_source') or {}) for h in hits]
    return ImageItemFacts(
        counts=ImageCounts(
            items=items,
            labeled=labeled,
            validated=int(shown['validated']['doc_count']),
            unlabeled=items - labeled,
            in_holdout=int(live['holdout']['doc_count']),
        ),
        excluded=int(bucket['excluded']['doc_count']),
        boxes=boxes_of(pairs),
        boxes_truncated=items > len(pairs),
    )


_EMPTY = ImageItemFacts(
    counts=ImageCounts(items=0, labeled=0, validated=0, unlabeled=0, in_holdout=0),
    excluded=0,
    boxes=[],
    boxes_truncated=False,
)


async def item_facts(
    opensearch: Any, image_ids: list[str], *, include_test: bool, box_limit: int = CARD_ITEMS_MAX
) -> dict[str, ImageItemFacts]:
    """The item counts (and boxes, when ``box_limit`` > 0) of each image; an image with no items
    gets zeros."""
    if not image_ids:
        return {}
    resp = await opensearch.search(
        index=items_index(),
        body={
            'size': 0,
            'query': {'terms': {'image_id': image_ids}},
            'aggs': {
                'images': {
                    'terms': {'field': 'image_id', 'size': len(image_ids)},
                    'aggs': _bucket_aggs(include_test, box_limit),
                }
            },
        },
    )
    buckets = resp['aggregations']['images']['buckets']
    found = {str(b['key']): _facts(b) for b in buckets}
    return {image_id: found.get(image_id, _EMPTY) for image_id in image_ids}
