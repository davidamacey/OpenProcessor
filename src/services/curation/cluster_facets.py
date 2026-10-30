"""Facet counts of one cluster's members, for the filter chips beside the crop grid.

One aggregation. Each count is built from the same predicate the matching filter of
``GET /crops`` applies (``detector_class_name``, ``class_source``, ``label_validated``,
``review_tab``, ``detector_conf_min`` / ``detector_conf_max``), so a chip's number is the
size of the page it opens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.curation_accessors import items_index
from src.core.exceptions import RequestRejectedError
from src.services.curation.cluster_cards import CLUSTER_REVIEW_TABS, review_aggs


if TYPE_CHECKING:
    from src.schemas.curation.clusters import ClusterFacets, ConfidenceBin, FacetCount


MAX_DETECTOR_CLASSES = 20
"""Detector classes listed (the most common)."""

MAX_CLASS_SOURCES = 50
"""Class sources listed; far more than exist, so this never cuts."""

CONFIDENCE_BINS = 10
"""Fixed bins of 0.1 over the detector confidence."""


def _bin_edges() -> list[tuple[float, float | None]]:
    edges = [round(i / CONFIDENCE_BINS, 1) for i in range(CONFIDENCE_BINS)]
    return [(lo, hi) for lo, hi in zip(edges, [*edges[1:], None], strict=True)]


def facets_body(cluster_id: int) -> dict[str, Any]:
    ranges = [
        {k: v for k, v in (('from', lo), ('to', hi)) if v is not None} for lo, hi in _bin_edges()
    ]
    return {
        'size': 0,
        'track_total_hits': True,
        'query': {
            'bool': {
                'filter': [{'term': {'cluster_id': cluster_id}}],
                # A reviewer's members: the frozen holdout and human-ignored items are not shown.
                'must_not': [{'term': {'class_excluded': True}}, {'term': {'test_holdout': True}}],
            }
        },
        'aggs': {
            'detector_classes': {
                'terms': {'field': 'detector_class_name', 'size': MAX_DETECTOR_CLASSES}
            },
            'class_sources': {'terms': {'field': 'class_source', 'size': MAX_CLASS_SOURCES}},
            'validated': {'filter': {'term': {'class_validated': True}}},
            'unvalidated': {'filter': {'term': {'class_validated': False}}},
            'detector_confidence': {'range': {'field': 'detector_confidence', 'ranges': ranges}},
            'detector_confidence_missing': {
                'filter': {'bool': {'must_not': {'exists': {'field': 'detector_confidence'}}}}
            },
            **review_aggs(),
        },
    }


def _counts(agg: dict[str, Any]) -> list[FacetCount]:
    return [{'value': str(b['key']), 'count': int(b['doc_count'])} for b in agg.get('buckets', [])]


def facets_from_response(cluster_id: int, resp: dict[str, Any]) -> ClusterFacets:
    aggs = resp.get('aggregations', {})
    review = aggs.get('review', {}).get('buckets', {})
    found = aggs.get('detector_confidence', {}).get('buckets', [])
    bins: list[ConfidenceBin] = [
        {
            'min': lo,
            'max': hi,
            'count': int(found[i]['doc_count']) if i < len(found) else 0,
        }
        for i, (lo, hi) in enumerate(_bin_edges())
    ]
    return {
        'cluster_id': cluster_id,
        'total': int(resp.get('hits', {}).get('total', {}).get('value', 0)),
        'detector_classes': _counts(aggs.get('detector_classes', {})),
        'class_sources': _counts(aggs.get('class_sources', {})),
        'validation': {
            'validated': int(aggs.get('validated', {}).get('doc_count') or 0),
            'unvalidated': int(aggs.get('unvalidated', {}).get('doc_count') or 0),
        },
        'review_tabs': {
            tab: int(review.get(tab, {}).get('doc_count') or 0) for tab in CLUSTER_REVIEW_TABS
        },
        'detector_confidence': bins,
        'detector_confidence_missing': int(
            aggs.get('detector_confidence_missing', {}).get('doc_count') or 0
        ),
    }


async def cluster_facets(opensearch: Any, cluster_id: int) -> ClusterFacets:
    try:
        resp = await opensearch.search(index=items_index(), body=facets_body(cluster_id))
    except Exception as exc:
        raise RequestRejectedError(500, f'cluster facets query failed: {exc}') from exc
    return facets_from_response(cluster_id, resp)
