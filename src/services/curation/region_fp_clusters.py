"""Region cluster cards: the clusters of the serving region-box partition, each with its
representative boxes (``GET /regions/clusters``).

Clusters hold *boxes*: a card's ``size`` is the items with at least one box in the cluster,
``box_count`` the boxes in it. One nested aggregation names the clusters and each cluster's best
boxes; the representatives are then read as full wire rows.
"""

from __future__ import annotations

from typing import Any

from src.config import get_region_fields
from src.config.curation_accessors import items_index
from src.config.project_context import project_api_base
from src.services.curation.clustering.pool_snapshot import GOOD_BOX_STATES
from src.services.curation.clustering.region_box_rows import box_state_clause
from src.services.curation.region_rows import rows_for_pairs
from src.services.curation.wire import box_thumbnail_url


async def region_cluster_cards(
    opensearch: Any,
    *,
    max_clusters: int,
    per_cluster: int,
    max_rank: int | None,
    source_excludes: list[str],
) -> list[dict[str, Any]]:
    """The cluster cards, largest first. An OpenSearch failure propagates."""
    F = get_region_fields()
    uid = f'{F.boxes}.cluster_uid'
    scope = {
        'bool': {
            'filter': [
                box_state_clause(list(GOOD_BOX_STATES), F),
                {'exists': {'field': uid}},
            ]
        }
    }
    filters: list[dict[str, Any]] = [{'nested': {'path': F.boxes, 'query': scope}}]
    if max_rank is not None:
        filters.append({'range': {'crop_rank_in_image': {'lte': max_rank}}})
    body = {
        'size': 0,
        'query': {'bool': {'filter': filters, 'must_not': [{'term': {'test_holdout': True}}]}},
        'aggs': {
            'boxes': {
                'nested': {'path': F.boxes},
                'aggs': {
                    'in_scope': {
                        'filter': scope,
                        'aggs': {
                            'clusters': {
                                'terms': {'field': uid, 'size': max_clusters},
                                'aggs': {
                                    'reps': {
                                        'top_hits': {
                                            'size': per_cluster,
                                            # Only the parent _id and the box id are read.
                                            '_source': False,
                                            'docvalue_fields': [f'{F.boxes}.box_id'],
                                            'sort': [
                                                {
                                                    f'{F.boxes}.cluster_score': {
                                                        'order': 'desc',
                                                        'missing': '_last',
                                                        'unmapped_type': 'float',
                                                    }
                                                }
                                            ],
                                        }
                                    },
                                    'items': {
                                        'reverse_nested': {},
                                        'aggs': {
                                            'validated': {'filter': {'term': {F.validated: True}}}
                                        },
                                    },
                                },
                            }
                        },
                    }
                },
            }
        },
    }
    resp = await opensearch.search(index=items_index(), body=body)

    scoped = ((resp.get('aggregations') or {}).get('boxes') or {}).get('in_scope') or {}
    buckets = (scoped.get('clusters') or {}).get('buckets') or []
    clusters: list[dict[str, Any]] = []
    rep_pairs: list[tuple[str, str | None, dict[str, Any]]] = []
    for b in buckets:
        rep_hits = (((b.get('reps') or {}).get('hits') or {}).get('hits')) or []
        reps = [
            (h.get('_id') or '', ((h.get('fields') or {}).get(f'{F.boxes}.box_id') or [None])[0])
            for h in rep_hits
        ]
        rep_pairs.extend((crop_id, box_id, {}) for crop_id, box_id in reps)
        items = b.get('items') or {}
        clusters.append(
            {
                'cluster_uid': str(b['key']),
                'size': int(items.get('doc_count', 0)),
                'box_count': int(b['doc_count']),
                'validated_count': int((items.get('validated') or {}).get('doc_count', 0)),
                'representative_crop_ids': [crop_id for crop_id, _box_id in reps],
                'representative_box_ids': [box_id for _crop_id, box_id in reps],
                'representative_thumb_urls': [
                    box_thumbnail_url(project_api_base(), crop_id, box_id or '')
                    for crop_id, box_id in reps
                ],
            }
        )
    rows = await rows_for_pairs(
        opensearch,
        index=items_index(),
        pairs=rep_pairs,
        source_excludes=source_excludes,
    )
    by_key = {(r['crop_id'], r['region_box_id']): r for r in rows}
    for c in clusters:
        c['representatives'] = [
            by_key[key]
            for key in zip(c['representative_crop_ids'], c['representative_box_ids'], strict=True)
            if key in by_key
        ]
    clusters.sort(key=lambda c: -c['size'])
    return clusters
