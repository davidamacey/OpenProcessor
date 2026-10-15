"""Query bodies and response shaping for the raw-label review endpoints
(``GET /review/unmatched_terms``, ``GET /review/raw_label_clusters``)."""

from __future__ import annotations

from typing import Any

from src.services.curation.raw_label_clusters import (
    CLUSTER_ID_FIELD,
    CLUSTER_NAME_FIELD,
    RAW_LABEL_FIELD,
    UNMATCHED_CLASS_SOURCE,
)


def unmatched_terms_body(size: int) -> dict[str, Any]:
    return {
        'size': 0,
        'query': {'term': {'class_source': 'vlm_unmatched'}},
        'aggs': {
            'top_raw': {
                'terms': {
                    'field': 'vlm_raw_label',
                    'size': size,
                    # Push rare/unknowns to the bottom and avoid empty buckets.
                    'min_doc_count': 1,
                }
            },
        },
        'track_total_hits': True,
    }


def unmatched_terms_response(resp: dict[str, Any]) -> dict[str, Any]:
    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    buckets = (((resp.get('aggregations') or {}).get('top_raw') or {}).get('buckets')) or []
    top_terms = [
        {'label': str(b.get('key', '')), 'count': int(b.get('doc_count', 0))} for b in buckets
    ]
    return {'total_unmatched': int(total), 'top_terms': top_terms}


def raw_label_clusters_body(size: int, samples_per_cluster: int) -> dict[str, Any]:
    return {
        'size': 0,
        'query': {'exists': {'field': CLUSTER_ID_FIELD}},
        'aggs': {
            'by_cluster': {
                'terms': {
                    'field': CLUSTER_ID_FIELD,
                    'size': size,
                    'order': {'_count': 'desc'},
                },
                'aggs': {
                    'name': {'terms': {'field': CLUSTER_NAME_FIELD, 'size': 1}},
                    'samples': {
                        'terms': {
                            'field': RAW_LABEL_FIELD,
                            'size': samples_per_cluster,
                        }
                    },
                    'unmatched': {'filter': {'term': {'class_source': UNMATCHED_CLASS_SOURCE}}},
                    # Most common already-resolved class within the cluster — used
                    # as the ``parent_class_suggestion`` hint. If the cluster is
                    # 100% vlm_unmatched the bucket is empty and we return None.
                    'parent': {
                        # ``class_name`` is mapped ``keyword`` directly on the
                        # live index — no ``.keyword`` subfield exists (the
                        # migration that assumed a ``text`` mapping has been
                        # failing silently on every startup; see
                        # the reference clusters router's top_class agg for the full story).
                        'terms': {'field': 'class_name', 'size': 1},
                    },
                },
            },
        },
    }


def raw_label_clusters_response(resp: dict[str, Any]) -> dict[str, Any]:
    buckets = (((resp.get('aggregations') or {}).get('by_cluster') or {}).get('buckets')) or []
    clusters: list[dict[str, Any]] = []
    for b in buckets:
        cid = int(b.get('key', 0))
        n_crops = int(b.get('doc_count', 0))
        name_buckets = (b.get('name') or {}).get('buckets') or []
        cluster_name = str(name_buckets[0]['key']) if name_buckets else f'cluster_{cid}'
        sample_buckets = (b.get('samples') or {}).get('buckets') or []
        samples = [
            {'label': str(sb.get('key', '')), 'count': int(sb.get('doc_count', 0))}
            for sb in sample_buckets
        ]
        unmatched = int((b.get('unmatched') or {}).get('doc_count', 0))
        parent_buckets = (b.get('parent') or {}).get('buckets') or []
        parent = str(parent_buckets[0]['key']) if parent_buckets else None
        clusters.append(
            {
                'cluster_id': cid,
                'cluster_name': cluster_name,
                'n_crops': n_crops,
                'n_unmatched': unmatched,
                'sample_terms': samples,
                'parent_class_suggestion': parent,
            }
        )
    status_str = 'ok' if clusters else 'empty'
    return {
        'status': status_str,
        'clusters': clusters,
        'hint': (
            'Run scripts/curation/cluster_raw_labels.py to populate '
            f'{CLUSTER_ID_FIELD} on items if status=empty.'
        )
        if not clusters
        else None,
    }
