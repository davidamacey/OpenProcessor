"""The query that ranks a cluster's representative crops.

Shared by the cluster cards (``GET /clusters``, ``GET /clusters/representatives``)
and the VLM ``representatives`` scope, so both mean the same crops by
"representative": nearest the cluster centre first.
"""

from __future__ import annotations

from typing import Any


REPS_SORT: list[dict[str, Any]] = [
    {'cluster_distance': {'order': 'asc', 'missing': '_last', 'unmapped_type': 'double'}},
    {'crop_id': 'asc'},
]
REPS_SOURCE = [
    'crop_id',
    'cluster_id',
    'cluster_distance',
    'cluster_distance_cluster_id',
    'class_name',
    'cluster_subid',
]


def representatives_msearch_body(cluster_id: int, per_cluster: int) -> dict[str, Any]:
    """One msearch query body: top ``per_cluster`` reps for one cluster.

    One ``_msearch`` entry per cluster in the caller's page replaces a
    ``top_hits`` sub-agg, which would decompress stored ``_source`` for every
    representative across *every* bucket.
    """
    return {
        'size': per_cluster,
        'query': {
            'bool': {
                'filter': [{'term': {'cluster_id': cluster_id}}],
                'must_not': [{'term': {'class_excluded': True}}],
            }
        },
        '_source': REPS_SOURCE,
        'sort': REPS_SORT,
    }
