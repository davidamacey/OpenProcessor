"""The query that ranks a cluster's representative crops.

Shared by the cluster cards (``GET /clusters``, ``GET /clusters/representatives``,
``partition_representatives_body``) and, until 4F-2d moves it, the VLM ``representatives`` scope (integer ids), so both mean the same crops by
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


PARTITION_REPS_SORT: list[dict[str, Any]] = [
    {'cluster_score': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'float'}},
    {'crop_id': 'asc'},
]
PARTITION_REPS_SOURCE = ['crop_id', 'cluster_score', 'class_name']


def partition_representatives_body(cluster_uid: str, per_cluster: int) -> dict[str, Any]:
    """One msearch query body: the ``per_cluster`` members of one served cluster scoring
    highest (``cluster_score`` is the similarity to the cluster's prototypes)."""
    return {
        'size': per_cluster,
        'query': {
            'bool': {
                'filter': [{'term': {'cluster_uid': cluster_uid}}],
                'must_not': [{'term': {'class_excluded': True}}],
            }
        },
        '_source': PARTITION_REPS_SOURCE,
        'sort': PARTITION_REPS_SORT,
    }
