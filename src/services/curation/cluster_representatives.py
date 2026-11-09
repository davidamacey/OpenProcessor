"""The query that ranks a cluster's representative crops.

Shared by the cluster cards (``GET /clusters``, ``GET /clusters/representatives``) and the VLM
``representatives`` scope, so both mean the same crops by "representative": best-scored first.
"""

from __future__ import annotations

from typing import Any


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
