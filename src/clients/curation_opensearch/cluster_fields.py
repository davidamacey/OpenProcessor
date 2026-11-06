"""The cluster fields of the ``items`` body: the item's cluster, its distance and sub-cluster,
the suggestion, and the geometry the clustering run stamps. Lane 4F owns this module.
"""

from __future__ import annotations

from typing import Any


# Cluster geometry written by the clustering run
# (src/services/curation/clustering/cluster_geometry.py): the cluster id a
# stored cluster_distance was measured against, so a reader can tell a
# distance that went stale when the item moved to another cluster.
CLUSTER_GEOMETRY_MAPPING: dict[str, Any] = {
    'cluster_distance_cluster_id': {'type': 'integer'},
    # Cluster whose centroid is nearest the item (cluster purity, DQ-M2).
    'cluster_nearest_id': {'type': 'integer'},
}

CLUSTER_ITEM_MAPPING: dict[str, Any] = {
    'cluster_id': {'type': 'integer'},
    'cluster_distance': {'type': 'float'},
    'cluster_subid': {'type': 'keyword'},  # AHC sub-cluster id (e.g. "47a")
    **CLUSTER_GEOMETRY_MAPPING,
    'cluster_auto_suggest': {'type': 'keyword'},
}
