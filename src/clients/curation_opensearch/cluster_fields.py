"""The cluster fields of the ``items`` body: the item's cluster, its distance and sub-cluster,
the suggestion, and the geometry the clustering run stamps. Lane 4F owns this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.clients.curation_opensearch.base import config


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = logging.getLogger(__name__)

# Cluster geometry written by the clustering run
# (src/services/curation/clustering/cluster_geometry.py): the cluster id a
# stored cluster_distance was measured against, so a reader can tell a
# distance that went stale when the item moved to another cluster.
CLUSTER_GEOMETRY_MAPPING: dict[str, Any] = {
    'cluster_distance_cluster_id': {'type': 'integer'},
    # Cluster whose centroid is nearest the item (cluster purity, DQ-M2).
    'cluster_nearest_id': {'type': 'integer'},
}

# The served assignment of the active partition (research clustering plan 5.3). One projection
# writer (services/clustering/projection.py) owns these; a box element carries the same keys.
CLUSTER_PARTITION_MAPPING: dict[str, Any] = {
    'cluster_partition_id': {'type': 'keyword'},
    'cluster_uid': {'type': 'keyword'},
    'cluster_state': {'type': 'keyword'},
    'cluster_score': {'type': 'float'},
    'cluster_margin': {'type': 'float'},
    'cluster_assigned_by': {'type': 'keyword'},
}

CLUSTER_ITEM_MAPPING: dict[str, Any] = {
    **CLUSTER_PARTITION_MAPPING,
    'cluster_id': {'type': 'integer'},
    'cluster_distance': {'type': 'float'},
    'cluster_subid': {'type': 'keyword'},  # AHC sub-cluster id (e.g. "47a")
    **CLUSTER_GEOMETRY_MAPPING,
    'cluster_auto_suggest': {'type': 'keyword'},
}


async def ensure_items_cluster_partition_fields(client: AsyncOpenSearch) -> dict[str, Any]:
    """PUT the partition fields onto an existing items mapping (additive and idempotent)."""
    index = config.items_index
    try:
        resp = await client.indices.put_mapping(
            index=index, body={'properties': CLUSTER_PARTITION_MAPPING}
        )
    except OpenSearchException as exc:  # reported to the bootstrap, which logs and continues
        logger.error('curation_mapping_migration_failed index=%s error=%s', index, exc)
        return {'acknowledged': False, 'index': index, 'error': str(exc)}
    return {
        'acknowledged': bool(resp.get('acknowledged', False)),
        'index': index,
        'fields_added': list(CLUSTER_PARTITION_MAPPING),
    }
