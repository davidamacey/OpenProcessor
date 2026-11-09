"""The cluster fields of the ``items`` body: the served cluster of the serving partition, its
distance and the suggestion. An existing index keeps the retired integer fields until it is
recreated (OpenSearch cannot drop a mapped field); nothing reads or writes them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.clients.curation_opensearch.base import config


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = logging.getLogger(__name__)

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
    'cluster_distance': {'type': 'float'},
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
