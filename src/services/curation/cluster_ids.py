"""Cluster-id namespaces and cluster-membership facts, as served data.

Leaf module (no heavy imports) so the item serializer, the cluster-card
router and the clustering orchestrator share one definition.

* ``cluster_id < 0`` — unassigned (noise, parked, excluded).
* ``0 .. RESIDUAL_CLUSTER_ID_OFFSET-1`` — class clusters
  (``cluster_id == class_id``).
* ``>= RESIDUAL_CLUSTER_ID_OFFSET`` — candidate clusters from the residual
  pass; they need a human or VLM class assignment.
"""

from __future__ import annotations

from typing import Any, Literal


RESIDUAL_CLUSTER_ID_OFFSET = 10000

ClusterKind = Literal['class', 'candidate', 'unassigned']

CORE_SIMILARITY_MIN = 0.75
"""A member whose ``cluster_similarity`` is at least this is a core member
of its cluster (the cluster view's cut line)."""


def cluster_kind(cluster_id: Any) -> ClusterKind | None:
    """Kind of cluster ``cluster_id`` names; ``None`` when there is none."""
    if not isinstance(cluster_id, int) or isinstance(cluster_id, bool):
        return None
    if cluster_id < 0:
        return 'unassigned'
    if cluster_id >= RESIDUAL_CLUSTER_ID_OFFSET:
        return 'candidate'
    return 'class'


def cluster_similarity(cluster_distance: Any) -> float | None:
    """``1 - cluster_distance`` (cosine distance to the centroid), clamped
    to ``[0, 1]``; ``None`` without a numeric distance."""
    if not isinstance(cluster_distance, int | float) or isinstance(cluster_distance, bool):
        return None
    return min(max(1.0 - float(cluster_distance), 0.0), 1.0)


__all__ = [
    'CORE_SIMILARITY_MIN',
    'RESIDUAL_CLUSTER_ID_OFFSET',
    'ClusterKind',
    'cluster_kind',
    'cluster_similarity',
]
