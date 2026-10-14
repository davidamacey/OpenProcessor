"""What the bulk and ingest mixins need from the clustering service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    import numpy as np

    from src.config.cluster_index import ClusterIndex


class ClusterAssignmentLike(Protocol):
    cluster_id: int
    distance: float


class ClusterAssigner(Protocol):
    """Assigns an embedding to a cluster once the index is trained
    (``src.services.visual_search.clustering.ClusteringService``)."""

    def is_trained(self, index: ClusterIndex) -> bool: ...

    def assign_cluster(
        self, index: ClusterIndex, embedding: np.ndarray[Any, Any]
    ) -> ClusterAssignmentLike: ...
