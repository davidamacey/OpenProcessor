"""Cluster configuration and result records for the FAISS clustering service."""

from dataclasses import dataclass

from src.config.cluster_index import ClusterIndex


@dataclass
class ClusterConfig:
    """Configuration for a category's clustering."""

    n_clusters: int
    nprobe: int  # Number of clusters to search during queries
    rebalance_threshold: float  # Ratio of new items that triggers rebalance


# Default configurations based on expected data distribution
# Rule of thumb: n_clusters ≈ sqrt(n_vectors) to n_vectors/10 for fine separation
# More clusters = better separation but slightly more memory
DEFAULT_CONFIGS: dict[ClusterIndex, ClusterConfig] = {
    ClusterIndex.GLOBAL: ClusterConfig(n_clusters=1024, nprobe=32, rebalance_threshold=0.5),
    ClusterIndex.VEHICLES: ClusterConfig(
        n_clusters=512, nprobe=24, rebalance_threshold=0.5
    ),  # Higher for brand/type separation
    ClusterIndex.PEOPLE: ClusterConfig(n_clusters=512, nprobe=24, rebalance_threshold=0.5),
    ClusterIndex.FACES: ClusterConfig(
        n_clusters=2048, nprobe=64, rebalance_threshold=0.5
    ),  # Highest for face identity
}


@dataclass
class ClusterStats:
    """Statistics about a FAISS cluster index."""

    index_name: str
    n_clusters: int
    n_vectors: int
    avg_cluster_size: float
    min_cluster_size: int
    max_cluster_size: int
    empty_clusters: int
    trained_at: str | None
    is_trained: bool


@dataclass
class ClusterBalance:
    """Cluster balance assessment."""

    index_name: str
    is_balanced: bool
    imbalance_ratio: float  # max_size / min_size (excluding empty)
    empty_ratio: float  # empty_clusters / n_clusters
    vectors_since_training: int
    needs_rebalance: bool
    reason: str | None


@dataclass
class ClusterAssignment:
    """Result of cluster assignment."""

    cluster_id: int
    distance: float  # Distance to centroid (lower = closer to cluster center)
