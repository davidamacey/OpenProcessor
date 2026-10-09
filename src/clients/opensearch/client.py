"""
OpenSearch Client for Visual Search

Multi-Index Architecture with FAISS IVF Clustering:

Indexes:
1. visual_search_global - Whole image similarity (scene matching)
2. visual_search_vehicles - Vehicle detections (car, truck, motorcycle, bus, boat)
3. visual_search_people - Person appearance (clothing, pose)
4. visual_search_faces - Face identity matching (future: ArcFace embeddings)

Key Features:
- Category-specific indexes for faster, more accurate search
- Automatic class-to-category routing during ingestion
- FAISS IVF clustering for industry-standard similarity grouping
- Cluster-filtered search for optimized queries
- Independent HNSW tuning per category

Usage:
    # Initialize client with clustering
    from src.services.clustering import get_clustering_service

    client = OpenSearchClient(hosts=["http://localhost:9200"])
    clustering = get_clustering_service()

    # Create all indexes
    await client.create_all_indexes()

    # Ingest image with auto-routing and cluster assignment
    await client.ingest_image(
        image_id="img_001",
        image_path="/path/to/image.jpg",
        global_embedding=np.array([...]),  # 512-dim
        box_embeddings=np.array([[...]]),  # [N, 512]
        normalized_boxes=np.array([[...]]),  # [N, 4]
        det_classes=[0, 2, 7],  # person, car, truck
        det_scores=[0.95, 0.87, 0.76],
        clustering_service=clustering,  # Optional: enables cluster assignment
    )
    # Routes: person -> visual_search_people, car/truck -> visual_search_vehicles
    # Each document gets cluster_id and cluster_distance fields

    # Search with cluster optimization
    results = await client.search_vehicles(
        query_embedding,
        top_k=10,
        cluster_ids=[5, 12, 23],  # Optional: narrow search to specific clusters
    )
"""

from __future__ import annotations

import logging

from opensearchpy import AsyncOpenSearch

from src.clients.opensearch.bulk import BulkMixin
from src.clients.opensearch.clusters import ClustersMixin
from src.clients.opensearch.indexes import IndexesMixin
from src.clients.opensearch.ingest import IngestMixin
from src.clients.opensearch.ocr import OcrMixin
from src.clients.opensearch.search import SearchMixin


logger = logging.getLogger(__name__)


class OpenSearchClient(IndexesMixin, IngestMixin, BulkMixin, SearchMixin, ClustersMixin, OcrMixin):
    """
    Async OpenSearch client for multi-index visual search with MobileCLIP embeddings.

    Supports four category-specific indexes:
    - visual_search_global: Whole image embeddings
    - visual_search_vehicles: Vehicle detections (car, truck, motorcycle, bus, boat)
    - visual_search_people: Person detections
    - visual_search_faces: Face identity embeddings (future: ArcFace)
    """

    def __init__(
        self,
        hosts: list[str] | None = None,
        http_auth: tuple | None = None,
        verify_certs: bool = False,
        ssl_show_warn: bool = False,
        timeout: int = 30,
    ):
        """
        Initialize OpenSearch async client.

        Args:
            hosts: List of OpenSearch node URLs
            http_auth: Tuple of (username, password) for authentication
            verify_certs: Whether to verify SSL certificates
            ssl_show_warn: Whether to show SSL warnings
            timeout: Request timeout in seconds
        """
        # Use OPENSEARCH_HOSTS env var if set (for Docker), otherwise localhost
        import json
        import os

        default_hosts = os.environ.get('OPENSEARCH_HOSTS')
        if default_hosts:
            try:
                default_hosts = json.loads(default_hosts)
            except json.JSONDecodeError:
                default_hosts = [default_hosts]
        else:
            default_hosts = ['http://localhost:9200']
        hosts = hosts or default_hosts
        client_kwargs = {
            'hosts': hosts,
            'use_ssl': False,
            'verify_certs': verify_certs,
            'ssl_show_warn': ssl_show_warn,
            'timeout': timeout,
        }

        if http_auth:
            client_kwargs['http_auth'] = http_auth

        self.client = AsyncOpenSearch(**client_kwargs)
        # Every OpenSearch client carries the project guard from birth
        # (projects_plan.md §2.4); unbound visual-search calls reach only
        # indexes no project owns.
        from src.services.projects.guard import install_project_guard
        from src.services.projects.registry import get_project_registry

        install_project_guard(self.client, get_project_registry())
        self.embedding_dim = 512  # MobileCLIP2-S2

    async def close(self):
        """Close the OpenSearch client connection."""
        await self.client.close()

    async def ping(self) -> bool:
        """Check if OpenSearch is reachable."""
        try:
            return await self.client.ping()
        except Exception as e:
            logger.error(f'OpenSearch ping failed: {e}')
            return False
