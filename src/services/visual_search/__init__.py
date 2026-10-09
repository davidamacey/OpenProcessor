"""
Visual Search Service.

Orchestrates inference + OpenSearch operations for visual search.
Bridges InferenceService (Triton) and OpenSearchClient (multi-index vector search).

Multi-Index Architecture:
- visual_search_global: Whole image similarity (scene matching)
- visual_search_vehicles: Vehicle detections (car, truck, motorcycle, bus, boat)
- visual_search_people: Person appearance (clothing, pose)
- visual_search_faces: Face identity matching (future: ArcFace)
- visual_search_ocr: Text content (OCR with trigram search)

Key Features:
- Auto-routing ingestion by detection class
- Category-specific search (vehicles, people, global)
- Index lifecycle management for all indexes

Usage:
    service = VisualSearchService(opensearch_client)

    # Ingest (auto-routes by class)
    result = await service.ingest_image(image_bytes, image_id)
    # Result: {global: True, vehicles: 2, people: 1, skipped: 0}

    # Category-specific search
    results = await service.search_global(query_bytes, top_k=10)
    results = await service.search_vehicles(query_embedding, top_k=10)
    results = await service.search_people(query_embedding, top_k=10)

The service is assembled from per-concern mixins: ``ingest_single``,
``ingest_batch``, ``search``, ``lifecycle`` and ``clusters``.
"""

from src.clients.opensearch.client import OpenSearchClient
from src.services.inference import InferenceService
from src.services.visual_search.clusters import ClusterMixin
from src.services.visual_search.ingest_batch import IngestBatchMixin
from src.services.visual_search.ingest_single import IngestSingleMixin
from src.services.visual_search.lifecycle import IndexLifecycleMixin
from src.services.visual_search.search import SearchMixin


class VisualSearchService(
    IngestSingleMixin,
    IngestBatchMixin,
    SearchMixin,
    IndexLifecycleMixin,
    ClusterMixin,
):
    """
    Service for visual search operations combining inference and OpenSearch.

    Google Photos-like capabilities:
    - Similar image search (whole scene matching)
    - Search for people by appearance
    - Search for vehicles
    - Search for faces (future - ArcFace identity matching)
    - Extensible for products, animals, etc.

    Design:
    - Accepts OpenSearchClient via dependency injection (testability)
    - Creates InferenceService internally (follows existing patterns)
    - All OpenSearch methods are async (client is async)
    - Inference is sync (FastAPI thread pool handles blocking)
    """

    def __init__(self, opensearch_client: OpenSearchClient):
        """
        Initialize visual search service.

        Args:
            opensearch_client: Async OpenSearch client for vector operations
        """
        self.opensearch = opensearch_client
        self.inference = InferenceService()
