"""
Face Identity Service for 1:N face identification.

Provides face identity management and search capabilities:
- Face ingestion with automatic embedding extraction
- 1:N face identification against database
- Person management (grouping faces by identity)
- Face-to-person assignment

Uses OpenSearch 'visual_search_faces' index for storage and k-NN search.

The service is assembled from per-concern mixins: ``identification``,
``face_records`` and ``person_management``.
"""

from functools import lru_cache

from src.clients.opensearch import OpenSearchClient
from src.config.settings import Settings, get_settings
from src.services.face_identity.face_records import FaceRecordMixin
from src.services.face_identity.identification import IdentificationMixin
from src.services.face_identity.person_management import PersonManagementMixin
from src.services.inference import InferenceService


class FaceIdentityService(IdentificationMixin, FaceRecordMixin, PersonManagementMixin):
    """
    Service for face identity management and 1:N identification.

    Features:
    - Ingest faces with ArcFace embeddings
    - 1:N face identification (find matching persons in database)
    - Person management (assign faces to persons)
    - Face quality assessment and thumbnail generation

    Design:
    - Uses OpenSearchClient for vector storage and k-NN search
    - Uses InferenceService for face detection and embedding extraction
    - Stores face crops as base64 thumbnails for UI preview
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize face identity service.

        Args:
            settings: Optional settings instance. If None, uses singleton.
        """
        self.settings = settings or get_settings()
        self.inference = InferenceService()
        self._opensearch: OpenSearchClient | None = None

    @property
    def opensearch(self) -> OpenSearchClient:
        """Lazy-load OpenSearch client."""
        if self._opensearch is None:
            self._opensearch = OpenSearchClient(
                hosts=[self.settings.opensearch_url],
                timeout=self.settings.opensearch_timeout,
            )
        return self._opensearch

    async def close(self):
        """Close OpenSearch connection."""
        if self._opensearch is not None:
            await self._opensearch.close()
            self._opensearch = None


# Singleton instance


@lru_cache(maxsize=1)
def get_face_identity_service() -> FaceIdentityService:
    """
    Get singleton FaceIdentityService instance (cached).

    Returns:
        FaceIdentityService: Service instance
    """
    return FaceIdentityService()
