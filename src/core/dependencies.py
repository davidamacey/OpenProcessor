"""
FastAPI dependency injection for shared resources.

Uses FastAPI's Depends() pattern for proper lifecycle management.
Resources are created once and reused across requests.
"""

import logging
from typing import Annotated, Any

import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as grpcclient_aio
from fastapi import Depends

from src.clients.opensearch.client import OpenSearchClient
from src.clients.triton_client_manager import TritonClientManager
from src.config.settings import get_settings


logger = logging.getLogger(__name__)


# =============================================================================
# Application State (managed by lifespan context)
# =============================================================================
class AppState:
    """
    Application state container for shared resources.

    Resources are initialized in lifespan and accessed via dependencies.
    This avoids global state while providing singleton-like behavior.
    """

    def __init__(self) -> None:
        self.pytorch_models: dict[str, Any] = {}
        self._triton_async_client: grpcclient_aio.InferenceServerClient | None = None
        self._opensearch_client: OpenSearchClient | None = None

    @property
    def triton_url(self) -> str:
        return get_settings().triton_url

    @property
    def opensearch_url(self) -> str:
        return get_settings().opensearch_url


# Global app state - initialized in lifespan
app_state = AppState()


# =============================================================================
# Triton Client Factory
# =============================================================================
class TritonClientFactory:
    """
    Factory for creating Triton inference clients.

    Provides both sync and async clients with connection pooling.
    """

    @staticmethod
    def get_sync_client() -> grpcclient.InferenceServerClient:
        """Get sync Triton gRPC client with connection pooling."""
        return TritonClientManager.get_sync_client(app_state.triton_url)

    @staticmethod
    async def get_async_client() -> grpcclient_aio.InferenceServerClient:
        """Get async Triton gRPC client (lazy initialization)."""
        if app_state._triton_async_client is None:
            logger.info(f'Initializing async Triton client ({app_state.triton_url})...')
            app_state._triton_async_client = await TritonClientManager.get_async_client(
                app_state.triton_url
            )
            logger.info('Async Triton client ready (batching enabled)')
        return app_state._triton_async_client

    @staticmethod
    async def close_all() -> None:
        """Close all Triton connections."""
        await TritonClientManager.close_all()
        app_state._triton_async_client = None
        logger.info('All Triton connections closed')


# =============================================================================
# OpenSearch Client Factory
# =============================================================================
class OpenSearchClientFactory:
    """Factory for OpenSearch client with lazy initialization."""

    @staticmethod
    async def get_client() -> OpenSearchClient:
        """Get OpenSearch client (lazy initialization)."""
        if app_state._opensearch_client is None:
            settings = get_settings()

            logger.info(f'Initializing OpenSearch client ({settings.opensearch_url})...')
            client = OpenSearchClient(
                hosts=[settings.opensearch_url], http_auth=None, timeout=settings.opensearch_timeout
            )
            # OpenSearchClient installs the project guard at construction.
            app_state._opensearch_client = client

            if not await client.ping():
                raise RuntimeError(f'OpenSearch connection failed: {settings.opensearch_url}')

            logger.info('OpenSearch client ready')
        return app_state._opensearch_client

    @staticmethod
    async def close() -> None:
        """Close OpenSearch client."""
        if app_state._opensearch_client is not None:
            await app_state._opensearch_client.close()
            app_state._opensearch_client = None
            logger.info('OpenSearch client closed')


async def bootstrap_opensearch_indexes() -> None:
    """Refuse to start unless every active project's indexes carry the current schema stamp.

    Index creation, the mapping migrations and the kNN warm-up run once in the ``db-migrate``
    step (``scripts/opensearch/migrate.py``), not in every worker: this only reads the stamps,
    one ``_mapping`` request per project, and raises
    :class:`~src.services.curation.index_bootstrap.MappingNotCurrentError`
    (``opensearch_mapping_not_current``, naming the index) when one is wrong or missing.
    A project the registry has not yet read cannot be checked here; the registry bootstrap
    that runs first retries until it can.
    """
    from src.clients.curation_opensearch.lifecycle import curation_index_names
    from src.services.curation.index_bootstrap import check_mappings_current
    from src.services.projects.bootstrap import for_each_project

    os_client: OpenSearchClient | None = None
    for slug in for_each_project(('active', 'archived')):
        if os_client is None:
            os_client = await OpenSearchClientFactory.get_client()
        await check_mappings_current(os_client.client, curation_index_names())
        logger.info(f'curation_index_mappings_current project={slug}')


# =============================================================================
# VisualSearchService Factory
# =============================================================================
class VisualSearchServiceFactory:
    """Factory for VisualSearchService with lazy initialization."""

    _instance: 'VisualSearchService | None' = None

    @classmethod
    async def get_service(cls) -> 'VisualSearchService':
        """Get VisualSearchService (lazy initialization)."""
        if cls._instance is None:
            from src.services.visual_search import VisualSearchService

            opensearch_client = await OpenSearchClientFactory.get_client()
            cls._instance = VisualSearchService(opensearch_client)
            logger.info('VisualSearchService initialized')
        return cls._instance

    @classmethod
    async def close(cls) -> None:
        """Close service resources."""
        cls._instance = None


# =============================================================================
# FastAPI Dependencies (use with Depends())
# =============================================================================
async def get_async_triton() -> grpcclient_aio.InferenceServerClient:
    """Dependency for async Triton client."""
    return await TritonClientFactory.get_async_client()


async def get_opensearch() -> OpenSearchClient:
    """Dependency for OpenSearch client."""
    return await OpenSearchClientFactory.get_client()


async def get_curation_opensearch() -> Any:
    """Dependency for the raw, project-guarded ``AsyncOpenSearch`` every
    curation route uses (one dependency for every curation router, so one
    override covers them all)."""
    from src.config.project_context import try_current_project
    from src.services.config_store import get_config_store
    from src.services.projects.guard import make_curation_opensearch

    client = await make_curation_opensearch()
    # Every project route reads the active region profile / prompt pack /
    # VLM from this API worker's snapshot (ingest seeds region work from it).
    # An activation made through another of the API's workers must reach this
    # one within a second, not only when a config route happens to land here.
    if try_current_project() is not None:
        await get_config_store().ensure_fresh()
    return client


async def get_visual_search_service() -> 'VisualSearchService':
    """Dependency for VisualSearchService."""
    return await VisualSearchServiceFactory.get_service()


def get_pytorch_models() -> dict[str, Any]:
    """Dependency for PyTorch models."""
    return app_state.pytorch_models


# Type aliases for cleaner endpoint signatures
AsyncTritonDep = Annotated[Any, Depends(get_async_triton)]
OpenSearchDep = Annotated[Any, Depends(get_opensearch)]
PyTorchModelsDep = Annotated[dict[str, Any], Depends(get_pytorch_models)]

# Import VisualSearchService for type annotation
from src.services.visual_search import VisualSearchService  # noqa: E402  # bottom-of-module import breaks a circular import


VisualSearchDep = Annotated[VisualSearchService, Depends(get_visual_search_service)]
