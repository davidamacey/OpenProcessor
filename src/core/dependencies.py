"""
FastAPI dependency injection for shared resources.

Uses FastAPI's Depends() pattern for proper lifecycle management.
Resources are created once and reused across requests.
"""

import asyncio
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


async def bootstrap_opensearch_indexes() -> 'asyncio.Task[None] | None':
    """Best-effort create the core + curation OpenSearch indexes at startup.

    Wrapped so a missing/not-yet-up OpenSearch instance doesn't block
    startup -- the curation router retries its own create on the first
    /curation/* request. Nothing else creates the core visual_search_*
    indexes, so a fresh install's first /ingest previously auto-created them with a dynamic
    (non-knn_vector) mapping, silently breaking every k-NN search
    forever after; create_all_indexes is idempotent, so calling it here
    on every boot is safe (see OpenSearchClient._search_index /
    IndexMappingError for the read-side half of this fix).

    Returns the background kNN-graph-warmup task (fire-and-forget, never
    blocks startup, its own failures are logged and swallowed), or None
    if bootstrap itself failed.
    """
    try:
        from src.services.curation.index_bootstrap import ensure_indexes, warm_knn_indexes
        from src.services.projects.bootstrap import for_each_project

        os_client = await OpenSearchClientFactory.get_client()
        await os_client.create_all_indexes(force_recreate=False)
        logger.info('core_visual_search_indexes_bootstrapped')
        for slug in for_each_project():
            # The full bootstrap (exists checks + mapping migrations), not just
            # index creation: it costs 380-470 ms cold and is cached per
            # (worker, project), so paying it here keeps it off the first
            # request of every recycled worker (#356). A failure is logged
            # inside and leaves the project uncached, so a request retries.
            await ensure_indexes(os_client.client)
            logger.info(f'curation_indexes_bootstrapped project={slug}')

        async def _warm_every_project() -> None:
            for _slug in for_each_project():
                await warm_knn_indexes(os_client.client)

        return asyncio.create_task(_warm_every_project())
    except Exception as exc:
        logger.warning(f'curation_indexes_bootstrap_skipped: {exc}')
        return None


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
        await get_config_store().ensure_fresh(client)
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
