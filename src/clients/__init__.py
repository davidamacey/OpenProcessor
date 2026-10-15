"""
Client modules for external services.

Provides unified interfaces for Triton Inference Server and OpenSearch.

Triton Clients:
- TritonClient: Unified client for all inference pipelines
- TritonClientManager: Connection pool manager (from triton_client_manager)
"""

from src.clients.opensearch.client import OpenSearchClient
from src.clients.triton_client import TritonClient, get_sync_triton_client
from src.clients.triton_client_manager import TritonClientManager


__all__ = [
    # OpenSearch
    'OpenSearchClient',
    # Unified Triton client (NEW)
    'TritonClient',
    'TritonClientManager',
    'get_sync_triton_client',
]
