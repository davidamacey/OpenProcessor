"""
Client modules for external services.

Provides unified interfaces for Triton Inference Server and OpenSearch.

Triton Clients:
- TritonClient: Unified client for all inference pipelines
- TritonClientManager: Connection pool manager (from triton_pool)
"""

from src.clients.opensearch import OpenSearchClient
from src.clients.triton_client import TritonClient, get_triton_client
from src.clients.triton_pool import TritonClientManager


__all__ = [
    # OpenSearch
    'OpenSearchClient',
    # Unified Triton client (NEW)
    'TritonClient',
    'TritonClientManager',
    'get_triton_client',
]
