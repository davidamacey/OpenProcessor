"""
Client modules for external services.

Provides unified interfaces for Triton Inference Server and OpenSearch.

Triton Clients:
- TritonClient: Unified client for all inference pipelines
- TritonClientManager: Connection pool manager (from triton_client_manager)

The names below load on first access (PEP 562), so a light submodule such as
``src.clients.task_queue`` can be imported without pulling in the Triton and OpenSearch
clients (and torch with them) that the job runtime's worker children must not carry.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from src.clients.opensearch.client import OpenSearchClient
    from src.clients.triton_client import TritonClient, get_sync_triton_client
    from src.clients.triton_client_manager import TritonClientManager

_LAZY = {
    'OpenSearchClient': 'src.clients.opensearch.client',
    'TritonClient': 'src.clients.triton_client',
    'get_sync_triton_client': 'src.clients.triton_client',
    'TritonClientManager': 'src.clients.triton_client_manager',
}


def __getattr__(name: str) -> Any:
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from None
    return getattr(importlib.import_module(module), name)


__all__ = [
    # OpenSearch
    'OpenSearchClient',
    # Unified Triton client (NEW)
    'TritonClient',
    'TritonClientManager',
    'get_sync_triton_client',
]
