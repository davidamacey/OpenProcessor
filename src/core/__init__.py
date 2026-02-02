"""
Core module with shared dependencies, exception handling, and logging.

Provides FastAPI dependency injection, custom exceptions, and structured logging.
"""

from src.core.dependencies import (
    AsyncTritonDep,
    OpenSearchClientFactory,
    OpenSearchDep,
    PyTorchModelsDep,
    TritonClientFactory,
    app_state,
    get_async_triton,
    get_opensearch,
    get_pytorch_models,
)
from src.core.exceptions import (
    ClientConnectionError,
    InferenceError,
    InvalidImageError,
    ModelNotFoundError,
)
from src.core.logging import configure_logging, get_logger


__all__ = [
    'AsyncTritonDep',
    'ClientConnectionError',
    'InferenceError',
    'InvalidImageError',
    'ModelNotFoundError',
    'OpenSearchClientFactory',
    'OpenSearchDep',
    'PyTorchModelsDep',
    'TritonClientFactory',
    'app_state',
    'configure_logging',
    'get_async_triton',
    'get_logger',
    'get_opensearch',
    'get_pytorch_models',
]
