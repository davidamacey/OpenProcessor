"""
Custom exceptions for the YOLO inference service.

Provides domain-specific exceptions for better error handling.
"""

from typing import Any


class InferenceError(Exception):
    """Base exception for inference-related errors."""

    def __init__(self, message: str, model_name: str | None = None):
        self.message = message
        self.model_name = model_name
        super().__init__(self.message)


class ModelNotFoundError(InferenceError):
    """Raised when requested model is not available."""

    def __init__(self, model_name: str, available_models: list):
        self.available_models = available_models
        message = f"Model '{model_name}' not found. Available: {available_models}"
        super().__init__(message, model_name)


class InvalidImageError(InferenceError):
    """Raised when image validation fails."""

    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        message = f"Invalid image '{filename}': {reason}"
        super().__init__(message)


class ClientConnectionError(InferenceError):
    """Raised when client connection fails."""

    def __init__(self, service: str, url: str, reason: str):
        self.service = service
        self.url = url
        self.reason = reason
        message = f'Failed to connect to {service} at {url}: {reason}'
        super().__init__(message)


class RequestRejectedError(Exception):
    """A service refused the request. The app answers ``status_code`` with
    ``{"detail": detail}``, the body an ``HTTPException`` carries, so services
    stay free of HTTP types."""

    def __init__(self, status_code: int, detail: Any):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f'{status_code}: {detail}')
