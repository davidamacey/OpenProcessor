"""Shared building blocks of the combined analysis pipeline (single and batch)."""

from functools import lru_cache
from typing import Any

from src.services.inference import InferenceService


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Get or create InferenceService singleton (cached)."""
    return InferenceService()


def ocr_failure(ocr_result: dict[str, Any] | None) -> str | None:
    """The message of an OCR result that failed (``None`` when it ran, even
    with no text, or was not requested): a failed OCR must not look like an
    image with no text."""
    if ocr_result and ocr_result.get('status') == 'error':
        return str(ocr_result.get('error') or 'OCR failed')
    return None
