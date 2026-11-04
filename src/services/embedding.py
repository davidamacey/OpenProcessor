"""
Embedding service for MobileCLIP encoding.

Handles image and text embedding generation with caching.
"""

import hashlib
import logging

import numpy as np
import tritonclient.grpc.aio as grpcclient_aio
from tritonclient.grpc import InferInput, InferRequestedOutput

from src.clients.model_residency import ensure_loaded
from src.clients.triton_client_manager import TritonClientManager
from src.clients.triton_deadline import ainfer_with_deadline
from src.clients.triton_preprocess import preprocess_for_mobileclip
from src.config import get_settings
from src.utils.cache import clip_text_tokens, get_image_cache, get_text_cache


logger = logging.getLogger(__name__)

# MobileCLIP2-S2 embedding dimension (from model config.pbtxt)
EMBEDDING_DIM = 512


class EmbeddingService:
    """
    MobileCLIP2-S2 embedding service with caching.

    Provides efficient encoding for:
    - Image embeddings (512-dim, L2-normalized)
    - Text embeddings (512-dim, L2-normalized)
    - Box embeddings (512-dim per detected object)

    Uses native async gRPC for Triton batching.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._triton_client: grpcclient_aio.InferenceServerClient | None = None
        self.embedding_dim = EMBEDDING_DIM

    async def _get_client(self) -> grpcclient_aio.InferenceServerClient:
        """Get async Triton client (lazy init)."""
        if self._triton_client is None:
            self._triton_client = await TritonClientManager.get_async_client(
                self.settings.triton_url
            )
        return self._triton_client

    # =========================================================================
    # Embedding Generation
    # =========================================================================
    async def encode_image(self, image_bytes: bytes, use_cache: bool = True) -> np.ndarray:
        """
        Encode image to 512-dim MobileCLIP embedding.

        Uses MobileCLIP2-S2 image encoder via Triton.
        Enables async gRPC for concurrent request batching.

        Args:
            image_bytes: JPEG/PNG bytes
            use_cache: Use embedding cache (default True)

        Returns:
            512-dim L2-normalized embedding vector
        """
        # Check cache
        cache_key = None
        if use_cache:
            cache = get_image_cache()
            cache_key = hashlib.sha256(image_bytes).hexdigest()
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        # Compute embedding
        client = await self._get_client()
        img_array = preprocess_for_mobileclip(image_bytes)

        input_tensor = InferInput('images', [1, 3, 256, 256], 'FP32')
        input_tensor.set_data_from_numpy(img_array)

        output = InferRequestedOutput('image_embeddings')

        await ensure_loaded(client, self.settings.models.CLIP_IMAGE_MODEL)
        response = await ainfer_with_deadline(
            client, self.settings.models.CLIP_IMAGE_MODEL, [input_tensor], [output]
        )

        embedding = response.as_numpy('image_embeddings')[0]

        # Cache result
        if use_cache and cache_key:
            cache.set(cache_key, embedding)

        typed: np.ndarray = embedding
        return typed

    async def encode_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Encode text to 512-dim MobileCLIP embedding.

        Uses MobileCLIP2-S2 text encoder via Triton.
        Tokenizer is cached singleton (~500ms savings on first call).

        Args:
            text: Query text string
            use_cache: Use embedding cache (default True)

        Returns:
            512-dim L2-normalized embedding vector
        """
        # Check cache
        cache_key = None
        if use_cache:
            cache = get_text_cache()
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        tokens = clip_text_tokens(text)

        # Compute embedding
        client = await self._get_client()

        input_tensor = InferInput('text_tokens', [1, 77], 'INT64')
        input_tensor.set_data_from_numpy(tokens)

        output = InferRequestedOutput('text_embeddings')

        await ensure_loaded(client, self.settings.models.CLIP_TEXT_MODEL)
        response = await ainfer_with_deadline(
            client, self.settings.models.CLIP_TEXT_MODEL, [input_tensor], [output]
        )

        embedding = response.as_numpy('text_embeddings')[0]

        # Cache result
        if use_cache and cache_key:
            cache.set(cache_key, embedding)

        typed: np.ndarray = embedding
        return typed
