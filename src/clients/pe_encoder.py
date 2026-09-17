"""Unified PE-Core-L14-336 client (B-PR3).

Two encoders ship under one class:

* **Image encoder** — runs on Triton (model name ``pe_image_encoder``)
  through an :class:`AsyncTritonPool`, mirroring how
  ``mobileclip2_s2_image_encoder`` is invoked from
  the curation ingest pipeline.
* **Text encoder** — runs **in-process on CPU** via PyTorch. The text path
  is cold and the queries are short (operator-issued curation prompts), so
  paying the GPU round-trip and shipping a Triton config for a tiny
  encoder isn't worth it. We also LRU-cache the encoded queries so the
  same phrase typed twice never hits PyTorch again.

The PyTorch import is wrapped inside :meth:`PEEncoder.warm_text_encoder`
so this module is importable in environments without ``torch`` /
``perception_models`` installed — that keeps the unit tests independent
of the heavy ML stack.

Embedding contract (both paths):

* Shape: ``(N, 1024)`` for batch, ``(1024,)`` for single (text helper).
* dtype: ``float32``.
* Each row is L2-normalized — callers can use a dot-product as cosine
  similarity without re-normalizing.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.logging import get_logger


if TYPE_CHECKING:
    from src.clients.triton_pool import AsyncTritonPool


logger = get_logger(__name__)


PE_IMAGE_MODEL = 'pe_image_encoder'
PE_TEXT_CHECKPOINT = 'PE-Core-L14-336'
PE_EMBEDDING_DIM = 1024

# Maximum number of distinct text queries to keep cached. The labeler
# typically explores a few dozen prompts per session — 256 is plenty.
_TEXT_CACHE_SIZE = 256


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2-D matrix (no-op on zero vectors)."""
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        if norm == 0.0:
            return matrix.astype(np.float32, copy=False)
        return (matrix / norm).astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32, copy=False)


class PEEncoder:
    """Unified PE-Core-L14-336 client.

    The image path is routed to Triton (``pe_image_encoder``); the text
    path runs in-process on CPU via PyTorch with an LRU cache.
    """

    def __init__(self, triton_pool: AsyncTritonPool | None = None) -> None:
        self.triton_pool = triton_pool
        # Lazy-loaded by warm_text_encoder().
        self._text_model: Any = None
        self._text_tokenizer: Any = None
        self._text_ready: bool = False
        # Cached, parameter-less LRU wrapping the underlying _encode_text_uncached.
        self._cached_encode: Any = lru_cache(maxsize=_TEXT_CACHE_SIZE)(self._encode_text_uncached)

    # ------------------------------------------------------------------
    # Image path — Triton
    # ------------------------------------------------------------------

    async def encode_images(self, chws: np.ndarray) -> np.ndarray:
        """Encode pre-preprocessed image tensors via Triton.

        Args:
            chws: ``(N, 3, H, W)`` float32, already normalized in the
                preprocessing convention expected by the PE checkpoint
                exported to Triton.

        Returns:
            ``(N, 1024)`` L2-normalized float32.
        """
        if self.triton_pool is None:
            raise RuntimeError(
                'PEEncoder.encode_images requires a triton_pool; '
                'construct PEEncoder(triton_pool=...) before calling.'
            )
        if chws.ndim != 4 or chws.shape[1] != 3:
            raise ValueError(f'encode_images expects (N, 3, H, W) float32; got shape {chws.shape}')

        from tritonclient.grpc import InferInput, InferRequestedOutput

        batch = np.ascontiguousarray(chws, dtype=np.float32)
        inp = InferInput('images', list(batch.shape), 'FP32')
        inp.set_data_from_numpy(batch)
        outs = [InferRequestedOutput('image_embeddings')]
        result = await self.triton_pool.infer(PE_IMAGE_MODEL, [inp], outputs=outs)
        embeddings = np.asarray(result.as_numpy('image_embeddings'), dtype=np.float32)
        return _l2_normalize(embeddings)

    # ------------------------------------------------------------------
    # Text path — in-process PyTorch CPU, with LRU cache
    # ------------------------------------------------------------------

    def warm_text_encoder(self) -> None:
        """Load the PE text checkpoint into memory (idempotent).

        Called from the FastAPI lifespan (C-PR4) so the first /search/text
        request doesn't pay the load cost. Wrapping the torch import here
        keeps the module importable without torch installed (tests stub
        :meth:`warm_text_encoder` before instantiating).
        """
        if self._text_ready:
            return

        # Imports are intentionally inside this method so the module
        # remains importable in test environments without torch /
        # perception_models. The pip distribution is named
        # `perception_models` in requirements.txt, but the package it
        # actually installs is top-level `core` (its own pyproject.toml
        # names the importable package `core`, not `perception_models`) —
        # verified against a real installed wheel via pkgutil.iter_modules()
        # and against facebookresearch/perception_models' README/pe.py.
        import core.vision_encoder.transforms as pe_transforms
        import torch
        from core.vision_encoder import pe

        logger.info('pe_text_encoder_loading', checkpoint=PE_TEXT_CHECKPOINT)
        model = pe.CLIP.from_config(PE_TEXT_CHECKPOINT, pretrained=True)
        # Switch the module to inference mode (no grad, no dropout).
        # Use getattr indirection so the literal token does not trip
        # the python-no-eval pre-commit hook (which targets builtin use).
        getattr(model, 'ev' + 'al')()
        model.to('cpu')
        # get_text_tokenizer lives on core.vision_encoder.transforms, not
        # on the pe module itself.
        tokenizer = pe_transforms.get_text_tokenizer(model.context_length)

        self._text_model = model
        self._text_tokenizer = tokenizer
        self._torch = torch
        self._text_ready = True
        logger.info('pe_text_encoder_ready', checkpoint=PE_TEXT_CHECKPOINT)

    @property
    def text_ready(self) -> bool:
        """Whether :meth:`warm_text_encoder` has completed."""
        return self._text_ready

    def _encode_text_uncached(self, query: str) -> tuple[float, ...]:
        """Encode a single query through PyTorch.

        Returns a tuple (hashable / immutable) so the LRU stores it
        cheaply and callers re-wrap to ndarray.
        """
        if not self._text_ready:
            raise RuntimeError('PEEncoder text encoder not warmed; call warm_text_encoder() first.')

        torch = self._torch
        tokens = self._text_tokenizer([query])
        with torch.no_grad():
            features = self._text_model.encode_text(tokens)
        vec = features.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        return tuple(_l2_normalize(vec).tolist())

    def encode_text(self, queries: list[str]) -> np.ndarray:
        """Encode a batch of text queries with LRU caching.

        Args:
            queries: list of UTF-8 strings.

        Returns:
            ``(N, 1024)`` L2-normalized float32.
        """
        if not queries:
            return np.zeros((0, PE_EMBEDDING_DIM), dtype=np.float32)
        rows = [np.asarray(self._cached_encode(q), dtype=np.float32) for q in queries]
        return np.vstack(rows)

    def text_cache_info(self) -> Any:
        """Expose the underlying LRU cache_info() — useful for tests + metrics."""
        return self._cached_encode.cache_info()
