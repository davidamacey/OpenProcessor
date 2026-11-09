"""Unified PE-Core-L14-336 client.

Two encoders ship under one class, both on Triton:

* **Image encoder**: the model ``pe_image_encoder`` through an :class:`AsyncTritonPool`.
* **Text encoder**: the text-tower graph from ``export/export_pe_text_encoder.py`` served as
  ``pe_text_encoder`` (onnxruntime backend, one CPU instance shared by every uvicorn worker),
  called synchronously through :class:`~src.clients.pe_text_backends.TritonTextBackend` and
  LRU-cached so the same phrase typed twice is never re-encoded. There is no in-process
  fallback: with the text model not ready :meth:`PEEncoder.warm_text_encoder` raises
  :class:`TextModelNotReady` (before the tokenizer, and so any ``torch`` import, is loaded) and
  semantic search answers 503 naming that model.

Tokenization stays in Python with PE's own ``SimpleTokenizer`` (context length 32). Because the
text tower's attention mask is strictly causal and pooling reads the EOT position, padding after
a row's EOT can never change its embedding, so the model is fed tokens trimmed to the batch's
last EOT (:func:`~src.clients.pe_text_backends.trim_text_tokens`).

Heavy imports (the tokenizer's package, tritonclient) live inside the loaders so this module
stays importable without them.

Embedding contract (both paths): shape ``(N, 1024)`` for a batch, ``float32``, each row
L2-normalized, so a dot product is the cosine similarity.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict, namedtuple
from typing import TYPE_CHECKING, Any

import numpy as np

from src.clients.pe_text_backends import (
    PE_TEXT_CHECKPOINT,
    PE_TEXT_CONTEXT_LENGTH,
    PE_TEXT_TRITON_MODEL,
    TritonTextBackend,
    trim_text_tokens,
)
from src.config.env import env_str
from src.config.settings import TritonModelConfig
from src.core.logging import get_logger
from src.utils.cpu_pool import run_cpu
from src.utils.pe_preprocess import (
    PE_SIZE,
    resize_crop_rgb,
    to_chw,
    to_chw_into,
    whole_frame_rgb_from_image,
)
from src.utils.search_timing import search_annotate
from src.utils.stage_timing import stage_timer


if TYPE_CHECKING:
    from PIL import Image

    from src.clients.triton_pool_core import AsyncTritonPool

# Chunk size for embed_crops -- respects the PE Triton model's configured
# max_batch_size on typical deployments.
PE_CROP_MAX_BATCH = 32


logger = get_logger(__name__)


PE_IMAGE_MODEL = TritonModelConfig.PE_IMAGE_MODEL
PE_EMBEDDING_DIM = 1024

# Maximum number of distinct text queries to keep cached. The labeler
# typically explores a few dozen prompts per session — 256 is plenty.
_TEXT_CACHE_SIZE = 256

TextCacheInfo = namedtuple('TextCacheInfo', ['hits', 'misses', 'maxsize', 'currsize'])


def _preprocess_crops(crops: list[np.ndarray]) -> np.ndarray:
    """PE preprocessing of a list of HWC-uint8 crops into one ``(N, 3, 336, 336)`` batch."""
    with stage_timer('pe_preprocess'):
        batch = np.empty((len(crops), 3, PE_SIZE, PE_SIZE), dtype=np.uint8)
        for slot, crop in zip(batch, crops, strict=True):
            to_chw_into(slot, resize_crop_rgb(crop))
        return batch


def _timed_whole_frame(preprocess: Any, source: Any) -> np.ndarray:
    """``preprocess(source)`` (a whole-frame square, decoding a path first) under the stage timer."""
    nbytes = len(source) if isinstance(source, bytes) else 0
    with stage_timer('whole_frame_preprocess', nbytes=nbytes):
        typed: np.ndarray = preprocess(source)
        return typed


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


class TextModelNotReady(RuntimeError):  # noqa: N818 - names the state, not an error class
    """Triton's PE text model is not ready; nothing else can answer a text query."""


class PEEncoder:
    """Unified PE-Core-L14-336 client.

    Both the image path (``pe_image_encoder``) and the text path (``pe_text_encoder``) run on
    Triton; text queries sit behind a per-query LRU cache. The text model name defaults to
    ``OP_PE_TEXT_TRITON_MODEL``; the keyword overrides it (tests, scripts).
    """

    def __init__(
        self,
        triton_pool: AsyncTritonPool | None = None,
        *,
        text_triton_model: str | None = None,
        triton_client_factory: Any = None,
    ) -> None:
        self.triton_pool = triton_pool
        self._text_triton_model = text_triton_model or env_str(
            'OP_PE_TEXT_TRITON_MODEL', PE_TEXT_TRITON_MODEL
        )
        self._triton_client_factory = triton_client_factory

        # Populated by warm_text_encoder().
        self._text_tokenizer: Any = None
        self._text_backend: Any = None
        self._text_ready: bool = False
        self._text_lock = threading.RLock()

        self._text_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Image path — Triton
    # ------------------------------------------------------------------

    async def encode_images(self, chws: np.ndarray) -> np.ndarray:
        """Encode pre-preprocessed image tensors via Triton.

        Args:
            chws: ``(N, 3, H, W)`` uint8 RGB pixels, already resized and
                center-cropped; the engine applies the PE-Core normalization.

        Returns:
            ``(N, 1024)`` L2-normalized float32.
        """
        if self.triton_pool is None:
            raise RuntimeError(
                'PEEncoder.encode_images requires a triton_pool; '
                'construct PEEncoder(triton_pool=...) before calling.'
            )
        if chws.ndim != 4 or chws.shape[1] != 3:
            raise ValueError(f'encode_images expects (N, 3, H, W) uint8; got shape {chws.shape}')

        from tritonclient.grpc import InferRequestedOutput

        from src.clients.triton_pool_core import uint8_images_input

        inp = await run_cpu(uint8_images_input, chws, nbytes=chws.nbytes)
        outs = [InferRequestedOutput('image_embeddings')]
        with stage_timer('embed', nbytes=chws.nbytes):
            result = await self.triton_pool.infer(PE_IMAGE_MODEL, [inp], outputs=outs)
        embeddings = np.asarray(result.as_numpy('image_embeddings'), dtype=np.float32)
        return _l2_normalize(embeddings)

    async def embed_crops(
        self,
        crops: list[np.ndarray],
        max_batch: int = PE_CROP_MAX_BATCH,
    ) -> np.ndarray:
        """Preprocess + encode a list of HWC-uint8 RGB item crops.

        Applies the canonical PE preprocessing (resize-shorter-edge,
        center-crop to uint8 pixels; the engine applies the PE-Core 0.5/0.5 normalize — see
        :mod:`src.utils.pe_preprocess`) to each crop, then
        chunks at ``max_batch`` to respect the encoder's configured Triton
        max batch size. :meth:`encode_images` already L2-normalizes its
        output, so the returned rows are unit-norm.

        Args:
            crops: HWC uint8 RGB arrays, any size (including degenerate /
                zero-size, which :func:`resize_crop_rgb` maps to zeros).
            max_batch: Max crops per Triton call.

        Returns:
            ``(len(crops), 1024)`` L2-normalized float32. Empty input ->
            ``(0, 1024)``.
        """
        if not crops:
            return np.zeros((0, PE_EMBEDDING_DIM), dtype=np.float32)

        chws = await run_cpu(_preprocess_crops, crops, nbytes=sum(c.nbytes for c in crops))

        rows: list[np.ndarray] = []
        for start in range(0, chws.shape[0], max_batch):
            chunk = chws[start : start + max_batch]
            rows.append(await self.encode_images(chunk))
        return np.concatenate(rows, axis=0)

    async def embed_whole_frame_image(self, img: Image.Image) -> np.ndarray:
        """PE-Core whole-frame embedding of a decoded frame (EXIF orientation applied).

        Goes through :func:`src.utils.pe_preprocess.whole_frame_rgb_from_image`, the
        one whole-frame preprocessing, so ingest, re-embedding and any backfill share a space.
        The caller passes the frame it already decoded: a photo is decoded once.
        """
        w, h = img.size
        rgb = await run_cpu(_timed_whole_frame, whole_frame_rgb_from_image, img, nbytes=w * h * 3)
        embedding = await self.encode_images(to_chw(rgb)[None, ...])
        typed: np.ndarray = embedding[0]
        return typed

    # ------------------------------------------------------------------
    # Text path: Triton's pe_text_encoder, LRU-cached per query
    # ------------------------------------------------------------------

    def warm_text_encoder(self) -> None:
        """Bind the Triton text backend and load the tokenizer (idempotent).

        Raises :class:`TextModelNotReady` when Triton does not report the text model ready; the
        tokenizer (and whatever it imports) is loaded only after that check passes.
        """
        if self._text_ready:
            return
        with self._text_lock:
            if self._text_ready:
                return
            backend = self._triton_backend()
            if self._text_tokenizer is None:
                self._text_tokenizer = self._load_tokenizer()
            self._text_backend = backend
            self._text_ready = True
            logger.info(
                'pe_text_encoder_ready',
                checkpoint=PE_TEXT_CHECKPOINT,
                model=self._text_triton_model,
            )

    def ensure_text_ready(self) -> bool:
        """:meth:`warm_text_encoder`, reporting instead of raising: whether text queries can run.

        Triton may come up after the API did, so a request retries the (cheap) readiness probe
        instead of trusting the one made at startup.
        """
        try:
            self.warm_text_encoder()
        except TextModelNotReady as exc:
            logger.info('pe_text_model_not_ready', error=str(exc))
            return False
        return True

    def prime_text_backend(self) -> None:
        """Send one trivial query to Triton, bypassing the LRU.

        Opens the gRPC channel and the model instance so a worker's first real search does not
        pay for them. A backend error propagates (the lifespan logs it and carries on; the first
        real query then retries).
        """
        if self._text_ready:
            self._encode_uncached(['warmup'])

    @property
    def text_model(self) -> str:
        """The Triton model that answers text queries."""
        return self._text_triton_model

    def text_status(self) -> dict[str, Any]:
        """Operator-facing snapshot for ``GET /health/pe_text``."""
        info = self.text_cache_info()
        return {
            # The LRU and its counters live in this uvicorn worker only; with N
            # workers each /health/pe_text call reports whichever one answered.
            'cache_scope': 'process',
            'pid': os.getpid(),
            'ready': self._text_ready,
            'triton_model': self._text_triton_model,
            'cache': info._asdict(),
        }

    # -- loaders ---------------------------------------------------------

    def _load_tokenizer(self) -> Any:
        # The `perception_models` pip distribution installs the top-level
        # package `core`. Import the tokenizer module directly: going through
        # core.vision_encoder.transforms would also pull in torchvision.
        from core.vision_encoder.tokenizer import SimpleTokenizer

        return SimpleTokenizer(context_length=PE_TEXT_CONTEXT_LENGTH)

    def _triton_backend(self) -> TritonTextBackend:
        try:
            client = self._make_triton_client()
            ready = client.is_model_ready(self._text_triton_model)
        except Exception as exc:
            raise TextModelNotReady(
                f'Triton is unreachable, so its text model {self._text_triton_model!r} is not '
                f'ready: {exc}'
            ) from exc
        if not ready:
            raise TextModelNotReady(f'Triton text model {self._text_triton_model!r} is not ready')
        return TritonTextBackend(client, self._text_triton_model)

    def _make_triton_client(self) -> Any:
        if self._triton_client_factory is not None:
            return self._triton_client_factory()
        from src.clients.triton_client_manager import TritonClientManager
        from src.config import get_settings

        return TritonClientManager.get_sync_client(get_settings().triton_url)

    # -- encoding --------------------------------------------------------

    def _tokenize(self, queries: list[str]) -> np.ndarray:
        tokens = np.asarray(self._text_tokenizer(queries), dtype=np.int64)
        return trim_text_tokens(tokens)

    def _encode_uncached(self, queries: list[str]) -> np.ndarray:
        if not self._text_ready:
            raise RuntimeError('PEEncoder text encoder not warmed; call warm_text_encoder() first.')
        tokens = self._tokenize(queries)
        vectors = self._text_backend.encode(tokens)
        return _l2_normalize(np.asarray(vectors, dtype=np.float32).reshape(len(queries), -1))

    def encode_text(self, queries: list[str]) -> np.ndarray:
        """Encode a batch of text queries with LRU caching.

        Cache misses are encoded together in one backend call.

        Args:
            queries: list of UTF-8 strings.

        Returns:
            ``(N, 1024)`` L2-normalized float32.
        """
        if not queries:
            return np.zeros((0, PE_EMBEDDING_DIM), dtype=np.float32)

        found: dict[str, np.ndarray] = {}
        missing: list[str] = []
        with self._cache_lock:
            for query in queries:
                if query in found or query in missing:
                    continue
                cached = self._text_cache.get(query)
                if cached is None:
                    missing.append(query)
                    self._cache_misses += 1
                else:
                    self._text_cache.move_to_end(query)
                    found[query] = cached
                    self._cache_hits += 1

        search_annotate('encode_cache_hit', int(not missing))
        if missing:
            vectors = self._encode_uncached(missing)
            with self._cache_lock:
                for query, vec in zip(missing, vectors, strict=True):
                    vec.setflags(write=False)
                    found[query] = vec
                    self._text_cache[query] = vec
                    self._text_cache.move_to_end(query)
                while len(self._text_cache) > _TEXT_CACHE_SIZE:
                    self._text_cache.popitem(last=False)

        return np.vstack([found[q] for q in queries]).astype(np.float32)

    def text_cache_info(self) -> TextCacheInfo:
        """Hits/misses/size of the per-query cache (``lru_cache``-shaped)."""
        with self._cache_lock:
            return TextCacheInfo(
                self._cache_hits, self._cache_misses, _TEXT_CACHE_SIZE, len(self._text_cache)
            )
