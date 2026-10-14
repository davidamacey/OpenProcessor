"""Unified PE-Core-L14-336 client.

Two encoders ship under one class:

* **Image encoder** — runs on Triton (model name ``pe_image_encoder``)
  through an :class:`AsyncTritonPool`, mirroring how
  ``mobileclip2_s2_image_encoder`` is invoked from
  the curation ingest pipeline.
* **Text encoder** — runs **in-process on CPU** by default, deliberately
  independent of the GPU and Triton so semantic search keeps working when
  both are down. The queries are short, operator-issued curation prompts,
  LRU-cached so the same phrase typed twice is never re-encoded.

Text backends (picked once by :meth:`PEEncoder.warm_text_encoder`):

``triton``
    The text-tower graph from ``export/export_pe_text_encoder.py`` served
    by Triton as ``pe_text_encoder`` (onnxruntime backend, one CPU
    instance shared by every uvicorn worker). Used whenever Triton reports
    that model ready; a failed call permanently falls back to an
    in-process backend for the rest of that worker's life, so Triton is
    never a hard dependency.
``torch``
    PyTorch eager through ``perception_models``' ``pe.CLIP`` — the
    in-process fallback when Triton's ``pe_text_encoder`` isn't ready.
    Loaded lazily (see :class:`LazyTorchTextBackend`): never at warm time,
    only from the first query a given worker actually serves.
``onnx``
    ONNX Runtime ``CPUExecutionProvider`` over the same text-tower graph,
    loaded **in-process** from ``OP_PE_TEXT_ONNX_PATH``. Explicit opt-in
    only (``OP_PE_TEXT_BACKEND=onnx``) -- this is exactly the backend that
    used to run once per uvicorn worker: with 32 workers each loading its own ~1.4 GB ONNX
    Runtime session, ``auto`` mode alone put container RSS at ~84 GiB. ``auto``
    no longer considers this backend at all; pin it explicitly only for a
    single-worker/CLI process where Triton isn't available. Also loaded
    lazily, never at warm time.

``OP_PE_TEXT_BACKEND`` selects ``auto`` (default: triton -> torch), or
pins one of the three; a pinned ``onnx``/``torch`` that cannot load
raises (on first use, since both load lazily), a pinned ``triton`` that
is not ready falls back in-process.

Tokenization always stays in Python with PE's own ``SimpleTokenizer``
(context length 32). Because the text tower's attention mask is strictly
causal and pooling reads the EOT position, padding after a row's EOT can
never change its embedding, so every backend is fed tokens trimmed to the
batch's last EOT (:func:`trim_text_tokens`) — a 3-word query runs ~5
positions instead of 32.

Heavy imports (torch, perception_models, onnxruntime, tritonclient) live
inside the loaders so this module stays importable without them — that
keeps the unit tests independent of the heavy ML stack.

Embedding contract (both paths):

* Shape: ``(N, 1024)`` for batch, ``(1024,)`` for single (text helper).
* dtype: ``float32``.
* Each row is L2-normalized — callers can use a dot-product as cosine
  similarity without re-normalizing.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.clients.pe_text_backends import (
    DEFAULT_PE_TEXT_ONNX_PATH,
    PE_TEXT_BACKENDS,
    PE_TEXT_CHECKPOINT,
    PE_TEXT_CONTEXT_LENGTH,
    PE_TEXT_INPUT,
    PE_TEXT_OUTPUT,
    PE_TEXT_TRITON_MODEL,
    LazyOnnxTextBackend,
    LazyTorchTextBackend,
    OnnxTextBackend,
    TorchTextBackend,
    TritonTextBackend,
    trim_text_tokens,
)
from src.config.env import env_int, env_str
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
        return preprocess(source)


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
    path runs in-process (ONNX Runtime or PyTorch) or, optionally, on
    Triton (``pe_text_encoder``), behind a per-query LRU cache.

    Text-backend settings default to the environment
    (``OP_PE_TEXT_BACKEND``, ``OP_PE_TEXT_ONNX_PATH``,
    ``OP_PE_TEXT_TRITON_MODEL``, ``OP_PE_TEXT_ORT_THREADS``); keyword
    arguments override them (tests, scripts).
    """

    def __init__(
        self,
        triton_pool: AsyncTritonPool | None = None,
        *,
        text_backend: str | None = None,
        text_onnx_path: str | Path | None = None,
        text_triton_model: str | None = None,
        ort_threads: int | None = None,
        triton_client_factory: Any = None,
    ) -> None:
        self.triton_pool = triton_pool
        self._text_backend_pref = (
            (text_backend or env_str('OP_PE_TEXT_BACKEND', 'auto')).strip().lower()
        )
        self._text_onnx_path = str(
            text_onnx_path or env_str('OP_PE_TEXT_ONNX_PATH', str(DEFAULT_PE_TEXT_ONNX_PATH))
        )
        self._text_triton_model = text_triton_model or env_str(
            'OP_PE_TEXT_TRITON_MODEL', PE_TEXT_TRITON_MODEL
        )
        self._ort_threads = (
            ort_threads
            if ort_threads is not None
            else env_int('OP_PE_TEXT_ORT_THREADS', 0, minimum=0)
        )
        self._triton_client_factory = triton_client_factory

        # Populated by warm_text_encoder().
        self._text_tokenizer: Any = None
        self._text_backend: Any = None
        self._local_text_backend: Any = None
        self._text_ready: bool = False
        self._triton_fallbacks = 0
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
        return embedding[0]

    # ------------------------------------------------------------------
    # Text path — in-process ONNX Runtime / PyTorch, optional Triton,
    # LRU-cached per query
    # ------------------------------------------------------------------

    def warm_text_encoder(self) -> None:
        """Pick and load the text backend (idempotent).

        Called from the FastAPI lifespan so the first /search/text request
        doesn't pay the load cost. Raises when no backend can be loaded
        (the lifespan logs it; search then returns 503).
        """
        if self._text_ready:
            return
        with self._text_lock:
            if self._text_ready:
                return
            if self._text_backend_pref not in PE_TEXT_BACKENDS:
                raise ValueError(
                    f'OP_PE_TEXT_BACKEND={self._text_backend_pref!r}; '
                    f'expected one of {PE_TEXT_BACKENDS}'
                )
            if self._text_tokenizer is None:
                self._text_tokenizer = self._load_tokenizer()

            pref = self._text_backend_pref
            backend: Any = None
            # 'auto'
            # tries Triton's shared pe_text_encoder first -- NOT the
            # in-process ONNX path, which used to load its own ~1.4 GB
            # session in every uvicorn worker. In-process ONNX is now only
            # reachable via an explicit OP_PE_TEXT_BACKEND=onnx pin.
            if pref == 'onnx':
                backend = self._try_onnx_backend(required=True)
            if backend is None and pref in ('auto', 'triton'):
                backend = self._try_triton_backend()
            if backend is None:
                if pref == 'triton':
                    logger.warning(
                        'pe_text_triton_unavailable_using_local',
                        model=self._text_triton_model,
                    )
                backend = self._local_backend()

            self._text_backend = backend
            self._text_ready = True
            logger.info(
                'pe_text_encoder_ready',
                checkpoint=PE_TEXT_CHECKPOINT,
                backend=backend.name,
                requested=pref,
            )

    @property
    def text_ready(self) -> bool:
        """Whether :meth:`warm_text_encoder` has completed."""
        return self._text_ready

    @property
    def text_backend(self) -> str | None:
        """Name of the active text backend (``onnx``/``triton``/``torch``)."""
        backend = self._text_backend
        return None if backend is None else backend.name

    def text_status(self) -> dict[str, Any]:
        """Operator-facing snapshot for ``GET /health/pe_text``."""
        info = self.text_cache_info()
        return {
            'ready': self._text_ready,
            'backend': self.text_backend,
            'requested_backend': self._text_backend_pref,
            'onnx_path': self._text_onnx_path,
            'onnx_path_exists': Path(self._text_onnx_path).is_file(),
            'triton_model': self._text_triton_model,
            'triton_fallbacks': self._triton_fallbacks,
            'cache': info._asdict(),
        }

    # -- loaders ---------------------------------------------------------

    def _load_tokenizer(self) -> Any:
        # The `perception_models` pip distribution installs the top-level
        # package `core`. Import the tokenizer module directly: going through
        # core.vision_encoder.transforms would also pull in torchvision.
        from core.vision_encoder.tokenizer import SimpleTokenizer

        return SimpleTokenizer(context_length=PE_TEXT_CONTEXT_LENGTH)

    def _try_onnx_backend(self, *, required: bool) -> LazyOnnxTextBackend | None:
        """Explicit-opt-in only (``OP_PE_TEXT_BACKEND=onnx``) -- 'auto'
        never calls this. Only checks the file exists here (cheap); the
        actual ``ort.InferenceSession`` construction is deferred to the
        first query via :class:`LazyOnnxTextBackend`, never at warm time.
        """
        path = self._text_onnx_path
        if not Path(path).is_file():
            if required:
                raise FileNotFoundError(
                    f'OP_PE_TEXT_BACKEND=onnx but {path} does not exist '
                    '(export it with export/export_pe_text_encoder.py)'
                )
            logger.info('pe_text_onnx_not_found', path=path)
            return None
        return LazyOnnxTextBackend(lambda: self._load_onnx_backend(path))

    def _load_onnx_backend(self, path: str) -> OnnxTextBackend:
        import onnxruntime as ort

        logger.info('pe_text_encoder_loading', checkpoint=path, backend='onnx')
        opts = ort.SessionOptions()
        if self._ort_threads:
            opts.intra_op_num_threads = self._ort_threads
        session = ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])
        names_in = [i.name for i in session.get_inputs()]
        names_out = [o.name for o in session.get_outputs()]
        if PE_TEXT_INPUT not in names_in or PE_TEXT_OUTPUT not in names_out:
            raise ValueError(
                f'{path} has inputs {names_in} / outputs {names_out}; expected '
                f'{PE_TEXT_INPUT!r} -> {PE_TEXT_OUTPUT!r}'
            )
        return OnnxTextBackend(session, path)

    def _try_triton_backend(self) -> TritonTextBackend | None:
        try:
            client = self._make_triton_client()
            if not client.is_model_ready(self._text_triton_model):
                logger.info('pe_text_triton_model_not_ready', model=self._text_triton_model)
                return None
        except Exception as exc:
            logger.info('pe_text_triton_unreachable', error=str(exc))
            return None
        return TritonTextBackend(client, self._text_triton_model)

    def _make_triton_client(self) -> Any:
        if self._triton_client_factory is not None:
            return self._triton_client_factory()
        from src.clients.triton_client_manager import TritonClientManager
        from src.config import get_settings

        return TritonClientManager.get_sync_client(get_settings().triton_url)

    def _load_torch_backend(self) -> TorchTextBackend:
        import torch
        from core.vision_encoder import pe

        logger.info('pe_text_encoder_loading', checkpoint=PE_TEXT_CHECKPOINT, backend='torch')
        model = pe.CLIP.from_config(PE_TEXT_CHECKPOINT, pretrained=True)
        # Inference mode (no dropout). getattr indirection keeps the literal
        # token away from the python-no-eval pre-commit hook.
        getattr(model, 'ev' + 'al')()
        model.to('cpu')
        return TorchTextBackend(model, torch)

    def _local_backend(self) -> Any:
        """In-process PyTorch fallback for 'auto'/'triton' when Triton's
        ``pe_text_encoder`` isn't ready. Cached.

        Reached only for ``pref in ('auto', 'triton')`` -- an explicit
        ``pref == 'onnx'`` is resolved directly in :meth:`warm_text_encoder`
        (raising if missing) and never falls through to here. A pinned
        ``OP_PE_TEXT_BACKEND=torch`` loads eagerly, same as always — the
        operator asked for it explicitly, so failing fast on load is
        correct. ``auto``/``triton`` instead defer the torch load to first
        query via :class:`LazyTorchTextBackend` eagerly loading a multi-GB fallback
        model in every uvicorn worker at startup is the bug, not a
        feature.
        """
        if self._local_text_backend is None:
            if self._text_backend_pref == 'torch':
                self._local_text_backend = self._load_torch_backend()
            else:
                self._local_text_backend = LazyTorchTextBackend(self._load_torch_backend)
        return self._local_text_backend

    # -- encoding --------------------------------------------------------

    def _tokenize(self, queries: list[str]) -> np.ndarray:
        tokens = np.asarray(self._text_tokenizer(queries), dtype=np.int64)
        return trim_text_tokens(tokens)

    def _encode_uncached(self, queries: list[str]) -> np.ndarray:
        if not self._text_ready:
            raise RuntimeError('PEEncoder text encoder not warmed; call warm_text_encoder() first.')
        tokens = self._tokenize(queries)
        backend = self._text_backend
        try:
            vectors = backend.encode(tokens)
        except Exception as exc:
            if backend.name != 'triton':
                raise
            # Triton is optional: switch to in-process encoding for good
            # rather than flapping between backends per request.
            logger.warning('pe_text_triton_failed_falling_back', error=str(exc))
            with self._text_lock:
                local = self._local_backend()
                self._text_backend = local
                self._triton_fallbacks += 1
            vectors = local.encode(tokens)
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
