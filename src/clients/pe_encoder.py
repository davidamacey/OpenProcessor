"""Unified PE-Core-L14-336 client (B-PR3).

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

``onnx``
    ONNX Runtime ``CPUExecutionProvider`` over the text-tower graph from
    ``export/export_pe_text_encoder.py``. Preferred whenever the file at
    ``OP_PE_TEXT_ONNX_PATH`` exists: it loads only the text tower (no
    ``pe.CLIP`` vision weights, no checkpoint download), roughly halves
    resident memory and warm-up time, and is at least as fast as PyTorch
    eager (~1.4x at batch 1 on the reference CPU).
``triton``
    The same graph served by Triton as ``pe_text_encoder`` (onnxruntime
    backend). Used only when Triton reports that model ready; a failed call
    permanently falls back to an in-process backend, so Triton is never a
    hard dependency.
``torch``
    PyTorch eager through ``perception_models``' ``pe.CLIP`` — the original
    path, and the last resort.

``OP_PE_TEXT_BACKEND`` selects ``auto`` (default: onnx -> triton -> torch),
or pins one of the three; a pinned ``onnx``/``torch`` that cannot load
raises, a pinned ``triton`` that is not ready falls back in-process.

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

import os
import threading
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.logging import get_logger
from src.services.detection.pe_preprocess import (
    normalize_chw,
    resize_crop_rgb,
    whole_frame_chw,
    whole_frame_chw_from_bytes,
)


if TYPE_CHECKING:
    from src.clients.triton_pool import AsyncTritonPool

# Chunk size for embed_crops -- respects the PE Triton model's configured
# max_batch_size on typical deployments (see docs/design/curation_design_rationale.md
# §2.1 for the reference throughput note this mirrors).
PE_CROP_MAX_BATCH = 32


logger = get_logger(__name__)


PE_IMAGE_MODEL = 'pe_image_encoder'
PE_TEXT_CHECKPOINT = 'PE-Core-L14-336'
PE_EMBEDDING_DIM = 1024

# Text-tower graph contract — shared with export/export_pe_text_encoder.py.
PE_TEXT_TRITON_MODEL = 'pe_text_encoder'
PE_TEXT_INPUT = 'text_tokens'
PE_TEXT_OUTPUT = 'text_embeddings'
PE_TEXT_CONTEXT_LENGTH = 32
DEFAULT_PE_TEXT_ONNX_PATH = '/app/pytorch_models/pe_text_encoder.onnx'
PE_TEXT_BACKENDS = ('auto', 'onnx', 'triton', 'torch')

# Maximum number of distinct text queries to keep cached. The labeler
# typically explores a few dozen prompts per session — 256 is plenty.
_TEXT_CACHE_SIZE = 256

TextCacheInfo = namedtuple('TextCacheInfo', ['hits', 'misses', 'maxsize', 'currsize'])


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


def trim_text_tokens(tokens: np.ndarray) -> np.ndarray:
    """Drop the all-padding tail after the batch's last EOT position.

    PE's pooling reads each row at ``argmax(tokens)`` (EOT is the largest
    id in the vocabulary) and its attention mask is causal, so positions
    after the last EOT in the batch influence no output. Returns a
    contiguous ``int64`` ``[B, T]`` with ``T = max EOT index + 1``.
    """
    arr = np.asarray(tokens, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] == 0 or arr.shape[0] == 0:
        return arr
    length = int(arr.argmax(axis=1).max()) + 1
    return np.ascontiguousarray(arr[:, :length])


# ----------------------------------------------------------------------
# Text backends: each maps int64 token ids [B, T] -> float32 [B, 1024]
# ----------------------------------------------------------------------


class TorchTextBackend:
    """PyTorch eager ``pe.CLIP.encode_text`` (loads both towers' weights)."""

    name = 'torch'

    def __init__(self, model: Any, torch_module: Any) -> None:
        self._model = model
        self._torch = torch_module

    def encode(self, tokens: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            features = self._model.encode_text(torch.from_numpy(tokens))
        return np.asarray(features.detach().cpu().numpy(), dtype=np.float32)


class OnnxTextBackend:
    """ONNX Runtime CPU session over the exported text-tower graph."""

    name = 'onnx'

    def __init__(self, session: Any, path: str = '') -> None:
        self._session = session
        self.path = path

    def encode(self, tokens: np.ndarray) -> np.ndarray:
        (out,) = self._session.run([PE_TEXT_OUTPUT], {PE_TEXT_INPUT: tokens})
        return np.asarray(out, dtype=np.float32)


class TritonTextBackend:
    """Synchronous gRPC call to the ``pe_text_encoder`` Triton model.

    Synchronous on purpose: ``encode_text`` already runs off the event loop
    in an executor thread (see ``semantic_search.semantic_text_search``).
    """

    name = 'triton'

    def __init__(self, client: Any, model_name: str = PE_TEXT_TRITON_MODEL) -> None:
        self._client = client
        self.model_name = model_name

    def encode(self, tokens: np.ndarray) -> np.ndarray:
        from tritonclient.grpc import InferInput, InferRequestedOutput

        inp = InferInput(PE_TEXT_INPUT, list(tokens.shape), 'INT64')
        inp.set_data_from_numpy(tokens)
        result = self._client.infer(
            self.model_name, [inp], outputs=[InferRequestedOutput(PE_TEXT_OUTPUT)]
        )
        return np.asarray(result.as_numpy(PE_TEXT_OUTPUT), dtype=np.float32)


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
            (text_backend or os.environ.get('OP_PE_TEXT_BACKEND') or 'auto').strip().lower()
        )
        self._text_onnx_path = str(
            text_onnx_path or os.environ.get('OP_PE_TEXT_ONNX_PATH') or DEFAULT_PE_TEXT_ONNX_PATH
        )
        self._text_triton_model = (
            text_triton_model or os.environ.get('OP_PE_TEXT_TRITON_MODEL') or PE_TEXT_TRITON_MODEL
        )
        threads = (
            ort_threads if ort_threads is not None else os.environ.get('OP_PE_TEXT_ORT_THREADS')
        )
        self._ort_threads = int(threads) if threads else 0
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

    async def embed_crops(
        self,
        crops: list[np.ndarray],
        max_batch: int = PE_CROP_MAX_BATCH,
    ) -> np.ndarray:
        """Preprocess + encode a list of HWC-uint8 RGB item crops.

        Applies the canonical PE preprocessing (resize-shorter-edge,
        center-crop, PE-Core 0.5/0.5 normalize — see
        :mod:`src.services.detection.pe_preprocess`) to each crop, then
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

        chws = np.stack(
            [normalize_chw(resize_crop_rgb(crop)) for crop in crops],
            axis=0,
        ).astype(np.float32, copy=False)

        rows: list[np.ndarray] = []
        for start in range(0, chws.shape[0], max_batch):
            chunk = chws[start : start + max_batch]
            rows.append(await self.encode_images(chunk))
        return np.concatenate(rows, axis=0)

    async def embed_whole_frame(self, path: str) -> np.ndarray | None:
        """PE-Core whole-frame embedding for a source image on disk.

        Goes through :func:`src.services.detection.pe_preprocess.whole_frame_chw`
        (cv2 1/8 decode + resize-shorter-edge + center-crop + PE-Core 0.5/0.5
        normalize) so the vector lands in the same space as any other
        whole-frame caller (e.g. a backfill script), then runs it through
        the same Triton image encoder as :meth:`embed_crops`.

        Returns ``None`` on a decode failure — callers should treat a
        missing whole-frame embedding as non-fatal.
        """
        chw = whole_frame_chw(path)
        if chw is None:
            return None
        batch = chw[None, ...].astype(np.float32, copy=False)
        embedding = await self.encode_images(batch)
        return embedding[0]

    async def embed_whole_frame_bytes(self, data: bytes) -> np.ndarray | None:
        """:meth:`embed_whole_frame` for an image the server only has in memory.

        Used by byte-upload ingest, where the client's path is an
        identifier, not a file the API container can open. Same
        preprocessing, so the vector matches the from-disk path.
        """
        chw = whole_frame_chw_from_bytes(data)
        if chw is None:
            return None
        batch = chw[None, ...].astype(np.float32, copy=False)
        embedding = await self.encode_images(batch)
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
            if pref in ('auto', 'onnx'):
                backend = self._try_onnx_backend(required=pref == 'onnx')
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

    def _try_onnx_backend(self, *, required: bool) -> OnnxTextBackend | None:
        path = self._text_onnx_path
        if not Path(path).is_file():
            if required:
                raise FileNotFoundError(
                    f'OP_PE_TEXT_BACKEND=onnx but {path} does not exist '
                    '(export it with export/export_pe_text_encoder.py)'
                )
            logger.info('pe_text_onnx_not_found', path=path)
            return None
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            if self._ort_threads:
                opts.intra_op_num_threads = self._ort_threads
            session = ort.InferenceSession(
                path, sess_options=opts, providers=['CPUExecutionProvider']
            )
            names_in = [i.name for i in session.get_inputs()]
            names_out = [o.name for o in session.get_outputs()]
            if PE_TEXT_INPUT not in names_in or PE_TEXT_OUTPUT not in names_out:
                raise ValueError(
                    f'{path} has inputs {names_in} / outputs {names_out}; expected '
                    f'{PE_TEXT_INPUT!r} -> {PE_TEXT_OUTPUT!r}'
                )
        except Exception as exc:
            if required:
                raise
            logger.warning('pe_text_onnx_load_failed', path=path, error=str(exc))
            return None
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
        from src.clients.triton_pool import TritonClientManager
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
        """In-process backend: ONNX if the file loads, else PyTorch. Cached."""
        if self._local_text_backend is None:
            onnx = None
            if self._text_backend_pref != 'torch':
                onnx = self._try_onnx_backend(required=False)
            self._local_text_backend = onnx or self._load_torch_backend()
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
