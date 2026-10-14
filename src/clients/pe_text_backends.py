"""Text-tower backends for :class:`src.clients.pe_encoder.PEEncoder`.

Each backend maps int64 token ids ``[B, T]`` to float32 ``[B, 1024]``:
PyTorch eager (``torch``), ONNX Runtime on CPU (``onnx``) and a synchronous
gRPC call to Triton's ``pe_text_encoder`` (``triton``). The torch and onnx
backends have lazy wrappers that defer the multi-GB load to the first query.

The heavy imports (torch, perception_models, onnxruntime) live in the loaders
:class:`PEEncoder` passes in, and tritonclient is imported inside
:meth:`TritonTextBackend.encode`, so this module stays importable without them.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from src.config.settings import TritonModelConfig
from src.core.logging import get_logger


logger = get_logger(__name__)


PE_TEXT_CHECKPOINT = 'PE-Core-L14-336'

# Text-tower graph contract — shared with export/export_pe_text_encoder.py.
PE_TEXT_TRITON_MODEL = TritonModelConfig.PE_TEXT_MODEL
PE_TEXT_INPUT = 'text_tokens'
PE_TEXT_OUTPUT = 'text_embeddings'
PE_TEXT_CONTEXT_LENGTH = 32
DEFAULT_PE_TEXT_ONNX_PATH = '/app/pytorch_models/pe_text_encoder.onnx'
PE_TEXT_BACKENDS = ('auto', 'onnx', 'triton', 'torch')


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


class LazyTorchTextBackend:
    """Defers the ~2.7 GB ``pe.CLIP`` eager load until the first real query.

    ``PEEncoder.warm_text_encoder()`` runs from the FastAPI lifespan in
    *every* uvicorn worker at startup. When ``OP_PE_TEXT_BACKEND=auto`` (the
    default) and neither the ONNX export nor Triton's ``pe_text_encoder`` is
    available yet, eagerly loading the torch fallback there multiplies a
    multi-GB checkpoint load by the worker count before a single
    ``/curation/search/text`` request has ever been made — on a fresh
    install (before ``make export-pe`` has produced the ONNX file) this was
    observed to peak container RSS at 153 GiB with 32 workers and OOM-kill
    the container before any export command could even run.

    Reports ``name == 'torch'`` immediately (health/status endpoints see
    the truth about which backend *will* serve text queries), but only
    calls the real loader — once, thread-safely — from the first
    :meth:`encode`, so only workers that actually receive a text-search
    request ever pay the load cost, and they pay it spread out over time
    rather than all at once at startup.
    """

    name = 'torch'

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._real: Any = None
        self._lock = threading.Lock()

    def encode(self, tokens: np.ndarray) -> np.ndarray:
        real = self._real
        if real is None:
            with self._lock:
                real = self._real
                if real is None:
                    logger.info(
                        'pe_text_encoder_lazy_loading',
                        checkpoint=PE_TEXT_CHECKPOINT,
                        backend='torch',
                    )
                    real = self._loader()
                    self._real = real
        return real.encode(tokens)


class OnnxTextBackend:
    """ONNX Runtime CPU session over the exported text-tower graph."""

    name = 'onnx'

    def __init__(self, session: Any, path: str = '') -> None:
        self._session = session
        self.path = path

    def encode(self, tokens: np.ndarray) -> np.ndarray:
        (out,) = self._session.run([PE_TEXT_OUTPUT], {PE_TEXT_INPUT: tokens})
        return np.asarray(out, dtype=np.float32)


class LazyOnnxTextBackend:
    """Defers ``ort.InferenceSession`` construction until the first query.

    Even as an
    explicit ``OP_PE_TEXT_BACKEND=onnx`` opt-in, this must not load at
    warm time -- ``warm_text_encoder()`` runs in every uvicorn worker at
    startup, and only workers that actually serve a text-search request
    should ever pay the ~1.4 GB session cost. Reports ``name == 'onnx'``
    immediately; the loader (which raises if the file is missing/invalid)
    only actually runs from the first :meth:`encode`.
    """

    name = 'onnx'

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._real: OnnxTextBackend | None = None
        self._lock = threading.Lock()

    def encode(self, tokens: np.ndarray) -> np.ndarray:
        real = self._real
        if real is None:
            with self._lock:
                real = self._real
                if real is None:
                    logger.info('pe_text_encoder_lazy_loading', backend='onnx')
                    real = self._loader()
                    self._real = real
        return real.encode(tokens)


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
