"""The text-tower contract of :class:`src.clients.pe_encoder.PEEncoder` and its Triton call.

:class:`TritonTextBackend` maps int64 token ids ``[B, T]`` to float32 ``[B, 1024]`` with a
synchronous gRPC call to Triton's ``pe_text_encoder``; it is the only text backend. tritonclient
is imported inside :meth:`TritonTextBackend.encode`, so this module stays importable without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.clients.triton_deadline import infer_with_deadline
from src.config.settings import TritonModelConfig


PE_TEXT_CHECKPOINT = 'PE-Core-L14-336'

# Text-tower graph contract — shared with export/export_pe_text_encoder.py.
PE_TEXT_TRITON_MODEL = TritonModelConfig.PE_TEXT_MODEL
PE_TEXT_INPUT = 'text_tokens'
PE_TEXT_OUTPUT = 'text_embeddings'
PE_TEXT_CONTEXT_LENGTH = 32


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
        result = infer_with_deadline(
            self._client, self.model_name, [inp], [InferRequestedOutput(PE_TEXT_OUTPUT)]
        )
        return np.asarray(result.as_numpy(PE_TEXT_OUTPUT), dtype=np.float32)
