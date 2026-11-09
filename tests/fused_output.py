"""The fused detector tensor, built the way the engine emits it.

A served detector answers with one ``[batch, 300, 6]`` tensor of rows
``x1, y1, x2, y2, score, class`` in letterboxed network-input pixels, zero-padded past the last
detection. Tests script detections as fractions of the network input (so a case does not depend on
the input size) and this helper scales them to pixels.
"""

from __future__ import annotations

import numpy as np
from tritonclient.utils import InferenceServerException

from src.domain.model_families import MAX_DET


Detection = tuple[float, float, float, float, float, int]


def fused_tensor(detections: list[Detection], *, net: int, batch: int = 1) -> np.ndarray:
    """``[batch, MAX_DET, 6]`` float32: every image gets ``detections`` (box fractions times
    ``net``), the rest of each image's rows are zero padding."""
    rows = np.zeros((batch, MAX_DET, 6), dtype=np.float32)
    for k, (x1, y1, x2, y2, score, cls) in enumerate(detections):
        rows[:, k] = (x1 * net, y1 * net, x2 * net, y2 * net, score, cls)
    return rows


class GpuDecodeAbsent:
    """A Triton pool whose repository has no ``gpu_image_prep`` model, as a server that does not
    load it: ingest reads the failed load as "GPU decode unavailable" and decodes on the CPU."""

    async def is_model_ready(self, _model_name: str) -> bool:
        return False

    async def load_model(self, _model_name: str) -> None:
        raise InferenceServerException('unknown model: not in the repository')
