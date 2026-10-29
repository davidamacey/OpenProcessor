"""The embed pass on the GPU decode (#259): decode a stored photo once on the GPU and embed its
stored boxes there.

The ``embed`` reprocess scope decoded every photo in the API, cut the stored boxes and resized
them on the CPU: a 21 MP photo costs about 0.5 CPU-s there while the GPU sits 10 to 16 % busy.
With the ingest GPU decode usable (:func:`~src.services.curation.ingest_gpu_decode.gpu_decode_ready`),
a photo at or above the ingest size gate (``OP_INGEST_GPU_MIN_MEGAPIXELS``) is sent to
``gpu_image_prep`` with its stored ``bbox_norm`` boxes (stored-box mode, ``models/gpu_image_prep``):
the model decodes it once (nvJPEG, EXIF orientation), cuts each box by the rectangle the CPU pass
cuts and returns the whole-frame vector and one vector per box. Only vectors come back.

Fail closed, photo by photo, exactly where ingest falls back: below the gate, an unreadable
header, a status other than ``ok`` (progressive, truncated, decoder, ...), a failed call or an
answer with the wrong row count take the CPU path, counted in ``op_reprocess_decode_total``.
Model files older than this API (no ``boxes`` input) leave every photo on the CPU path without a
call per photo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from tritonclient.utils import InferenceServerException

from src.clients.pe_encoder import _l2_normalize
from src.core.logging import get_logger
from src.services.curation.ingest_gpu_decode import (
    GPU_IMAGE_PREP_MODEL,
    call_gpu_image_prep,
    gpu_decode_ready,
    megapixels,
)
from src.services.curation.ops_metrics import OP_REPROCESS_DECODE_TOTAL


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.services.curation.ingest import CurationIngestService


logger = get_logger(__name__)

BOXES_INPUT = 'boxes'


@dataclass(frozen=True)
class GpuEmbedRoute:
    """Where and from what size the embed pass uses the GPU decode."""

    triton_pool: Any
    min_megapixels: float


@dataclass(frozen=True)
class StoredBoxVectors:
    """L2-normalized vectors of one photo: the whole frame and one row per box sent, in order."""

    frame: np.ndarray
    boxes: np.ndarray


async def gpu_embed_route(service: CurationIngestService) -> GpuEmbedRoute | None:
    """The route when photos of the embed pass go to the GPU decode, else ``None`` (every photo
    then takes the CPU path): the ingest readiness rule, plus model files that take stored boxes."""
    if not await gpu_decode_ready(service):
        return None
    try:
        inputs = await service.triton_pool.get_model_input_names(GPU_IMAGE_PREP_MODEL)
    except (InferenceServerException, OSError, TimeoutError) as exc:
        logger.warning('reprocess_gpu_embed_unavailable', error=str(exc))
        return None
    if BOXES_INPUT not in inputs:
        logger.warning('reprocess_gpu_embed_model_outdated', inputs=inputs)
        return None
    return GpuEmbedRoute(service.triton_pool, service.config.ingest_gpu_min_megapixels)


def _count(route: str) -> None:
    OP_REPROCESS_DECODE_TOTAL.labels(route=route).inc()


async def embed_on_gpu(
    route: GpuEmbedRoute, data: bytes, boxes: Sequence[Sequence[float]]
) -> StoredBoxVectors | None:
    """The vectors of the encoded photo ``data`` and its ``boxes`` (stored ``bbox_norm``), or
    ``None`` when this photo takes the CPU path."""
    mp = megapixels(data)
    if mp is None or mp < route.min_megapixels:
        _count('cpu_not_image' if mp is None else 'cpu_small')
        return None
    rows = np.asarray(boxes, np.float64).reshape(-1, 4)
    result = await call_gpu_image_prep(
        route.triton_pool, data, [(BOXES_INPUT, rows, 'FP64')], caller='reprocess_embed'
    )
    if isinstance(result, str):
        _count(f'cpu_{result}')
        return None
    frame = result.as_numpy('whole_embedding').astype(np.float32)
    vectors = result.as_numpy('crop_embeddings').astype(np.float32).reshape(-1, frame.shape[0])
    if vectors.shape[0] != rows.shape[0]:
        _count('cpu_box_count')
        return None
    _count('gpu')
    return StoredBoxVectors(frame=_l2_normalize(frame), boxes=_l2_normalize(vectors))


__all__ = [
    'BOXES_INPUT',
    'GpuEmbedRoute',
    'StoredBoxVectors',
    'embed_on_gpu',
    'gpu_embed_route',
]
