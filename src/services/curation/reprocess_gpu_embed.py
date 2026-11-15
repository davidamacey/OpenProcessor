"""The embed pass on the GPU decode (#259): decode a stored photo once on the GPU and embed its
stored boxes there.

The ``embed`` reprocess scope decoded every photo in the API, cut the stored boxes and resized
them on the CPU: a 21 MP photo costs about 0.5 CPU-s there while the GPU sits 10 to 16 % busy.
With the ingest engine stage usable (:func:`~src.services.curation.ingest_gpu_decode.engine_stage_applies`
and :func:`~src.services.curation.ingest_gpu_decode.engine_ready`), every photo is sent to the frame
engine (which decodes any image) with its stored ``bbox_norm`` boxes (the ``boxes`` input of protocol version 1, a plan of
``embed_frame`` and ``embed_crops``): the engine decodes it once (nvJPEG, EXIF orientation), cuts
each box by the rectangle the CPU pass cuts and returns the whole-frame vector and one vector per
box. Only vectors come back.

Fail closed, photo by photo: a status other than ``ok`` (corrupt, decoder, ...), an engine still
overloaded after its deadline, a failed call or an answer with the wrong row count take the CPU
path, counted in ``op_decode_route_total``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.clients.pe_encoder import _l2_normalize
from src.domain import frame_engine_wire as wire
from src.domain.decode_route import CPU_BOX_COUNT, GPU, prep_status_route
from src.services.curation.ingest_gpu_decode import (
    call_frame_engine,
    crop_embed_spec,
    embed_spec,
    engine_ready,
    engine_stage_applies,
)
from src.services.curation.ops_metrics import count_decode_route


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.services.curation.ingest import CurationIngestService


@dataclass(frozen=True)
class GpuEmbedRoute:
    """Where the embed pass uses the GPU decode."""

    frame_engine: Any


@dataclass(frozen=True)
class StoredBoxVectors:
    """L2-normalized vectors of one photo: the whole frame and one row per box sent, in order."""

    frame: np.ndarray
    boxes: np.ndarray


async def gpu_embed_route(service: CurationIngestService) -> GpuEmbedRoute | None:
    """The route when photos of the embed pass go to the GPU decode, else ``None`` (every photo
    then takes the CPU path): the ingest readiness rule."""
    if not engine_stage_applies(service) or not await engine_ready(service):
        return None
    return GpuEmbedRoute(service.frame_engine)


def _count(route: str) -> None:
    count_decode_route(route, 'reprocess')


async def embed_on_gpu(
    route: GpuEmbedRoute, data: bytes, boxes: Sequence[Sequence[float]]
) -> StoredBoxVectors | None:
    """The vectors of the encoded photo ``data`` and its ``boxes`` (stored ``bbox_norm``), or
    ``None`` when this photo takes the CPU path."""
    rows = np.asarray(boxes, np.float64).reshape(-1, 4)
    plan = wire.EnginePlan(
        branches=('embed_frame', 'embed_crops'), embed=embed_spec(), crop_embed=crop_embed_spec()
    )
    answer = await call_frame_engine(route.frame_engine, data, plan, rows, caller='reprocess_embed')
    if isinstance(answer, str):
        _count(prep_status_route(answer))
        return None
    frame = answer.outputs[wire.WHOLE_EMBEDDING].astype(np.float32)
    vectors = answer.outputs[wire.CROP_EMBEDDINGS].astype(np.float32)
    if (
        vectors.ndim != 2 or vectors.shape[0] != rows.shape[0]
    ):  # one row per box, the crop encoder's width
        _count(CPU_BOX_COUNT)
        return None
    _count(GPU)
    return StoredBoxVectors(frame=_l2_normalize(frame), boxes=_l2_normalize(vectors))


__all__ = [
    'GpuEmbedRoute',
    'StoredBoxVectors',
    'embed_on_gpu',
    'gpu_embed_route',
]
