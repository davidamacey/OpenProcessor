"""GPU decode for photos (#217): decode once on the GPU, derive every input there.

With ``OP_INGEST_GPU_DECODE`` on, a batch ingest sends each image at or above
``OP_INGEST_GPU_MIN_MEGAPIXELS`` to the frame engine (``frame_engine/``, the ``frame-engine``
service) instead of decoding it in the API. The engine decodes the photo once (nvJPEG), runs the
detector on its letterbox, embeds the whole-frame square and every detection's square with PE,
computes the blur sums and encodes the crop JPEGs; only boxes, vectors, sums and JPEGs come back
(:mod:`src.clients.frame_engine`, one request per image with a plan naming the profile's detector
and the PE encoder). The API turns the detector rows into items with the same
:meth:`WholeImageDetector.decode_primary_row` the CPU path uses and reads every other per-image
product from the returned :class:`GpuDecodeFrame`.

Fail closed, image by image: an image the GPU decode does not take (not a baseline 8-bit JPEG,
no end-of-image marker, below the gate, a decoder error), an engine still overloaded after the
client's retries (``gpu_memory``), and a call that fails or a branch that errors (``call_failed``)
are decoded on the CPU path as before, and counted in ``op_decode_route_total``; each photo's
route (:mod:`src.domain.decode_route`) is stored on its images document as ``decode_route``. The
GPU decode is used only while no secondary detector is configured (that one needs the CPU frame)
and the profile's letterbox fill is one grey; otherwise every image takes the CPU path.

Images below the gate stay on the CPU because they are faster there: a COCO frame decodes in
5 ms on one core, and the GPU it would use is what binds frame ingest (PE).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from src.clients import frame_engine
from src.clients.pe_encoder import PE_IMAGE_MODEL, _l2_normalize
from src.core.logging import get_logger
from src.domain import frame_engine_wire as wire
from src.services.curation.ops_metrics import count_decode_route
from src.services.detection.crop_quality import variance_from_sums
from src.utils.pe_preprocess import PE_SIZE
from src.utils.retry import TRITON_CALL_ERRORS


if TYPE_CHECKING:
    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.item_doc import DetectedItem


logger = get_logger(__name__)

#: Squares per encoder call inside the engine (the PE engine's max batch, as the prep model used).
PE_BATCH = 32
#: The answer when the engine stayed overloaded through the client's retries: the status the prep
#: model gave when its device-memory wait ran out.
OVERLOADED = 'gpu_memory'
CALL_FAILED = 'call_failed'


@dataclass
class GpuDecodeFrame:
    """One photo's products from the GPU decode; per-detection rows are indexed by the item
    objects :func:`run_gpu_decode` built (``det_of``)."""

    width: int
    height: int
    whole_embedding: np.ndarray
    crop_embeddings: np.ndarray | None
    blur_sums: np.ndarray
    crop_jpegs: list[bytes]
    det_of: dict[int, int] = field(default_factory=dict)

    def _row(self, item: DetectedItem) -> int:
        return self.det_of[id(item)]

    def full_var(self) -> float:
        n, total, squares = (int(v) for v in self.blur_sums[0])
        return variance_from_sums(n, total, squares)

    def box_var(self, item: DetectedItem) -> float | None:
        n, total, squares = (int(v) for v in self.blur_sums[1 + self._row(item)])
        return variance_from_sums(n, total, squares) if n > 0 else None

    def crop_embedding(self, item: DetectedItem) -> np.ndarray | None:
        if self.crop_embeddings is None:
            return None
        typed: np.ndarray | None = self.crop_embeddings[self._row(item)]
        return typed

    def crop_jpeg(self, item: DetectedItem) -> bytes:
        return self.crop_jpegs[self._row(item)]


def megapixels(data: bytes) -> float | None:
    """Pixel count in millions from the header (no decode); ``None`` when unreadable."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            w, h = img.size
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None
    return w * h / 1e6


#: What a readiness probe can raise besides the Triton client errors: a pool not initialized.
_PROBE_ERRORS: tuple[type[Exception], ...] = (*TRITON_CALL_ERRORS, RuntimeError)
_READY: set[tuple[str, int, int]] = set()


def embed_spec() -> wire.EmbedSpec:
    return wire.EmbedSpec(model=PE_IMAGE_MODEL, size=PE_SIZE, batch=PE_BATCH)


def detector_spec(service: CurationIngestService) -> wire.DetectorSpec | None:
    """The profile's detector as the engine runs it; ``None`` when the letterbox fill is not one
    grey (the engine fills the canvas with one value)."""
    p = service.profile
    if len(set(p.letterbox_fill)) != 1:
        return None
    return wire.DetectorSpec(p.detector_model, p.input_size, p.letterbox_fill[0])


async def gpu_decode_ready(service: CurationIngestService) -> bool:
    """Whether photos go to the GPU decode: enabled, an engine client, no secondary detector, a
    one-grey letterbox fill, the engine ready and the detector and PE encoder loaded in Triton
    (explicit model control: they are loaded here when they are not). A ready verdict is kept per
    detector shape per process; a failed probe is not, so the next batch asks again."""
    if (
        not service.config.ingest_gpu_decode
        or service.frame_engine is None
        or service.detector.secondary_profile is not None
    ):
        return False
    det = detector_spec(service)
    if det is None:
        logger.warning('ingest_gpu_decode_fill_not_uniform', fill=service.profile.letterbox_fill)
        return False
    key = (det.model, det.input_size, det.letterbox_fill)
    if key in _READY:
        return True
    try:
        if not await service.frame_engine.is_model_ready(wire.MODEL_NAME):
            logger.warning('ingest_gpu_decode_engine_not_ready')
            return False
        for model in (det.model, PE_IMAGE_MODEL):
            if not await service.triton_pool.is_model_ready(model):
                await service.triton_pool.load_model(model)
    except _PROBE_ERRORS as exc:
        # A failed call is not a verdict: do not pin the worker to the CPU path.
        logger.warning('ingest_gpu_decode_unavailable', error=str(exc))
        return False
    _READY.add(key)
    return True


@dataclass(frozen=True)
class GpuRoute:
    """Which images of a batch the GPU decode takes (``chosen``); ``unreadable`` are those whose
    header cannot be read, the rest of the others are below the size gate. ``ready`` is false
    when the GPU decode is off or unusable: every image then takes the CPU path."""

    ready: bool
    chosen: tuple[int, ...] = ()
    unreadable: tuple[int, ...] = ()


async def gpu_route(
    service: CurationIngestService, images: list[bytes], indices: list[int]
) -> GpuRoute:
    """Route ``indices`` of ``images`` before any of them is admitted or decoded, so the
    admission gate can charge each for what it will hold in the API process."""
    if not indices or not await gpu_decode_ready(service):
        return GpuRoute(ready=False)
    gate = service.config.ingest_gpu_min_megapixels
    chosen: list[int] = []
    unreadable: list[int] = []
    for idx in indices:
        mp = megapixels(images[idx])
        if mp is None:
            unreadable.append(idx)
        elif mp >= gate:
            chosen.append(idx)
    return GpuRoute(ready=True, chosen=tuple(chosen), unreadable=tuple(unreadable))


async def call_frame_engine(
    pool: Any, data: bytes, plan: wire.EnginePlan, boxes: np.ndarray | None = None, *, caller: str
) -> frame_engine.EngineAnswer | str:
    """One frame-engine request for the encoded image ``data``: the answer when the image was
    decoded and every branch ran, else the reason the image takes the CPU path as a string (a
    status from :data:`src.domain.decode_route.PREP_STATUSES`, :data:`OVERLOADED` when the engine
    stayed full through the retries, :data:`CALL_FAILED` when the call failed or a branch
    errored)."""
    try:
        answer = await frame_engine.process(pool, data, plan, boxes)
    except frame_engine.EngineCallError as exc:
        logger.warning(
            'frame_engine_call_failed', caller=caller, overloaded=exc.overloaded, error=str(exc)
        )
        return OVERLOADED if exc.overloaded else CALL_FAILED
    if answer.status != wire.STATUS_OK:
        return answer.status
    if not answer.ok:
        logger.warning('frame_engine_branch_failed', caller=caller, branches=answer.branches)
        return CALL_FAILED
    return answer


async def run_gpu_decode(
    service: CurationIngestService, data: bytes, *, embed_crops: bool
) -> tuple[GpuDecodeFrame, list[DetectedItem]] | str:
    """The GPU decode's products and the detected items for one image, or the reason the image
    takes the CPU path."""
    det = detector_spec(service)
    if det is None:
        raise RuntimeError('run_gpu_decode without a ready GPU decode (gpu_decode_ready)')
    branches: list[wire.EngineBranch] = ['detect', 'embed_frame']
    if embed_crops:
        branches.append('embed_crops')
    plan = wire.EnginePlan(
        branches=tuple(branches), detector=det, embed=embed_spec(), crops='inline'
    )
    answer = await call_frame_engine(service.frame_engine, data, plan, caller='ingest')
    if isinstance(answer, str):
        return answer
    out = answer.outputs
    width, height = (int(v) for v in out[wire.SIZE])
    scale, pad_w, pad_h = (float(v) for v in out[wire.LETTERBOX])
    boxes = out[wire.DET_BOXES].reshape(-1, 4)
    n = boxes.shape[0]
    items = service.detector.decode_primary_row(
        boxes, out[wire.DET_SCORES], out[wire.DET_CLASSES], scale, (pad_w, pad_h)
    )
    whole = out[wire.WHOLE_EMBEDDING].astype(np.float32)
    crops = out[wire.CROP_EMBEDDINGS].reshape(-1, whole.shape[0]) if embed_crops else None
    frame = GpuDecodeFrame(
        width=width,
        height=height,
        whole_embedding=_l2_normalize(whole),
        crop_embeddings=_l2_normalize(crops.astype(np.float32))
        if crops is not None and n
        else None,
        blur_sums=out[wire.BLUR_SUMS].reshape(-1, 3),
        crop_jpegs=list(out[wire.CROP_JPEGS]),
        det_of={id(item): i for i, item in enumerate(items)},
    )
    return frame, items


def count_route(route: str) -> None:
    count_decode_route(route, 'ingest')
