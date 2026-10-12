"""GPU ingest front end for photos (#217): decode once on the GPU, derive every input there.

With ``OP_INGEST_GPU_FRONTEND`` on, a batch ingest sends each image at or above
``OP_INGEST_GPU_MIN_MEGAPIXELS`` to the ``image_frontend`` Triton model instead of decoding it in
the API. That model decodes the photo once (nvJPEG), runs the detector on its letterbox, embeds
the whole-frame square and every detection's square with PE, computes the blur sums and encodes
the crop JPEGs; only boxes, vectors, sums and JPEGs come back. The API turns the detector rows
into items with the same :meth:`WholeImageDetector.decode_primary_row` the CPU path uses and
reads every other per-image product from the returned :class:`FrontendFrame`.

Fail closed, image by image: an image the front end does not take (not a baseline 8-bit JPEG,
no end-of-image marker, below the gate, a decoder error) or a front-end call that fails is
decoded on the CPU path as before, and counted in ``op_ingest_decode_total``. The front end is
used only while its detector, network size and letterbox fill match the ingest profile and no
secondary detector is configured (that one needs the CPU frame); otherwise every image takes the
CPU path.

Images below the gate stay on the CPU because they are faster there: a COCO frame decodes in
5 ms on one core, and the GPU it would use is what binds frame ingest (PE).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from src.clients.pe_encoder import _l2_normalize
from src.config.settings import TritonModelConfig
from src.core.logging import get_logger
from src.services.curation.ops_metrics import OP_INGEST_DECODE_TOTAL
from src.services.detection.crop_quality import variance_from_sums


if TYPE_CHECKING:
    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.item_doc import DetectedItem


logger = get_logger(__name__)

FRONTEND_MODEL = TritonModelConfig.IMAGE_FRONTEND_MODEL


@dataclass
class FrontendFrame:
    """One photo's products from the front end; per-detection rows are indexed by the item
    objects :func:`run_frontend` built (``det_of``)."""

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
        return self.crop_embeddings[self._row(item)]

    def crop_jpeg(self, item: DetectedItem) -> bytes:
        return self.crop_jpegs[self._row(item)]


def megapixels(data: bytes) -> float | None:
    """Pixel count in millions from the header (no decode); ``None`` when unreadable."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            w, h = img.size
    except Exception:
        return None
    return w * h / 1e6


_READY: dict[tuple[str, int, tuple[int, int, int]], bool] = {}


async def frontend_ready(service: CurationIngestService) -> bool:
    """Whether photos go to the front end: enabled, no secondary detector, and the loaded model
    built for the ingest profile's detector input. The model check runs once per profile
    shape per process; a mismatch or a missing model is logged and keeps the CPU path."""
    if not service.config.ingest_gpu_frontend or service.detector.secondary_profile is not None:
        return False
    profile = service.profile
    key = (profile.detector_model, profile.input_size, profile.letterbox_fill)
    cached = _READY.get(key)
    if cached is not None:
        return cached
    ready = False
    try:
        if not await service.triton_pool.is_model_ready(FRONTEND_MODEL):
            # Not in Triton's default load list: it holds GPU memory only where it is used.
            await service.triton_pool.load_model(FRONTEND_MODEL)
        params = await service.triton_pool.get_model_parameters(FRONTEND_MODEL)
        expected = {
            'DETECTOR_MODEL': profile.detector_model,
            'NET_SIZE': str(profile.input_size),
            'LETTERBOX_FILL': str(profile.letterbox_fill[0]),
        }
        mismatch = {k: (params.get(k), v) for k, v in expected.items() if params.get(k) != v}
        ready = not mismatch and len(set(profile.letterbox_fill)) == 1
        if not ready:
            logger.warning(
                'ingest_frontend_profile_mismatch', mismatch=mismatch, fill=profile.letterbox_fill
            )
    except Exception as exc:
        logger.warning('ingest_frontend_unavailable', error=str(exc))
    _READY[key] = ready
    return ready


def _input(name: str, arr: np.ndarray, dtype: str) -> Any:
    from tritonclient.grpc import InferInput

    inp = InferInput(name, list(arr.shape), dtype)
    inp.set_data_from_numpy(arr)
    return inp


async def run_frontend(
    service: CurationIngestService, data: bytes, *, embed_crops: bool
) -> tuple[FrontendFrame, list[DetectedItem]] | str:
    """The front end's products and the detected items for one image, or the reason the image
    takes the CPU path."""
    inputs = [
        _input('jpeg', np.frombuffer(data, dtype=np.uint8), 'UINT8'),
        _input('embed_crops', np.array([embed_crops]), 'BOOL'),
    ]
    try:
        result = await service.triton_pool.stream_one(FRONTEND_MODEL, inputs)
    except Exception as exc:
        logger.warning('ingest_frontend_call_failed', error=str(exc))
        return 'call_failed'
    status = result.as_numpy('status')[0]
    status = status.decode() if isinstance(status, bytes) else str(status)
    if status != 'ok':
        return status
    width, height = (int(v) for v in result.as_numpy('size'))
    scale, pad_w, pad_h = (float(v) for v in result.as_numpy('letterbox'))
    boxes = result.as_numpy('det_boxes').reshape(-1, 4)
    n = boxes.shape[0]
    items = service.detector.decode_primary_row(
        np.array([n]),
        boxes,
        result.as_numpy('det_scores'),
        result.as_numpy('det_classes'),
        scale,
        (pad_w, pad_h),
        service.profile.input_size,
    )
    crops = result.as_numpy('crop_embeddings').reshape(
        -1, result.as_numpy('whole_embedding').shape[0]
    )
    frame = FrontendFrame(
        width=width,
        height=height,
        whole_embedding=_l2_normalize(result.as_numpy('whole_embedding').astype(np.float32)),
        crop_embeddings=_l2_normalize(crops.astype(np.float32)) if embed_crops and n else None,
        blur_sums=result.as_numpy('blur_sums').reshape(-1, 3),
        crop_jpegs=[bytes(b) for b in result.as_numpy('crop_jpegs').reshape(-1)],
        det_of={id(item): i for i, item in enumerate(items)},
    )
    return frame, items


def count_route(route: str) -> None:
    OP_INGEST_DECODE_TOTAL.labels(route=route).inc()
