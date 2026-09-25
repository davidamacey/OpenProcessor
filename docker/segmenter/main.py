"""OpenProcessor segmenter service — promptable segmentation over HTTP.

The detection cascade's segmenter leg (``scripts/curation/worker/client.py``,
``src/services/detection/cascade_detect.py``) calls out to an HTTP service
when the primary detector misses a region or a VLM rejects its candidate.
This is that service: a FastAPI process that loads Meta's Segment Anything
3 once at startup and answers segment requests with candidate bounding
boxes in the submitted image's normalized frame.

**It is prompt-driven and domain-neutral.** ``text_prompt`` is a required
per-request field with no server-side default: what you are segmenting
for is a deployment decision. Nothing here knows about any particular
class, dataset or directory.

Wire contract (the shipped client's expectations, which this server is
built to match):

* ``POST /segment`` — request ``{crop_jpeg_b64, text_prompt,
  max_candidates}``, response ``{candidates: [{bbox_norm, score,
  mask_iou}], ...}``.
* ``POST /segment/batch`` — the same thing for N images in one round trip.
* ``GET /health`` — ``loaded`` flips true once the model pool is up.

Coordinate frame: ``bbox_norm`` is normalized to the **submitted image**.
When that image is a crop of a larger frame, re-projecting to the source
frame is the caller's job (the worker does it via
``crop_norm_to_source_norm``).

See ``NOTICE`` for third-party attribution and ``README.md`` for build,
deployment and tuning notes.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import sam3_backend
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from sam3_backend import Candidate, ProcessorPool


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence


logging.basicConfig(
    level=os.getenv('SEGMENTER_LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s [segmenter] %(levelname)s: %(message)s',
)
log = logging.getLogger('segmenter')


# Process singleton, populated by the lifespan handler. Empty until the
# model pool finishes loading, which is what /health reports on.
_pool = ProcessorPool([])


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the model pool off the event loop, then serve."""
    global _pool  # noqa: PLW0603 — FastAPI lifespan populates module state
    _pool = ProcessorPool(await asyncio.to_thread(sam3_backend.load_processors))
    log.info('segmenter ready: %d processor(s) on %s', len(_pool), sam3_backend.device_name())
    yield


app = FastAPI(
    title='OpenProcessor Segmenter',
    version='1.0.0',
    description='Promptable segmentation (SAM 3) for the detection cascade',
    lifespan=lifespan,
)


# =============================================================================
# Wire models
# =============================================================================


class SegmentRequest(BaseModel):
    """One image plus the prompt describing what to segment in it."""

    crop_jpeg_b64: str = Field(
        ...,
        description='Base64-encoded JPEG bytes. May include the standard'
        ' ``data:image/jpeg;base64,`` prefix (stripped).',
    )
    text_prompt: str = Field(
        ...,
        min_length=1,
        description='What to segment for, e.g. the region type this deployment'
        ' curates. Required: the server has no domain of its own and so has no'
        ' sensible default.',
    )
    max_candidates: int = Field(
        default=4,
        ge=1,
        le=32,
        description='Top-K candidates to return, sorted by score descending.',
    )


class BatchSegmentRequest(BaseModel):
    """N images sharing one prompt, in a single round trip."""

    crops_jpeg_b64: list[str] = Field(
        ...,
        min_length=1,
        max_length=64,
        description='Base64-encoded JPEG bytes per image. May include a data: prefix.',
    )
    text_prompt: str = Field(
        ...,
        min_length=1,
        description='Applied to every image in the batch.',
    )
    max_candidates: int = Field(default=4, ge=1, le=32)


class SegmentCandidate(BaseModel):
    """One segmented instance. ``bbox_norm`` is in the submitted image's frame."""

    bbox_norm: tuple[float, float, float, float]
    score: float
    mask_iou: float | None = Field(
        default=None,
        description='Rectangularity (mask area / mask-bbox area); null when the'
        ' mask head is disabled via SEGMENTER_ENABLE_MASKS=0.',
    )


class SegmentResponse(BaseModel):
    candidates: list[SegmentCandidate]
    elapsed_ms: float
    crop_size: tuple[int, int]
    prompt: str


class BatchSegmentResult(BaseModel):
    """Per-image result. An empty ``candidates`` list means nothing was found."""

    candidates: list[SegmentCandidate]
    crop_size: tuple[int, int]


class BatchSegmentResponse(BaseModel):
    """Aligned 1:1 with the request's ``crops_jpeg_b64`` order."""

    results: list[BatchSegmentResult]
    total_elapsed_ms: float
    prompt: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    loaded: bool
    instances: int


# =============================================================================
# Helpers
# =============================================================================


def _decode_jpeg(b64: str) -> Image.Image:
    """Decode a base64 JPEG string into an RGB :class:`PIL.Image.Image`."""
    if b64.startswith('data:'):
        b64 = b64.split(',', 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert('RGB')


def _decode_all(items: Sequence[str]) -> list[Image.Image]:
    """Decode every image, reporting which index failed."""
    images: list[Image.Image] = []
    for i, b64 in enumerate(items):
        try:
            images.append(_decode_jpeg(b64))
        except Exception as exc:
            detail = f'bad jpeg at idx {i}: {exc}'
            raise HTTPException(status_code=400, detail=detail) from exc
    return images


def _to_wire(candidates: list[Candidate]) -> list[SegmentCandidate]:
    return [
        SegmentCandidate(bbox_norm=c.bbox_norm, score=c.score, mask_iou=c.mask_iou)
        for c in candidates
    ]


def _require_ready() -> None:
    """503 while the model pool is still building.

    Checked before the request body is decoded: an unready service is
    unready regardless of what was sent, and reporting that is more
    useful than a 400 about a payload nobody was going to process.
    """
    if not _pool:
        raise HTTPException(status_code=503, detail='model still loading')


async def _run(images: Sequence[Image.Image], prompt: str, top_k: int) -> list[list[Candidate]]:
    """Acquire a processor and run the batch on it, off the event loop."""
    _require_ready()
    async with _pool.acquire() as processor:
        return await asyncio.to_thread(
            sam3_backend.segment_images, processor, images, prompt, top_k
        )


# =============================================================================
# Routes
# =============================================================================


@app.get('/health', response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + readiness probe. ``loaded`` flips true after lifespan."""
    loaded = bool(_pool)
    return HealthResponse(
        status='healthy' if loaded else 'loading',
        model='sam3',
        device=sam3_backend.device_name(),
        loaded=loaded,
        instances=len(_pool),
    )


@app.post('/segment', response_model=SegmentResponse)
async def segment(req: SegmentRequest) -> SegmentResponse:
    """Segment ``text_prompt`` instances in one image. Returns top-K boxes."""
    _require_ready()
    images = _decode_all([req.crop_jpeg_b64])
    t0 = time.perf_counter()
    per_image = await _run(images, req.text_prompt, req.max_candidates)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return SegmentResponse(
        candidates=_to_wire(per_image[0]),
        elapsed_ms=elapsed,
        crop_size=images[0].size,
        prompt=req.text_prompt,
    )


@app.post('/segment/batch', response_model=BatchSegmentResponse)
async def segment_batch(req: BatchSegmentRequest) -> BatchSegmentResponse:
    """Segment N images under a single processor lock.

    The caller chunks: 4-16 images per request is the useful range.
    Larger batches hold one processor longer and cut overall concurrency;
    smaller ones don't amortize the per-call overhead enough to matter.
    """
    _require_ready()
    images = _decode_all(req.crops_jpeg_b64)
    t0 = time.perf_counter()
    per_image = await _run(images, req.text_prompt, req.max_candidates)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return BatchSegmentResponse(
        results=[
            BatchSegmentResult(candidates=_to_wire(cands), crop_size=img.size)
            for cands, img in zip(per_image, images, strict=True)
        ],
        total_elapsed_ms=elapsed,
        prompt=req.text_prompt,
    )


if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host='0.0.0.0',  # nosec B104 — container-internal; compose maps the port
        port=int(os.getenv('SEGMENTER_LISTEN_PORT', '8000')),
        log_level=os.getenv('SEGMENTER_LOG_LEVEL', 'info').lower(),
    )
