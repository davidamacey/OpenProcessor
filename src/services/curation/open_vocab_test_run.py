"""Run one unsaved open-vocabulary target on one image and report every
candidate with what selection did with it (``POST /open_vocab/test``).

It is the read-only half of the real pass
(:func:`~src.services.curation.open_vocab_run.plan_open_vocab_image`), so a
test shows exactly what a run would keep. Nothing is written: no item, no
class, no status.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, UnidentifiedImageError

from src.services.curation.image_serving import is_servable_image_path
from src.services.curation.open_vocab_run import plan_open_vocab_image
from src.services.detection.open_vocab_set import decode_open_vocab_set


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.curation.open_vocab_run import SegmentImage

#: Largest uploaded image (decoded bytes) a test accepts.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


class TrialImageError(ValueError):
    """The image to test on could not be loaded (maps to a 4xx)."""


@dataclass
class TrialHit:
    bbox_norm: tuple[float, float, float, float]
    score: float
    mask_polygon: tuple[tuple[float, float], ...] | None
    selected: bool
    drop_reason: str | None


@dataclass
class TrialOutcome:
    width: int
    height: int
    hits: list[TrialHit]
    elapsed_ms: float


def decode_upload(image_base64: str) -> Image.Image:
    try:
        data = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise TrialImageError('image_base64 is not valid base64') from exc
    if len(data) > MAX_UPLOAD_BYTES:
        raise TrialImageError(f'the uploaded image is over {MAX_UPLOAD_BYTES // (1024 * 1024)} MB')
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise TrialImageError('the uploaded bytes are not a readable image') from exc
    return img


async def load_stored_image(image_doc: dict[str, Any]) -> Image.Image:
    path = image_doc.get('image_path') or ''
    if not is_servable_image_path(path):
        raise TrialImageError(f'image path is not under a configured source root: {path!r}')

    def read() -> Image.Image:
        img = Image.open(Path(path))
        img.load()
        return img

    try:
        return await asyncio.to_thread(read)
    except (UnidentifiedImageError, OSError) as exc:
        raise TrialImageError('the stored image could not be read') from exc


async def run_open_vocab_test(
    opensearch: AsyncOpenSearch,
    *,
    image_id: str,
    pil: Image.Image,
    target_body: dict[str, Any],
    image_max_side: int,
    dedup_iou: float,
    segment: SegmentImage,
) -> TrialOutcome:
    """Raises ``SegmenterCallError`` when the segmenter cannot answer."""
    ov = decode_open_vocab_set(
        'draft',
        {
            'targets': [{**target_body, 'enabled': True}],
            'image_max_side': image_max_side,
            'dedup_iou': dedup_iou,
        },
    )
    started = time.monotonic()
    plan = await plan_open_vocab_image(opensearch, image_id, pil, ov, ov.targets, segment=segment)
    hits = [
        TrialHit(h.candidate.bbox_norm, h.candidate.score, h.candidate.mask_polygon, True, None)
        for h in plan.selection.kept
    ] + [
        TrialHit(h.candidate.bbox_norm, h.candidate.score, h.candidate.mask_polygon, False, reason)
        for h, reason in plan.selection.dropped
    ]
    hits.sort(key=lambda t: -t.score)
    return TrialOutcome(
        width=pil.width,
        height=pil.height,
        hits=hits,
        elapsed_ms=round((time.monotonic() - started) * 1000, 1),
    )


__all__ = [
    'MAX_UPLOAD_BYTES',
    'TrialHit',
    'TrialImageError',
    'TrialOutcome',
    'decode_upload',
    'load_stored_image',
    'run_open_vocab_test',
]
