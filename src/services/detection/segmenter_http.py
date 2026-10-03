"""One-shot segmenter call for the API process (W5, ``POST /region_profiles/test``).

The detection worker talks to the segmenter through
``scripts/curation/worker/client.py``, which carries a per-host circuit
breaker, retry budget and metrics; none of that belongs in an author's
"try this prompt" call, and ``src`` must not import from ``scripts``. This is
the plain ``POST /segment``: it returns EVERY candidate (the worker keeps the
selection step for itself), optionally with mask polygons, and reports an
outage as :class:`SegmenterCallError` so it can never be mistaken for "the
prompt found nothing".
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

import httpx

from src.services.detection.cascade_detect import RegionCandidate


#: How many candidates the worker asks the segmenter for per crop (its client
#: default); a test run asks for the same, so it shows what the worker sees.
DEFAULT_MAX_CANDIDATES = 4

#: Wall-clock bound for one test call (the segmenter's own cold start is
#: reported by ``GET /health``, not waited out here).
DEFAULT_TIMEOUT_S = 30.0


class SegmenterCallError(RuntimeError):
    """The segmenter could not answer (unreachable, non-2xx, unreadable reply)."""


@dataclass(frozen=True)
class SegmentCandidateData:
    """One raw segmenter candidate, in the submitted crop's normalized frame."""

    bbox_norm: tuple[float, float, float, float]
    score: float
    mask_iou: float | None
    mask_polygon: list[tuple[float, float]] | None


def first_segmenter_url() -> str | None:
    """The first URL of ``OP_SEGMENTER_URL`` (it may list several), or
    ``None`` when no segmenter is configured."""
    raw = os.environ.get('OP_SEGMENTER_URL', '').strip()
    first = raw.split(',')[0].strip().rstrip('/')
    return first or None


def _candidate(raw: Any) -> SegmentCandidateData | None:
    if not isinstance(raw, dict):
        return None
    bbox = raw.get('bbox_norm')
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return None
    polygon = raw.get('mask_polygon')
    return SegmentCandidateData(
        bbox_norm=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
        score=float(raw.get('score') or 0.0),
        mask_iou=float(raw['mask_iou']) if raw.get('mask_iou') is not None else None,
        mask_polygon=[(float(x), float(y)) for x, y in polygon] if polygon else None,
    )


async def segment_once(
    url: str,
    jpeg: bytes,
    prompt: str,
    *,
    max_candidates: int,
    return_masks: bool,
    min_score: float | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    client: httpx.AsyncClient | None = None,
) -> list[SegmentCandidateData]:
    """``POST {url}/segment`` for one crop; every candidate, score order as served.

    Raises :class:`SegmenterCallError` on any transport failure, non-2xx
    status or unreadable body. A redirect is never followed.
    """
    payload = {
        'crop_jpeg_b64': base64.b64encode(jpeg).decode('ascii'),
        'text_prompt': prompt,
        'max_candidates': max_candidates,
        'return_masks': return_masks,
    }
    if min_score is not None:
        payload['min_score'] = min_score
    owned = client is None
    http = client or httpx.AsyncClient(timeout=timeout_s, follow_redirects=False)
    try:
        resp = await http.post(f'{url.rstrip("/")}/segment', json=payload, timeout=timeout_s)
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        msg = f'segmenter call failed: {type(exc).__name__}: {exc}'
        raise SegmenterCallError(msg) from exc
    finally:
        if owned:
            await http.aclose()
    if not isinstance(body, dict):
        msg = 'segmenter call failed: the reply is not a JSON object'
        raise SegmenterCallError(msg)
    try:
        parsed = [_candidate(c) for c in body.get('candidates') or []]
    except (TypeError, ValueError) as exc:
        msg = f'segmenter call failed: unreadable candidate: {type(exc).__name__}'
        raise SegmenterCallError(msg) from exc
    return [c for c in parsed if c is not None]


async def segmenter_instances(url: str, *, timeout_s: float = 3.0) -> int | None:
    """``instances`` from the segmenter's ``GET /health`` (its in-flight
    forward capacity), or ``None`` when it cannot be read. Only used to
    estimate a pass's duration, never to decide anything."""
    try:
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=False) as http:
            resp = await http.get(f'{url.rstrip("/")}/health')
            resp.raise_for_status()
            value = resp.json().get('instances')
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    return int(value) if isinstance(value, int) and value > 0 else None


async def segment_image_http(
    jpeg: bytes,
    prompt: str,
    *,
    min_score: float | None,
    max_candidates: int,
    return_masks: bool,
) -> list[RegionCandidate]:
    """One whole-image call to the first configured segmenter, as
    :class:`RegionCandidate` (``bbox_norm`` / ``mask_polygon`` in the
    submitted image's frame). Raises :class:`SegmenterCallError` when no
    segmenter is configured or it cannot answer: an outage is never "no hit"."""
    url = first_segmenter_url()
    if url is None:
        msg = 'segmenter call failed: OP_SEGMENTER_URL is not configured'
        raise SegmenterCallError(msg)
    found = await segment_once(
        url,
        jpeg,
        prompt,
        max_candidates=max_candidates,
        return_masks=return_masks,
        min_score=min_score,
    )
    return [
        RegionCandidate(
            bbox_norm=c.bbox_norm,
            score=c.score,
            source='sam3',
            rectangularity=c.mask_iou,
            mask_polygon=tuple(c.mask_polygon) if c.mask_polygon else None,
        )
        for c in found
    ]


__all__ = [
    'DEFAULT_MAX_CANDIDATES',
    'DEFAULT_TIMEOUT_S',
    'SegmentCandidateData',
    'SegmenterCallError',
    'first_segmenter_url',
    'segment_image_http',
    'segment_once',
    'segmenter_instances',
]
