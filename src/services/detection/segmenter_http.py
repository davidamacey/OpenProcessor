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
    parsed = (_candidate(c) for c in body.get('candidates') or [])
    return [c for c in parsed if c is not None]


__all__ = [
    'DEFAULT_MAX_CANDIDATES',
    'DEFAULT_TIMEOUT_S',
    'SegmentCandidateData',
    'SegmenterCallError',
    'first_segmenter_url',
    'segment_once',
]
