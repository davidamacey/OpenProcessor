"""Wave 1 of the full-image SAM 3 plan: ``SegmenterClient.segment_image``.

The per-call prompt, score floor and mask request are new; the crop path
(``segment_multi``) must send exactly the payload it always did.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from scripts.curation.worker import client as client_mod
from scripts.curation.worker.client import (
    SegmenterClient,
    SegmenterRequestFailed,
    SegmenterUnavailable,
)


pytestmark = pytest.mark.asyncio

_JPEG = b'\xff\xd8\xff\xe0fake-jpeg'
_HOST = 'http://sam3-full-image-fake:7000'


def _client(handler: Any, **kw: Any) -> SegmenterClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    return SegmenterClient(base_url=_HOST, client=http, source_name='sam3', **kw)


async def test_segment_image_sends_per_call_prompt_floor_and_masks() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={'candidates': []})

    sam = _client(handler, text_prompt='client-level')
    await sam.segment_image(
        _JPEG, 'traffic cone', min_score=0.6, max_candidates=7, return_masks=True
    )
    body = seen[0]
    assert body['text_prompt'] == 'traffic cone'
    assert body['min_score'] == 0.6
    assert body['max_candidates'] == 7
    assert body['return_masks'] is True


async def test_segment_image_surfaces_the_polygon() -> None:
    poly = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]

    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'candidates': [
                    {
                        'bbox_norm': [0.1, 0.1, 0.4, 0.4],
                        'score': 0.9,
                        'mask_iou': 0.8,
                        'mask_polygon': poly,
                    },
                    {'bbox_norm': [0.5, 0.5, 0.9, 0.9], 'score': 0.7},
                ]
            },
        )

    out = await _client(handler).segment_image(_JPEG, 'cup', return_masks=True)
    assert out[0].mask_polygon == ((0.1, 0.1), (0.4, 0.1), (0.4, 0.4))
    assert out[0].rectangularity == 0.8
    assert out[1].mask_polygon is None


async def test_segment_image_failure_is_unavailable_not_empty(monkeypatch) -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={'error': 'boom'})

    sam = _client(handler)
    monkeypatch.setattr(client_mod, '_RETRY_BACKOFFS', (0.0, 0.0))
    with pytest.raises(SegmenterUnavailable):
        await sam.segment_image(_JPEG, 'cup')
    assert issubclass(SegmenterRequestFailed, SegmenterUnavailable)


async def test_crop_path_payload_is_unchanged() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={'candidates': []})

    await _client(handler, text_prompt='wheel', max_candidates=3).segment_multi(_JPEG)
    assert set(seen[0]) == {'crop_jpeg_b64', 'text_prompt', 'max_candidates'}
    assert seen[0]['text_prompt'] == 'wheel'
    assert seen[0]['max_candidates'] == 3


async def test_crop_path_ignores_polygons_the_server_sends() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'candidates': [
                    {'bbox_norm': [0, 0, 1, 1], 'score': 0.9, 'mask_polygon': [[0, 0], [1, 1]]}
                ]
            },
        )

    out = await _client(handler, text_prompt='wheel').segment_multi(_JPEG)
    assert out[0].mask_polygon is None


async def test_disabled_client_returns_empty_without_a_call() -> None:
    sam = SegmenterClient(base_url=None, source_name='sam3')
    assert await sam.segment_image(_JPEG, 'cup') == []
