"""The crop path (``segment_multi``) sends exactly the payload it always did."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from scripts.curation.worker.client import SegmenterClient


pytestmark = pytest.mark.asyncio

_JPEG = b'\xff\xd8\xff\xe0fake-jpeg'
_HOST = 'http://sam3-full-image-fake:7000'


def _client(handler: Any, **kw: Any) -> SegmenterClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    return SegmenterClient(base_url=_HOST, client=http, source_name='sam3', **kw)


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
    assert await sam.segment_multi(_JPEG) == []
