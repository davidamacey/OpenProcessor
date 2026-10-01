"""W5: the API-side segmenter call used by ``POST /region_profiles/test``.

It is a plain one-shot HTTP call (no worker circuit breaker) that returns
EVERY candidate with its polygon, so an author sees what a prompt finds."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from src.services.detection.segmenter_http import (
    SegmenterCallError,
    first_segmenter_url,
    segment_once,
)


def _client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_every_candidate_is_returned_with_its_polygon_and_the_request_is_exact() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == 'http://seg:8000/segment'
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                'candidates': [
                    {
                        'bbox_norm': [0.1, 0.1, 0.4, 0.4],
                        'score': 0.9,
                        'mask_iou': 0.8,
                        'mask_polygon': [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
                    },
                    {'bbox_norm': [0.5, 0.5, 0.9, 0.9], 'score': 0.4, 'mask_iou': None},
                    {'bbox_norm': [0.5], 'score': 0.3},
                ],
                'elapsed_ms': 3.0,
                'crop_size': [10, 10],
                'prompt': 'wheel',
            },
        )

    async with _client(handler) as http:
        out = await segment_once(
            'http://seg:8000/',
            b'JPEG',
            'wheel',
            max_candidates=8,
            return_masks=True,
            client=http,
        )

    assert seen == [
        {
            'crop_jpeg_b64': base64.b64encode(b'JPEG').decode(),
            'text_prompt': 'wheel',
            'max_candidates': 8,
            'return_masks': True,
        }
    ]
    # The malformed third entry is dropped; the other two keep order.
    assert [c.score for c in out] == [0.9, 0.4]
    assert out[0].bbox_norm == (0.1, 0.1, 0.4, 0.4)
    assert out[0].mask_iou == 0.8
    assert out[0].mask_polygon == [(0.1, 0.1), (0.4, 0.1), (0.4, 0.4)]
    assert out[1].mask_polygon is None


@pytest.mark.asyncio
async def test_a_transport_failure_or_bad_reply_is_a_typed_error_not_an_empty_list() -> None:
    """An outage must not look like "the prompt found nothing"."""

    def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('refused')

    def five_hundred(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text='loading')

    def not_json(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='<html>')

    for handler in (down, five_hundred, not_json):
        async with _client(handler) as http:
            with pytest.raises(SegmenterCallError):
                await segment_once(
                    'http://seg:8000',
                    b'x',
                    'wheel',
                    max_candidates=4,
                    return_masks=False,
                    client=http,
                )


@pytest.mark.asyncio
async def test_a_redirect_is_not_followed() -> None:
    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        return httpx.Response(302, headers={'location': 'http://evil.example/segment'})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(SegmenterCallError):
            await segment_once(
                'http://seg:8000', b'x', 'wheel', max_candidates=4, return_masks=False, client=http
            )
    assert hits == ['http://seg:8000/segment']


def test_the_first_configured_url_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_SEGMENTER_URL', ' http://a:8000/ , http://b:8000')
    assert first_segmenter_url() == 'http://a:8000'
    monkeypatch.setenv('OP_SEGMENTER_URL', '')
    assert first_segmenter_url() is None
    monkeypatch.delenv('OP_SEGMENTER_URL')
    assert first_segmenter_url() is None
