"""``VlmLabeler.prompt_visible``: the tier-2 yes/no. A failure or an unreadable
reply is ``None`` (no answer), never ``False``."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from src.services.labeling.vlm_labeler import VlmLabeler


def _labeler(handler: Any) -> VlmLabeler:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return VlmLabeler(
        base_url='http://vlm.invalid/v1', model='m', client=client, requests_per_second=1000.0
    )


def _reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={'choices': [{'message': {'content': content}}]})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('content', 'expected'),
    [
        ('{"visible": true}', True),
        ('{"visible": false}', False),
        ('```json\n{"visible": "no"}\n```', False),
        ('{"visible": "null"}', None),
        ('{"seen": true}', None),
        ('[true]', None),
        ('I think so', None),
    ],
)
async def test_the_answer_is_read_strictly(content: str, expected: bool | None) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _reply(content)

    assert await _labeler(handler).prompt_visible(b'JPEG', 'traffic cone') is expected
    body = seen[0]
    assert 'traffic cone' in body['messages'][1]['content'][0]['text']
    assert body['messages'][1]['content'][1]['image_url']['url'].startswith(
        'data:image/jpeg;base64,'
    )


@pytest.mark.asyncio
async def test_a_failed_call_is_no_answer_not_a_no() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={'error': 'bad'})

    assert await _labeler(handler).prompt_visible(b'JPEG', 'cup') is None
