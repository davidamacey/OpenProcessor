"""The VLM labeler maps only what a chat call really raises to "no answer"; a bug surfaces.

``CHAT_CALL_ERRORS`` is what ``post_chat_with_retry`` and the egress check raise (httpx errors,
a non-JSON reply, the egress refusal). A ``KeyError`` from our own code is not one of them: it
used to be logged as a failed chunk, so a programming error looked like a VLM outage.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from src.services.labeling.vlm_endpoints import VlmEndpointUnavailableError
from src.services.labeling.vlm_labeler import VlmLabeler
from src.services.labeling.vlm_models import RegionCrop


def _labeler() -> VlmLabeler:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={}))
    )
    return VlmLabeler(
        base_url='http://vlm.invalid/v1',
        model='m',
        client=client,
        requests_per_second=1000.0,
        usage_source=None,
    )


def _raising(exc: BaseException) -> Any:
    async def post(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise exc

    return post


@pytest.mark.asyncio
async def test_a_bug_in_the_chat_call_is_not_an_unreachable_endpoint() -> None:
    labeler = _labeler()
    labeler._post_chat = _raising(KeyError('bug'))  # type: ignore[method-assign]
    with pytest.raises(KeyError):
        await labeler.health()


@pytest.mark.asyncio
async def test_a_bug_in_the_chat_call_is_not_no_answer_for_the_visibility_probe() -> None:
    labeler = _labeler()
    labeler._post_chat = _raising(KeyError('bug'))  # type: ignore[method-assign]
    with pytest.raises(KeyError):
        await labeler.prompt_visible(b'JPEG', 'cup')


@pytest.mark.asyncio
async def test_a_bug_is_not_a_failed_verify_chunk() -> None:
    labeler = _labeler()
    labeler._post_chat = _raising(KeyError('bug'))  # type: ignore[method-assign]
    crops = [RegionCrop(crop_id='c1', jpeg_bytes=b'JPEG')]
    with pytest.raises(KeyError):
        await labeler._verify_region_chunk(crops)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'exc',
    [
        httpx.ConnectError('down'),
        httpx.HTTPStatusError(
            'boom', request=httpx.Request('POST', 'http://x'), response=httpx.Response(500)
        ),
        ValueError('reply is not JSON'),
        VlmEndpointUnavailableError('refused'),
    ],
)
async def test_what_a_chat_call_raises_is_still_no_answer(exc: Exception) -> None:
    labeler = _labeler()
    labeler._post_chat = _raising(exc)  # type: ignore[method-assign]
    assert await labeler.prompt_visible(b'JPEG', 'cup') is None
    assert (await labeler.health()).reachable is False
