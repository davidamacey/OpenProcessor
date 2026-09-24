"""Unit tests for ``src.services.labeling.vlm_labeler.VlmLabeler``.

Ported from the reference ``tests/test_gemma_labeler.py`` (§5 Chunk 7 —
see ``docs/design/curation_design_rationale.md`` for the genericization
approach). Mechanism only — the reference file used generic placeholder
class names ('acura', 'bmw', 'porsche') purely as opaque strings for the
chunking/parsing tests; this port swaps them for equally-opaque neutral
strings so nothing vehicle-specific survives, per the "port the
mechanism, not the vocabulary" principle.

Tests are intentionally hermetic:
- All upstream HTTP traffic is replaced with a fake httpx transport.
- Tenacity retry waits are monkeypatched to zero so retry tests don't
  block the suite.
- Async tests use ``asyncio.run`` directly so we don't depend on the
  optional ``pytest-asyncio`` plugin for this file (other ported files
  in this chunk do use it).
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from src.services.labeling.vlm_labeler import (
    ItemCrop as _ItemCrop,
    RegionCrop,
    VlmClassPrediction,
    VlmHealth,
    VlmLabeler,
    VlmRegionVerdict,
    _strip_markdown_fences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_response(content: str) -> dict[str, object]:
    """Wrap ``content`` in an OpenAI-shaped /chat/completions response."""

    return {
        'id': 'chatcmpl-test',
        'object': 'chat.completion',
        'choices': [
            {
                'index': 0,
                'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': content},
            }
        ],
    }


class _FakeTransport(httpx.AsyncBaseTransport):
    """Records calls and replies according to a scripted handler."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return self.handler(request, len(self.calls))


def _fake_client(handler) -> tuple[httpx.AsyncClient, _FakeTransport]:
    transport = _FakeTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=5.0)
    return client, transport


def _make_jpeg(tag: bytes = b'\xff\xd8\xff\xe0fake') -> bytes:
    """Return any non-empty bytes blob — base64 encoder doesn't validate."""

    return tag


def _run(coro):
    """Run an async coroutine in a fresh event loop."""

    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _fast_retry_waits(monkeypatch):
    """Make tenacity ``wait_exponential`` wait ~0s so retry tests are fast."""

    import tenacity

    real_wait_exponential = tenacity.wait_exponential

    def _instant_wait(*_args, **_kwargs):
        # Multiplier=0, max=0 → tenacity always waits 0s between attempts.
        return real_wait_exponential(multiplier=0, max=0)

    monkeypatch.setattr('src.services.labeling.vlm_client.wait_exponential', _instant_wait)


# ---------------------------------------------------------------------------
# Marker-fence helpers
# ---------------------------------------------------------------------------


def test_strip_markdown_fences_handles_json_block():
    assert (
        _strip_markdown_fences('```json\n[{"img":1,"class":"alpha","confidence":"high"}]\n```')
        == '[{"img":1,"class":"alpha","confidence":"high"}]'
    )


def test_strip_markdown_fences_handles_unlabeled_block():
    assert _strip_markdown_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_markdown_fences_passes_through_plain_text():
    raw = '[{"img":1,"class":"alpha","confidence":"low"}]'
    assert _strip_markdown_fences(raw) == raw


# ---------------------------------------------------------------------------
# Batch labeling: chunking
# ---------------------------------------------------------------------------


def test_label_vehicle_batch_chunks_more_than_four_crops_into_multiple_calls():
    """5 crops + max 4 per call should produce exactly 2 upstream calls."""

    def handler(request: httpx.Request, call_idx: int) -> httpx.Response:
        body = json.loads(request.content)
        # Count how many image_url parts are in the user message.
        user_msg = body['messages'][-1]['content']
        n_images = sum(1 for part in user_msg if part.get('type') == 'image_url')
        # Echo back a well-formed prediction for each numbered image.
        items = [{'img': i, 'class': 'alpha', 'confidence': 'high'} for i in range(1, n_images + 1)]
        return httpx.Response(200, json=_make_chat_response(json.dumps(items)))

    client, transport = _fake_client(handler)

    async def _go() -> list[VlmClassPrediction]:
        async with VlmLabeler(
            base_url='http://fake/v1',
            model='fake-vlm',
            api_key='EMPTY',
            max_images_per_call=4,
            requests_per_second=1000.0,  # don't throttle the tests
            client=client,
        ) as labeler:
            crops = [_ItemCrop(img_id=f'crop-{i}', jpeg_bytes=_make_jpeg()) for i in range(5)]
            return await labeler.label_vehicle_batch(crops, class_names=['alpha', 'beta'])

    results = _run(_go())

    assert len(results) == 5
    assert all(r.class_name == 'alpha' for r in results)
    assert all(r.confidence == 'high' for r in results)
    # 5 crops → ceil(5/4) == 2 upstream calls
    assert len(transport.calls) == 2

    # First call should have 4 image parts, second should have 1.
    first_body = json.loads(transport.calls[0].content)
    second_body = json.loads(transport.calls[1].content)
    n_images_first = sum(
        1 for p in first_body['messages'][-1]['content'] if p.get('type') == 'image_url'
    )
    n_images_second = sum(
        1 for p in second_body['messages'][-1]['content'] if p.get('type') == 'image_url'
    )
    assert n_images_first == 4
    assert n_images_second == 1


def test_construction_fails_loudly_with_a_url_but_no_model():
    """S7: there is no hardcoded vendor model-id default. A base_url with
    an empty model must raise at construction, not silently talk to a
    vendor-named model."""
    with pytest.raises(ValueError, match='OP_VLM_MODEL'):
        VlmLabeler(base_url='http://fake/v1', model='')


def test_construction_with_no_url_and_no_model_does_not_raise():
    """An unconfigured deployment (no OP_VLM_URL) must not fail merely for
    lacking a model id -- the VLM leg is simply off."""
    labeler = VlmLabeler(base_url='', model='')
    assert labeler.model == ''
    _run(labeler.aclose())


def test_label_vehicle_batch_clamps_max_images_per_call_above_hard_cap():
    """Constructor clamps anything above the hard cap (8) down to it."""

    client, _ = _fake_client(lambda *_a: httpx.Response(200, json=_make_chat_response('[]')))
    labeler = VlmLabeler(
        base_url='http://fake/v1',
        max_images_per_call=16,  # higher than the hard cap
        requests_per_second=1000.0,
        client=client,
    )
    assert labeler.max_images_per_call == 8
    _run(labeler.aclose())


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def test_label_vehicle_batch_parses_markdown_fenced_response():
    """A VLM sometimes wraps output in ```json ... ``` despite the system prompt."""

    fenced = '```json\n[{"img":1,"class":"gamma","confidence":"medium"}]\n```'

    def handler(_request: httpx.Request, _call_idx: int) -> httpx.Response:
        return httpx.Response(200, json=_make_chat_response(fenced))

    client, _ = _fake_client(handler)

    async def _go() -> list[VlmClassPrediction]:
        async with VlmLabeler(
            base_url='http://fake/v1',
            client=client,
            requests_per_second=1000.0,
        ) as labeler:
            return await labeler.label_vehicle_batch(
                [_ItemCrop(img_id='only', jpeg_bytes=_make_jpeg())],
                class_names=['gamma'],
            )

    [result] = _run(_go())
    assert result.img_id == 'only'
    assert result.class_name == 'gamma'
    assert result.confidence == 'medium'


def test_label_vehicle_batch_returns_low_confidence_fallback_on_garbage():
    def handler(_request: httpx.Request, _call_idx: int) -> httpx.Response:
        return httpx.Response(200, json=_make_chat_response('this is not json'))

    client, _ = _fake_client(handler)

    async def _go() -> list[VlmClassPrediction]:
        async with VlmLabeler(
            base_url='http://fake/v1',
            client=client,
            requests_per_second=1000.0,
        ) as labeler:
            return await labeler.label_vehicle_batch(
                [
                    _ItemCrop(img_id='a', jpeg_bytes=_make_jpeg()),
                    _ItemCrop(img_id='b', jpeg_bytes=_make_jpeg()),
                ],
                class_names=['alpha'],
            )

    results = _run(_go())
    assert [r.class_name for r in results] == ['', '']
    assert [r.confidence for r in results] == ['low', 'low']


# ---------------------------------------------------------------------------
# Rate limiter — light contract test only
# ---------------------------------------------------------------------------


def test_rate_limiter_field_is_honored_via_constructor():
    """Token-bucket field is exposed; full timing test omitted (no time-machine)."""

    client, _ = _fake_client(lambda *_a: httpx.Response(200, json=_make_chat_response('[]')))
    labeler = VlmLabeler(base_url='http://fake/v1', requests_per_second=2.5, client=client)
    assert labeler.requests_per_second == 2.5
    # Token-bucket internal rate matches.
    assert labeler._bucket.rate == pytest.approx(2.5)
    _run(labeler.aclose())


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_retry_triggers_on_503_then_succeeds():
    """503 on call #1 should cause tenacity to retry; call #2 succeeds."""

    def handler(_request: httpx.Request, call_idx: int) -> httpx.Response:
        if call_idx == 1:
            return httpx.Response(503, text='upstream-overloaded')
        return httpx.Response(
            200,
            json=_make_chat_response('[{"img":1,"class":"beta","confidence":"high"}]'),
        )

    client, transport = _fake_client(handler)

    async def _go() -> list[VlmClassPrediction]:
        async with VlmLabeler(
            base_url='http://fake/v1',
            client=client,
            requests_per_second=1000.0,
        ) as labeler:
            return await labeler.label_vehicle_batch(
                [_ItemCrop(img_id='x', jpeg_bytes=_make_jpeg())],
                class_names=['beta'],
            )

    [result] = _run(_go())
    assert result.class_name == 'beta'
    assert result.confidence == 'high'
    assert len(transport.calls) == 2  # 1 failed + 1 success


def test_retry_gives_up_after_max_attempts_and_returns_fallback():
    """All 3 attempts return 503 — labeler should yield low-confidence fallback,
    not raise."""

    def handler(_request: httpx.Request, _call_idx: int) -> httpx.Response:
        return httpx.Response(503, text='still-overloaded')

    client, transport = _fake_client(handler)

    async def _go() -> list[VlmClassPrediction]:
        async with VlmLabeler(
            base_url='http://fake/v1',
            client=client,
            requests_per_second=1000.0,
        ) as labeler:
            return await labeler.label_vehicle_batch(
                [_ItemCrop(img_id='x', jpeg_bytes=_make_jpeg())],
                class_names=['beta'],
            )

    [result] = _run(_go())
    assert result.class_name == ''
    assert result.confidence == 'low'
    assert len(transport.calls) == 3  # full retry budget consumed


# ---------------------------------------------------------------------------
# Health probe
# ---------------------------------------------------------------------------


def test_health_returns_unreachable_on_connection_error():
    """All 3 attempts raise ConnectError → health() returns reachable=False."""

    def handler(_request: httpx.Request, _call_idx: int) -> httpx.Response:
        raise httpx.ConnectError('cannot connect')

    client, transport = _fake_client(handler)

    async def _go() -> VlmHealth:
        async with VlmLabeler(
            base_url='http://fake/v1',
            model='fake-vlm',
            client=client,
            requests_per_second=1000.0,
        ) as labeler:
            return await labeler.health()

    h = _run(_go())
    assert h.reachable is False
    assert h.model == 'fake-vlm'
    assert h.last_error is not None
    assert 'ConnectError' in h.last_error
    # 3 retry attempts before giving up.
    assert len(transport.calls) == 3


def test_health_returns_reachable_on_200():
    def handler(_request: httpx.Request, _call_idx: int) -> httpx.Response:
        return httpx.Response(200, json=_make_chat_response('OK'))

    client, _ = _fake_client(handler)

    async def _go() -> VlmHealth:
        async with VlmLabeler(
            base_url='http://fake/v1',
            client=client,
            requests_per_second=1000.0,
        ) as labeler:
            return await labeler.health()

    h = _run(_go())
    assert h.reachable is True
    assert h.last_error is None


# ---------------------------------------------------------------------------
# Region verification
# ---------------------------------------------------------------------------


def test_verify_plate_parses_well_formed_json():
    body = '{"is_region": true, "confidence": "high", "reason": "clearly a label region"}'

    def handler(_request: httpx.Request, _call_idx: int) -> httpx.Response:
        return httpx.Response(200, json=_make_chat_response(body))

    client, _ = _fake_client(handler)

    async def _go() -> VlmRegionVerdict:
        async with VlmLabeler(
            base_url='http://fake/v1',
            client=client,
            requests_per_second=1000.0,
        ) as labeler:
            return await labeler.verify_plate(
                RegionCrop(crop_id='region-1', jpeg_bytes=_make_jpeg())
            )

    verdict = _run(_go())
    assert verdict.crop_id == 'region-1'
    assert verdict.is_region is True
    assert verdict.confidence == 'high'
    assert 'label region' in verdict.reason


def test_verify_plate_returns_parse_failure_on_garbage():
    def handler(_request: httpx.Request, _call_idx: int) -> httpx.Response:
        return httpx.Response(200, json=_make_chat_response('definitely not json'))

    client, _ = _fake_client(handler)

    async def _go() -> VlmRegionVerdict:
        async with VlmLabeler(
            base_url='http://fake/v1',
            client=client,
            requests_per_second=1000.0,
        ) as labeler:
            return await labeler.verify_plate(
                RegionCrop(crop_id='region-1', jpeg_bytes=_make_jpeg())
            )

    verdict = _run(_go())
    assert verdict.is_region is False
    assert verdict.confidence == 'low'
    assert verdict.reason == 'parse_failure'
