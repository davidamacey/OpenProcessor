"""The endpoint probe (W9.4) against an in-process ``httpx.MockTransport``:
what it sends, what it records, and what it never does (send user data,
follow a redirect, echo a key)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from typing import Any

import httpx
import pytest
from PIL import Image

from src.services.labeling import vlm_endpoint_probe as probe_mod
from src.services.labeling.vlm_endpoint_body import VlmEndpointBody
from src.services.labeling.vlm_endpoint_probe import ProbeBusyError, probe_endpoint


BASE = 'http://vlm.test/v1'


def _body(**over: Any) -> VlmEndpointBody:
    return VlmEndpointBody(**{'base_url': BASE, 'model': 'm', 'max_images_per_call': 3, **over})


def _chat_reply(text: str, *, prompt_tokens: int = 10, reasoning: str | None = None) -> dict:
    message: dict[str, Any] = {'content': text}
    if reasoning is not None:
        message['reasoning_content'] = reasoning
    return {'choices': [{'message': message}], 'usage': {'prompt_tokens': prompt_tokens}}


class Server:
    """A scriptable OpenAI-shaped server that records every request."""

    def __init__(self, **over: Any) -> None:
        self.requests: list[httpx.Request] = []
        self.models = over.get(
            'models',
            {'data': [{'id': 'm', 'root': 'org/real-model', 'max_model_len': 8192}]},
        )
        self.reject_json_mode = over.get('reject_json_mode', False)
        self.image_status = over.get('image_status', 200)
        self.many_status = over.get('many_status', 200)
        self.image_reply = over.get('image_reply', '{"color": "red"}')
        self.models_status = over.get('models_status', 200)
        self.redirect = over.get('redirect', False)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.redirect:
            return httpx.Response(302, headers={'location': 'http://169.254.169.254/x'})
        if request.url.path.endswith('/models'):
            return httpx.Response(self.models_status, json=self.models)
        payload = json.loads(request.content)
        content = payload['messages'][0]['content']
        images = (
            [p for p in content if isinstance(p, dict) and p.get('type') == 'image_url']
            if isinstance(content, list)
            else []
        )
        if not images:
            return httpx.Response(200, json=_chat_reply('OK', prompt_tokens=10))
        if 'response_format' in payload and self.reject_json_mode:
            return httpx.Response(400, text='response_format json_object is not supported')
        if len(images) > 1:
            if self.many_status != 200:
                return httpx.Response(self.many_status, text='too many images: limit is 2')
            return httpx.Response(200, json=_chat_reply('OK'))
        if self.image_status != 200:
            return httpx.Response(self.image_status, text='images are not supported')
        return httpx.Response(200, json=_chat_reply(self.image_reply, prompt_tokens=270))

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


async def _probe(server: Server, body: VlmEndpointBody | None = None, key: str | None = None):
    async with server.client() as client:
        return await probe_endpoint(body or _body(), api_key=key, client=client)


def _codes(record) -> set[str]:
    return {i['code'] for i in record.issues}


@pytest.mark.asyncio
async def test_a_healthy_endpoint_records_its_identity() -> None:
    record = await _probe(Server())
    assert record.ok
    assert record.issues == []
    assert record.model_listed is True
    assert record.root == 'org/real-model'
    assert record.max_model_len == 8192
    assert record.vision_ok is True
    assert record.json_mode_supported is True
    assert record.max_images_ok is True
    assert record.image_tokens == 260  # 270 with the image, 10 without
    assert record.models_listed == ['m']


@pytest.mark.asyncio
async def test_only_synthetic_images_are_sent() -> None:
    server = Server()
    await _probe(server, _body(max_images_per_call=4))
    sizes = set()
    for request in server.requests:
        if request.method != 'POST':
            continue
        for part in json.loads(request.content)['messages'][0]['content']:
            if isinstance(part, dict) and part.get('type') == 'image_url':
                data = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
                image = Image.open(io.BytesIO(data))
                sizes.add(image.size)
                # a solid colour: no user content can be in it
                assert len(set(image.convert('RGB').getdata())) <= 8
    assert sizes == {(64, 64)}


@pytest.mark.asyncio
async def test_the_cap_call_carries_exactly_the_configured_number_of_images() -> None:
    server = Server()
    await _probe(server, _body(max_images_per_call=5))
    counts = []
    for request in server.requests:
        if request.method == 'POST':
            content = json.loads(request.content)['messages'][0]['content']
            counts.append(
                sum(1 for p in content if isinstance(p, dict) and p['type'] == 'image_url')
            )
    assert counts == [0, 1, 5]


@pytest.mark.asyncio
async def test_a_cap_of_one_skips_the_multi_image_call() -> None:
    server = Server()
    record = await _probe(server, _body(max_images_per_call=1))
    assert record.max_images_ok is None
    assert len([r for r in server.requests if r.method == 'POST']) == 2


@pytest.mark.asyncio
async def test_an_unlisted_model_stops_the_probe_early() -> None:
    server = Server(models={'data': [{'id': 'other'}]})
    record = await _probe(server)
    assert not record.ok
    assert _codes(record) == {'vlm_model_not_listed'}
    assert record.model_listed is False
    assert [r.method for r in server.requests] == ['GET']  # nothing else was sent


@pytest.mark.asyncio
async def test_a_missing_models_route_is_only_a_warning() -> None:
    record = await _probe(Server(models_status=404))
    assert record.ok
    assert 'vlm_models_endpoint_missing' in _codes(record)


@pytest.mark.asyncio
async def test_json_mode_rejection_is_recorded_and_retried_without_it() -> None:
    server = Server(reject_json_mode=True)
    record = await _probe(server)
    assert record.ok
    assert record.json_mode_supported is False
    assert 'vlm_json_mode_unsupported' in _codes(record)
    bodies = [json.loads(r.content) for r in server.requests if r.method == 'POST']
    image_calls = [
        b
        for b in bodies
        if any(
            isinstance(p, dict) and p['type'] == 'image_url'
            for p in b['messages'][0]['content']
            if not isinstance(b['messages'][0]['content'], str)
        )
    ]
    assert 'response_format' in image_calls[0]
    assert 'response_format' not in image_calls[1]


@pytest.mark.asyncio
async def test_a_text_only_server_is_no_vision() -> None:
    record = await _probe(Server(image_status=400))
    assert not record.ok
    assert 'vlm_no_vision' in _codes(record)
    assert record.vision_ok is False


@pytest.mark.asyncio
async def test_a_wrong_colour_answer_is_a_warning_not_an_error() -> None:
    record = await _probe(Server(image_reply='{"color": "green"}'))
    assert record.ok
    assert record.vision_ok is False
    assert 'vlm_vision_answer_wrong' in _codes(record)


@pytest.mark.asyncio
async def test_a_server_that_rejects_its_own_image_cap_is_an_error() -> None:
    record = await _probe(Server(many_status=400), _body(max_images_per_call=4))
    assert not record.ok
    assert record.max_images_ok is False
    (item,) = [i for i in record.issues if i['code'] == 'vlm_max_images_exceeds_server']
    assert item['detail']['max_images_per_call'] == 4
    assert 'too many images' in item['detail']['upstream_message']


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [401, 403])
async def test_bad_credentials_are_an_auth_failure(status: int) -> None:
    record = await _probe(Server(models_status=status))
    assert 'vlm_auth_failed' in _codes(record)
    assert not record.ok


@pytest.mark.asyncio
async def test_a_redirect_is_never_followed() -> None:
    server = Server(redirect=True)
    record = await _probe(server)
    assert not record.ok
    assert 'vlm_http_error' in _codes(record)
    hosts = {r.url.host for r in server.requests}
    assert hosts == {'vlm.test'}  # never the redirect target
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_the_default_client_does_not_follow_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client the probe builds for itself is redirect-free too."""
    seen: dict[str, Any] = {}
    real = httpx.AsyncClient

    def spy(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        seen.update(kwargs)
        return real(*args, transport=httpx.MockTransport(Server(redirect=True)), **kwargs)

    monkeypatch.setattr(probe_mod.httpx, 'AsyncClient', spy)
    record = await probe_endpoint(_body(), api_key=None)
    assert seen.get('follow_redirects') is False
    assert not record.ok


@pytest.mark.asyncio
async def test_an_unreachable_server_is_recorded_not_raised() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('refused', request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as client:
        record = await probe_endpoint(_body(), api_key=None, client=client)
    assert _codes(record) == {'vlm_unreachable'}


@pytest.mark.asyncio
async def test_a_timeout_is_recorded_with_the_step_budget() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout('slow', request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
        record = await probe_endpoint(_body(), api_key=None, client=client)
    assert _codes(record) == {'vlm_timeout'}
    (item,) = record.issues
    assert '10s' in item['message']


@pytest.mark.asyncio
async def test_a_key_echoed_by_the_upstream_is_masked_and_never_in_the_record() -> None:
    key = 'sk-very-secret-value'

    class Echo(Server):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            response = super().__call__(request)
            if response.status_code == 400 and 'images are not supported' in response.text:
                return httpx.Response(400, text=f'bad request, header was Bearer {key}')
            return response

    echo = Echo(image_status=400)
    record = await _probe(echo, key=key)
    dumped = json.dumps(record.model_dump())
    assert key not in dumped
    assert '***' in dumped
    # the key still went to the server as a header, in memory only
    assert any(r.headers.get('authorization') == f'Bearer {key}' for r in echo.requests)


@pytest.mark.asyncio
async def test_the_per_process_concurrency_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'data': [{'id': 'm'}]})

    class Blocking(httpx.MockTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await gate.wait()
            return await super().handle_async_request(request)

    async def run() -> object:
        async with httpx.AsyncClient(transport=Blocking(handler)) as client:
            return await probe_endpoint(_body(), api_key=None, client=client)

    running = [asyncio.create_task(run()) for _ in range(probe_mod.MAX_CONCURRENT_PROBES)]
    await asyncio.sleep(0.05)
    with pytest.raises(ProbeBusyError):
        await asyncio.wait_for(run(), timeout=2)
    gate.set()
    await asyncio.gather(*running)
    # the slots are released afterwards
    async with Server().client() as client:
        assert (await probe_endpoint(_body(), api_key=None, client=client)).ok
