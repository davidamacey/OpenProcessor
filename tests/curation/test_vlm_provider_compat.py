"""Hosted OpenAI-compatible providers: request parameters are per-endpoint capabilities.

Anthropic's OpenAI SDK compatibility page (platform.claude.com/docs/en/api/openai-sdk): a
``response_format`` of ``json_object`` returns 400, ``temperature`` below 1 returns 400 on current
models, and its native ``/models`` listing needs the ``anthropic-version`` header. The local vLLM
path (the defaults) must send exactly what it always sent.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from src.domain.job_failure import JobFailedError
from src.services.labeling.vlm_endpoint_body import VlmEndpointBody
from src.services.labeling.vlm_endpoint_probe import probe_endpoint
from src.services.labeling.vlm_endpoints import env_endpoint
from src.services.labeling.vlm_labeler import VlmLabeler
from src.services.labeling.vlm_models import ItemCrop, VlmRequestRejectedError


BASE = 'http://vlm.test/v1'
OK_REPLY = {'choices': [{'message': {'content': '{"results": []}'}}], 'usage': {'prompt_tokens': 5}}


class Recorder:
    def __init__(self, *, chat_status: int = 200, reject_params: bool = False) -> None:
        self.requests: list[httpx.Request] = []
        self.chat_status = chat_status
        self.reject_params = reject_params

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': 'm'}]})
        payload = json.loads(request.content)
        if self.reject_params and ('temperature' in payload or 'response_format' in payload):
            return httpx.Response(
                400, json={'error': {'message': '`temperature` is deprecated for this model'}}
            )
        if self.chat_status != 200:
            return httpx.Response(self.chat_status, json={'error': {'message': 'nope'}})
        return httpx.Response(200, json=OK_REPLY)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))

    def chat_payloads(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.method == 'POST']


def _body(**over: Any) -> VlmEndpointBody:
    return VlmEndpointBody(**{'base_url': BASE, 'model': 'm', **over})


def _labeler(rec: Recorder, **over: Any) -> VlmLabeler:
    return VlmLabeler(
        base_url=BASE,
        model='m',
        client=rec.client(),
        requests_per_second=1000.0,
        usage_source=None,
        **over,
    )


CROP = ItemCrop(img_id='c1', jpeg_bytes=b'\xff\xd8\xff')


@pytest.mark.asyncio
async def test_the_local_defaults_send_temperature_zero_and_json_object() -> None:
    rec = Recorder()
    await _labeler(rec).label_or_propose_batch([CROP], ['cup'])
    payload = rec.chat_payloads()[0]
    assert payload['temperature'] == 0.0
    assert payload['response_format'] == {'type': 'json_object'}
    assert list(payload)[-3:] == ['temperature', 'max_tokens', 'response_format']


@pytest.mark.asyncio
async def test_a_null_temperature_is_left_out_of_every_payload() -> None:
    rec = Recorder()
    labeler = _labeler(rec, temperature=None, json_mode=False)
    await labeler.label_or_propose_batch([CROP], ['cup'])
    await labeler.health()
    for payload in rec.chat_payloads():
        assert 'temperature' not in payload
        assert 'response_format' not in payload


@pytest.mark.asyncio
async def test_the_provider_header_is_sent_only_for_the_provider() -> None:
    plain, hosted = Recorder(), Recorder()
    await probe_endpoint(_body(), api_key='k', client=plain.client())
    await probe_endpoint(_body(provider='anthropic'), api_key='k', client=hosted.client())
    assert all('anthropic-version' not in r.headers for r in plain.requests)
    assert {r.headers.get('anthropic-version') for r in hosted.requests} == {'2023-06-01'}
    models = next(r for r in hosted.requests if r.url.path.endswith('/models'))
    assert models.headers['anthropic-version'] == '2023-06-01'


@pytest.mark.asyncio
async def test_the_probe_uses_the_endpoints_configured_parameters() -> None:
    rec = Recorder(reject_params=True)
    body = _body(temperature=None, json_mode='off')
    record = await probe_endpoint(body, api_key=None, client=rec.client())
    assert not any('temperature' in p or 'response_format' in p for p in rec.chat_payloads()), (
        rec.chat_payloads()
    )
    assert 'vlm_http_error' not in {i['code'] for i in record.issues}


@pytest.mark.asyncio
async def test_the_probe_reports_a_rejected_temperature_as_a_configuration_error() -> None:
    rec = Recorder(reject_params=True)
    record = await probe_endpoint(_body(json_mode='off'), api_key=None, client=rec.client())
    assert not record.ok
    issue = next(i for i in record.issues if i['code'] == 'vlm_http_error')
    assert issue['detail']['status'] == 400
    assert 'temperature' in issue['detail']['upstream_message']


@pytest.mark.asyncio
async def test_a_4xx_is_raised_once_with_the_provider_message_not_retried() -> None:
    rec = Recorder(chat_status=400)
    with pytest.raises(VlmRequestRejectedError) as caught:
        await _labeler(rec).label_or_propose_batch([CROP], ['cup'])
    assert caught.value.status == 400
    assert 'nope' in caught.value.upstream_message
    assert len(rec.requests) == 1


@pytest.mark.asyncio
async def test_a_5xx_is_still_no_answer_after_the_retries() -> None:
    rec = Recorder(chat_status=503)
    predictions = await _labeler(rec).label_or_propose_batch([CROP], ['cup'])
    assert [p.failure for p in predictions] == ['request_failed']


@pytest.mark.asyncio
async def test_label_batch_fails_the_job_with_status_message_and_endpoint() -> None:
    from src.services.curation.vlm_batch import fail_on_rejection

    rec = Recorder(chat_status=400)
    labeler = _labeler(rec)
    labeler.identity = labeler.identity.__class__(endpoint_ref='anthropic@3', model='m')
    with pytest.raises(JobFailedError) as caught:
        async with fail_on_rejection(labeler):
            await labeler.label_or_propose_batch([CROP], ['cup'])
    assert caught.value.code == 'vlm_request_rejected'
    for part in ('400', 'nope', 'anthropic'):
        assert part in str(caught.value)


PROVIDER_PARAMS = {
    'vllm': (0.0, 'auto'),
    'ollama': (0.0, 'auto'),
    'openai_compatible': (0.0, 'auto'),
    'anthropic': (None, 'off'),
}


@pytest.mark.parametrize(('provider', 'expected'), PROVIDER_PARAMS.items())
def test_each_provider_has_its_own_parameter_defaults(
    provider: str, expected: tuple[float | None, str]
) -> None:
    body = _body(provider=provider)
    assert (body.temperature, body.json_mode) == expected


def test_a_provider_default_is_overridable_per_endpoint() -> None:
    body = _body(provider='anthropic', temperature=1.0, json_mode='on')
    assert (body.temperature, body.json_mode) == (1.0, 'on')
    assert _body(provider='ollama', temperature=None).temperature is None


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', list(PROVIDER_PARAMS))
async def test_each_provider_probe_sends_its_own_shape_and_succeeds(provider: str) -> None:
    rec = Recorder(reject_params=provider == 'anthropic')
    record = await probe_endpoint(_body(provider=provider), api_key='k', client=rec.client())
    assert record.ok, record.issues
    temperature = PROVIDER_PARAMS[provider][0]
    for payload in rec.chat_payloads():
        assert ('temperature' in payload) == (temperature is not None)
    versions = {r.headers.get('anthropic-version') for r in rec.requests}
    assert versions == ({'2023-06-01'} if provider == 'anthropic' else {None})


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', list(PROVIDER_PARAMS))
async def test_each_provider_maps_a_4xx_to_a_rejection_and_a_5xx_to_no_answer(
    provider: str,
) -> None:
    body = _body(provider=provider)
    kwargs = {'temperature': body.temperature, 'json_mode': body.json_mode != 'off'}
    with pytest.raises(VlmRequestRejectedError):
        await _labeler(
            Recorder(chat_status=401), provider=provider, **kwargs
        ).label_or_propose_batch([CROP], ['cup'])
    answers = await _labeler(
        Recorder(chat_status=500), provider=provider, **kwargs
    ).label_or_propose_batch([CROP], ['cup'])
    assert [a.failure for a in answers] == ['request_failed']


def test_the_env_endpoint_reads_the_documented_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_VLM_URL', 'https://api.anthropic.com/v1/')
    monkeypatch.setenv('OP_VLM_MODEL', 'claude-haiku-5-5')
    monkeypatch.setenv('OP_VLM_PROVIDER', 'anthropic')
    monkeypatch.setenv('OP_VLM_JSON_MODE', 'off')
    monkeypatch.setenv('OP_VLM_TEMPERATURE', 'none')
    endpoint = env_endpoint()
    assert endpoint is not None
    body = endpoint.body
    assert (body.provider, body.json_mode, body.temperature) == ('anthropic', 'off', None)
    assert endpoint.json_mode_on is False


def test_the_env_endpoint_defaults_are_the_local_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('OP_VLM_PROVIDER', 'OP_VLM_JSON_MODE', 'OP_VLM_TEMPERATURE'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm:8000/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'm')
    endpoint = env_endpoint()
    assert endpoint is not None
    assert (endpoint.body.provider, endpoint.body.json_mode, endpoint.body.temperature) == (
        'vllm',
        'auto',
        0.0,
    )
