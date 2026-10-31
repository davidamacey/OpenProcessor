"""DNS-rebinding window (#101): the VLM connection goes to the address that
passed the URL policy, never to whatever the name resolves to afterwards.

The connect target is observed one layer above the socket: httpcore's
``connect_tcp`` is replaced by a recorder that refuses the connection, so
what is asserted is the host the real client actually asked the network
layer to open."""

from __future__ import annotations

from typing import Any

import httpcore
import httpx
import pytest

import src.services.labeling.vlm_url_policy as policy
from src.services.labeling import vlm_factory
from src.services.labeling.vlm_endpoint_body import VlmEndpointBody
from src.services.labeling.vlm_endpoint_probe import probe_endpoint
from src.services.labeling.vlm_endpoints import VlmEndpoint
from src.services.labeling.vlm_factory import VlmEndpointDeniedError, labeler_for
from src.services.labeling.vlm_pinning import AddressPin, PinnedTransport
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


GOOD = '10.0.0.9'
EVIL = '10.0.0.66'  # also private/allowed by policy: only the pin can tell the two apart
METADATA = '169.254.169.254'


class _Dns:
    def __init__(self) -> None:
        self.answer: dict[str, list[str]] = {}
        self.calls: list[str] = []

    def __call__(self, host: str) -> list[str]:
        self.calls.append(host)
        return list(self.answer.get(host, []))


@pytest.fixture
def dns(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> _Dns:
    table = _Dns()
    monkeypatch.setattr(policy, '_resolve', table)
    monkeypatch.setenv('OP_VLM_SECRETS_DIR', str(tmp_path))
    policy.reset_policy_caches()
    vlm_factory.reset_labeler_cache()
    return table


@pytest.fixture
def connects(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Every (host, port) the HTTP stack asks the network layer to open."""
    seen: list[tuple[str, int]] = []

    async def connect_tcp(self: Any, host: str, port: int, **_kw: Any) -> Any:
        seen.append((host, port))
        raise httpcore.ConnectError('recorded, not connected')

    monkeypatch.setattr(
        'httpcore._backends.anyio.AnyIOBackend.connect_tcp', connect_tcp, raising=True
    )
    return seen


def _endpoint(base_url: str, **body: Any) -> VlmEndpoint:
    return VlmEndpoint(
        name='one',
        source='stored',
        revision=1,
        body=VlmEndpointBody(base_url=base_url, model='m', **body),
        etag='e',
        last_probe=None,
    )


@pytest.mark.asyncio
async def test_a_rebound_name_still_connects_to_the_validated_address(
    dns: _Dns, connects: list[tuple[str, int]]
) -> None:
    dns.answer['vlm.corp.example'] = [GOOD]
    labeler = labeler_for(_endpoint('http://vlm.corp.example:8000/v1'), GENERIC_ITEM_PACK)

    dns.answer['vlm.corp.example'] = [EVIL]  # the name is rebound after validation
    with pytest.raises(httpx.ConnectError):
        await labeler._post_chat({'x': 1}, attempts=1)

    assert connects == [(GOOD, 8000)]
    await labeler.aclose()


@pytest.mark.asyncio
async def test_the_pinned_client_does_not_resolve_the_name_itself(
    dns: _Dns, connects: list[tuple[str, int]]
) -> None:
    dns.answer['vlm.corp.example'] = [GOOD]
    labeler = labeler_for(_endpoint('http://vlm.corp.example/v1'), GENERIC_ITEM_PACK)
    resolved_by_policy = len(dns.calls)
    with pytest.raises(httpx.ConnectError):
        await labeler._post_chat({}, attempts=1)
    assert len(dns.calls) == resolved_by_policy
    assert [h for h, _ in connects] == [GOOD]
    await labeler.aclose()


@pytest.mark.asyncio
async def test_a_recheck_that_now_resolves_somewhere_denied_refuses_and_never_connects(
    dns: _Dns, connects: list[tuple[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    clock = {'t': 1000.0}
    monkeypatch.setattr(vlm_factory, 'time', SimpleNamespace(monotonic=lambda: clock['t']))
    dns.answer['vlm.corp.example'] = [GOOD]
    labeler = labeler_for(_endpoint('http://vlm.corp.example/v1'), GENERIC_ITEM_PACK)

    dns.answer['vlm.corp.example'] = [METADATA]
    clock['t'] += vlm_factory._RECHECK_S + 1
    with pytest.raises(VlmEndpointDeniedError):
        await labeler._post_chat({}, attempts=1)
    assert connects == []
    await labeler.aclose()


@pytest.mark.asyncio
async def test_a_recheck_that_passes_re_pins_to_its_fresh_answer(
    dns: _Dns, connects: list[tuple[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    clock = {'t': 1000.0}
    monkeypatch.setattr(vlm_factory, 'time', SimpleNamespace(monotonic=lambda: clock['t']))
    dns.answer['vlm.corp.example'] = [GOOD]
    labeler = labeler_for(_endpoint('http://vlm.corp.example/v1'), GENERIC_ITEM_PACK)

    dns.answer['vlm.corp.example'] = [EVIL]  # a legitimately moved container
    clock['t'] += vlm_factory._RECHECK_S + 1
    with pytest.raises(httpx.ConnectError):
        await labeler._post_chat({}, attempts=1)
    assert connects == [(EVIL, 80)]
    await labeler.aclose()


@pytest.mark.asyncio
async def test_a_literal_ip_endpoint_is_not_pinned(
    dns: _Dns, connects: list[tuple[str, int]]
) -> None:
    labeler = labeler_for(_endpoint('http://10.0.0.5:8000/v1'), GENERIC_ITEM_PACK)
    with pytest.raises(httpx.ConnectError):
        await labeler._post_chat({}, attempts=1)
    assert connects == [('10.0.0.5', 8000)]
    await labeler.aclose()


@pytest.mark.asyncio
async def test_the_probe_connects_only_to_its_vetted_address(
    dns: _Dns, connects: list[tuple[str, int]]
) -> None:
    dns.answer['vlm.corp.example'] = [GOOD]
    real_vet = vlm_factory.vet_endpoint_body

    def rebinding_vet(body: VlmEndpointBody, *, is_env: bool) -> Any:
        vetted = real_vet(body, is_env=is_env)
        dns.answer['vlm.corp.example'] = [EVIL]  # rebound right after the check
        return vetted

    from unittest import mock

    with mock.patch.object(vlm_factory, 'vet_endpoint_body', rebinding_vet):
        record = await probe_endpoint(
            VlmEndpointBody(base_url='http://vlm.corp.example/v1', model='m'), api_key='k'
        )
    assert not record.ok
    assert {h for h, _ in connects} == {GOOD}


@pytest.mark.asyncio
async def test_the_probe_refuses_a_host_that_now_resolves_to_a_denied_address(
    dns: _Dns, connects: list[tuple[str, int]]
) -> None:
    dns.answer['vlm.corp.example'] = [METADATA]
    record = await probe_endpoint(
        VlmEndpointBody(base_url='http://vlm.corp.example/v1', model='m'), api_key='k'
    )
    assert record.ok is False
    assert connects == []
    assert any('Refused' in i['message'] for i in record.issues)


@pytest.mark.asyncio
async def test_the_request_keeps_the_endpoint_name_as_url_and_host_header(
    connects: list[tuple[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[httpx.Request] = []

    async def serve(_self: Any, request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', serve)
    pin = AddressPin('vlm.corp.example', [GOOD])
    async with httpx.AsyncClient(transport=PinnedTransport(pin)) as client:
        await client.post('https://vlm.corp.example:8443/v1/chat/completions', json={})
    (req,) = seen
    assert req.url.host == 'vlm.corp.example'
    assert req.headers['host'] == 'vlm.corp.example:8443'


@pytest.mark.asyncio
async def test_a_connect_failure_falls_to_the_next_vetted_address_only(
    connects: list[tuple[str, int]],
) -> None:
    pin = AddressPin('h.example', [GOOD, '10.0.0.7'])
    async with httpx.AsyncClient(transport=PinnedTransport(pin)) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get('http://h.example/')
    assert [h for h, _ in connects] == [GOOD, '10.0.0.7']


@pytest.mark.asyncio
async def test_a_foreign_host_or_an_empty_pin_is_refused_before_any_connect(
    connects: list[tuple[str, int]],
) -> None:
    async with httpx.AsyncClient(
        transport=PinnedTransport(AddressPin('h.example', [GOOD]))
    ) as client:
        with pytest.raises(httpx.ConnectError, match='not the pinned'):
            await client.get('http://other.example/')
    async with httpx.AsyncClient(transport=PinnedTransport(AddressPin('h.example', []))) as client:
        with pytest.raises(httpx.ConnectError, match='no vetted address'):
            await client.get('http://h.example/')
    assert connects == []


@pytest.mark.asyncio
async def test_https_handshake_uses_the_original_name_for_sni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the real httpcore stack: the TCP connect goes to the pinned IP
    and the TLS server_hostname (SNI + certificate check) is the name."""
    tls: list[str | None] = []

    class Stream(httpcore.AsyncNetworkStream):
        async def start_tls(
            self,
            ssl_context: Any,  # noqa: ARG002 - protocol signature
            server_hostname: str | None = None,
            timeout: float | None = None,  # noqa: ARG002 - protocol signature
        ) -> Any:
            tls.append(server_hostname)
            raise httpcore.ConnectError('stop: the handshake target is known')

        async def aclose(self) -> None:
            return None

    seen: list[str] = []

    async def connect_tcp(self: Any, host: str, port: int, **_kw: Any) -> Stream:
        seen.append(host)
        return Stream()

    monkeypatch.setattr('httpcore._backends.anyio.AnyIOBackend.connect_tcp', connect_tcp)
    async with httpx.AsyncClient(
        transport=PinnedTransport(AddressPin('vlm.corp.example', [GOOD]))
    ) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get('https://vlm.corp.example/v1/models')
    assert seen == [GOOD]
    assert tls == ['vlm.corp.example']
