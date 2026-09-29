"""The VLM endpoint URL policy (W9.4 / W9.9): syntax, the SSRF denials that
can never be bypassed, and the locality classification behind the
external-images acknowledgement. Every DNS answer here is pinned through
``_resolve``; nothing touches the network."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import src.services.labeling.vlm_url_policy as policy
from src.services.labeling.vlm_url_policy import (
    DENIED_INTERNAL_SERVICES,
    UrlSyntaxError,
    compute_locality,
    parse_endpoint_url,
    sends_images_externally,
    url_denial,
)


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def dns(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Pin name resolution: ``dns['name'] = ['1.2.3.4']``. Anything not in
    the table does not resolve."""
    table: dict[str, list[str]] = {}
    monkeypatch.setattr(policy, '_resolve', lambda host: list(table.get(host, [])))
    monkeypatch.setattr(policy, '_docker_gateway_addresses', lambda: frozenset({'172.17.0.1'}))
    policy.reset_policy_caches()
    return table


# ---- syntax ------------------------------------------------------------------


@pytest.mark.parametrize(
    'url',
    [
        '',
        '   ',
        'ftp://vlm:8000/v1',
        'file:///etc/passwd',
        'gopher://vlm/',
        'vlm:8000/v1',
        'http://user:pass@vlm:8000/v1',
        'http://user@vlm:8000/v1',
        'http://vlm:8000/v1?key=1',
        'http://vlm:8000/v1#frag',
        'http://vlm:8000/v1 ',
        'http://vlm:8000/v1\nHost: evil',
        'http://vlm\\@evil/v1',
        'http://vlm:notaport/v1',
        'http://vlm:99999/v1',
        'http:///v1',
        'http://[fe80::1%25eth0]/v1',
    ],
)
def test_syntax_rejects(url: str) -> None:
    if url == 'http://vlm:8000/v1 ':
        # trailing whitespace is stripped, not an error
        assert parse_endpoint_url(url).host == 'vlm'
        return
    with pytest.raises(UrlSyntaxError):
        parse_endpoint_url(url)


def test_syntax_normalises_the_stored_form() -> None:
    parsed = parse_endpoint_url('HTTP://VLM.Example.COM.:8000/v1/')
    assert (parsed.scheme, parsed.host, parsed.port) == ('http', 'vlm.example.com', 8000)
    assert parsed.base_url == 'HTTP://VLM.Example.COM.:8000/v1'


# ---- the denials that no flag bypasses --------------------------------------


@pytest.mark.parametrize(
    'host',
    [
        '169.254.169.254',  # AWS/GCP/Azure metadata
        '2852039166',  # the same address, decimal
        '0xa9fea9fe',  # hex
        '0251.0376.0251.0376',  # octal dotted
        '169.254.43518',  # short dotted
        '[::ffff:169.254.169.254]',  # IPv4-mapped IPv6
        '[::ffff:a9fe:a9fe]',
        '[fe80::1]',  # IPv6 link-local
        '[fd00:ec2::254]',  # AWS IPv6 metadata
        '100.100.100.200',  # Alibaba metadata
        '192.0.0.192',  # Oracle metadata
        '0.0.0.0',
        '0',
        '[::]',
        '[::a9fe:a9fe]',  # deprecated IPv4-compatible IPv6 form
        '224.0.0.1',  # multicast
        '169.254.1.1',  # any link-local
    ],
)
def test_never_reachable_addresses_are_denied_in_any_notation(host: str) -> None:
    denial = url_denial(f'http://{host}:8000/v1')
    assert denial is not None
    assert denial.code == 'vlm_url_denied_address'


@pytest.mark.parametrize('service', sorted(DENIED_INTERNAL_SERVICES))
def test_every_internal_service_name_is_denied(service: str) -> None:
    denial = url_denial(f'http://{service}:9200/v1')
    assert denial is not None
    assert denial.code == 'vlm_url_denied_internal_service'


def test_the_denied_service_list_is_every_non_vlm_compose_service() -> None:
    """A service added to docker-compose.yml without being listed here would
    silently become a request-proxy target."""
    compose = yaml.safe_load((REPO / 'docker-compose.yml').read_text())
    services = set(compose['services'])
    # `vlm` is the one legitimate target; `op-api` is the documented alias.
    assert 'vlm' in services
    missing = services - {'vlm'} - DENIED_INTERNAL_SERVICES
    assert missing == set(), f'not denied as VLM endpoints: {sorted(missing)}'


def test_container_names_of_this_stack_are_denied_except_the_vlm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('COMPOSE_PROJECT_NAME', 'acme')
    for name in ('acme-api', 'acme-opensearch', 'acme-triton', 'acme-anything'):
        denial = url_denial(f'http://{name}:8000/v1')
        assert denial is not None, name
        assert denial.code == 'vlm_url_denied_internal_service'
    assert url_denial('http://acme-vlm:8000/v1') is None
    assert url_denial('http://other-api:8000/v1') is None


def test_a_name_that_resolves_to_metadata_is_denied(dns: dict[str, list[str]]) -> None:
    dns['innocent.example.com'] = ['169.254.169.254']
    denial = url_denial('http://innocent.example.com/v1')
    assert denial is not None
    assert denial.code == 'vlm_url_denied_address'


def test_dns_rebinding_shaped_answers_are_denied_when_any_address_is_bad(
    dns: dict[str, list[str]],
) -> None:
    """One good answer next to a bad one is still denied: the resolver may
    hand the HTTP client either."""
    dns['rebind.example.com'] = ['93.184.216.34', '169.254.169.254']
    assert url_denial('http://rebind.example.com/v1') is not None
    dns['rebind6.example.com'] = ['93.184.216.34', '::ffff:169.254.169.254']
    assert url_denial('http://rebind6.example.com/v1') is not None


def test_an_alias_of_an_internal_service_address_is_denied(dns: dict[str, list[str]]) -> None:
    dns['opensearch'] = ['172.18.0.5']
    dns['sneaky.example.com'] = ['172.18.0.5']
    denial = url_denial('http://sneaky.example.com/v1')
    assert denial is not None
    assert denial.code == 'vlm_url_denied_internal_service'
    # ...and the literal address itself.
    literal = url_denial('http://172.18.0.5:9200/v1')
    assert literal is not None
    assert literal.code == 'vlm_url_denied_internal_service'
    mapped = url_denial('http://[::ffff:172.18.0.5]:9200/v1')
    assert mapped is not None
    assert mapped.code == 'vlm_url_denied_internal_service'
    # The address of a service that is NOT denied is fine.
    dns['vlm'] = ['172.18.0.9']
    assert url_denial('http://vlm:8000/v1') is None


@pytest.mark.parametrize(
    'url',
    [
        'http://vlm:8000/v1',
        'http://localhost:8000/v1',
        'http://127.0.0.1:8000/v1',
        'http://[::1]:8000/v1',
        'http://10.1.2.3:8000/v1',
        'https://api.example.com/v1',
        'http://host.docker.internal:8000/v1',
    ],
)
def test_ordinary_endpoints_are_not_denied(url: str, dns: dict[str, list[str]]) -> None:
    dns['api.example.com'] = ['93.184.216.34']
    dns['vlm'] = ['172.18.0.9']
    assert url_denial(url) is None


# ---- locality ----------------------------------------------------------------


def test_locality_classification(dns: dict[str, list[str]]) -> None:
    dns['vlm'] = ['172.18.0.9']
    dns['gpu-box.lan'] = ['192.168.1.20']
    dns['api.example.com'] = ['93.184.216.34']
    dns['dockerhost.example'] = ['172.17.0.1']
    assert compute_locality('http://vlm:8000/v1') == 'compose'
    assert compute_locality('http://host.docker.internal:8000/v1') == 'host'
    assert compute_locality('http://172.17.0.1:8000/v1') == 'host'
    assert compute_locality('http://dockerhost.example:8000/v1') == 'host'
    assert compute_locality('http://gpu-box.lan:8000/v1') == 'private'
    assert compute_locality('http://127.0.0.1:8000/v1') == 'private'
    assert compute_locality('http://[fd12::1]:8000/v1') == 'private'
    assert compute_locality('http://100.64.1.1:8000/v1') == 'private'  # CGNAT
    assert compute_locality('http://[::ffff:10.0.0.5]:8000/v1') == 'private'
    assert compute_locality('https://api.example.com/v1') == 'external'
    assert compute_locality('http://8.8.8.8/v1') == 'external'
    assert compute_locality('http://never-resolves.example/v1') == 'unknown'


def test_a_name_with_one_public_address_is_external(dns: dict[str, list[str]]) -> None:
    dns['mixed.example.com'] = ['10.0.0.5', '93.184.216.34']
    assert compute_locality('http://mixed.example.com/v1') == 'external'


@pytest.mark.parametrize(
    'host',
    ['16909060', '0x01020304', '1.2.3.4', '[::ffff:1.2.3.4]', '0x7f000001.1'],
)
def test_numeric_hosts_are_classified_by_their_real_address(
    host: str, dns: dict[str, list[str]]
) -> None:
    locality = compute_locality(f'http://{host}/v1')
    if host == '0x7f000001.1':
        # not a valid IPv4 spelling: treated as a name, which does not resolve
        assert locality == 'unknown'
    else:
        assert locality == 'external'


def test_decimal_loopback_is_private_not_external(dns: dict[str, list[str]]) -> None:
    assert compute_locality('http://2130706433:8000/v1') == 'private'


def test_unknown_and_external_send_images_externally() -> None:
    assert sends_images_externally('external')
    assert sends_images_externally('unknown')
    for local in ('compose', 'host', 'private'):
        assert not sends_images_externally(local)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_gates_resolve_fresh_while_listings_may_cache(dns: dict[str, list[str]]) -> None:
    dns['flip.example.com'] = ['10.0.0.5']
    assert await policy.acompute_locality('http://flip.example.com/v1', cached=True) == 'private'
    dns['flip.example.com'] = ['93.184.216.34']
    # a listing may serve the cached answer...
    assert await policy.acompute_locality('http://flip.example.com/v1', cached=True) == 'private'
    # ...a gate never does
    assert await policy.acompute_locality('http://flip.example.com/v1') == 'external'
