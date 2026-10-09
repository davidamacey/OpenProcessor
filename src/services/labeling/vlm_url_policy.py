"""URL policy for VLM endpoints: syntax, SSRF denials and locality (W9.4, W9.9).

The API has no auth (SECURITY.md), so a caller can make it issue requests
through a registered endpoint. This module is the ONE place that decides
whether a ``base_url`` may be used and where it points:

- :func:`parse_endpoint_url` -- syntax (http/https, no userinfo / query /
  fragment / control characters, valid port), with numeric host forms
  (decimal, hex, octal, short dotted, IPv4-mapped IPv6) normalised so they
  cannot dodge the checks below.
- :func:`url_denial` -- the never-bypassable denials: this stack's own
  non-VLM services (by name, by container name and by the addresses those
  names resolve to), link-local / cloud-metadata / unspecified addresses,
  for the literal host AND for every address the name resolves to.
- :func:`compute_locality` -- ``compose`` / ``host`` / ``private`` /
  ``external`` / ``unknown`` (unresolvable is treated as external, fail
  closed) for the external-images acknowledgement (W9.9).

Everything that touches DNS goes through the module-level ``_resolve`` so
tests can pin it. Blocking DNS: async callers use :func:`acompute_locality`
/ :func:`aurl_denial`.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
import time
import unicodedata
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit, urlunsplit


Locality = Literal['compose', 'host', 'private', 'external', 'unknown']

#: Non-VLM services of this stack (compose service names plus the aliases
#: docs/env use). A VLM endpoint must never be pointed at these: the probe
#: and the labeler would become a request proxy into OpenSearch, Triton,
#: the API itself, the trainer, ... (``tests/curation/test_vlm_url_policy.py``
#: pins that every non-``vlm`` service in docker-compose.yml is listed).
DENIED_INTERNAL_SERVICES: frozenset[str] = frozenset(
    {
        'opensearch',
        'opensearch-dashboards',
        'triton-server',
        'triton-sdk',
        'yolo-api',
        'op-api',
        'segmenter',
        'curation-detection-worker',
        'curation-vlm-worker',
        'curation-auto-label-worker',
        'curation-cluster-refresh',
        'curation-evaluator',
        'curation-mlflow',
        'cropwright',
        'curation-trainer',
        'mlflow',
        'prometheus',
        'grafana',
        'loki',
        'alloy',
        'node-exporter',
        'dcgm-exporter',
    }
)

#: Names that mean "the Docker host".
_HOST_ALIASES = frozenset(
    {'host.docker.internal', 'host.containers.internal', 'gateway.docker.internal'}
)

_METADATA_ADDRESSES = frozenset(
    {
        ipaddress.ip_address('169.254.169.254'),
        ipaddress.ip_address('fd00:ec2::254'),
        ipaddress.ip_address('100.100.100.200'),
        ipaddress.ip_address('192.0.0.192'),
    }
)
_CGNAT = ipaddress.ip_network('100.64.0.0/10')

_NUMERIC_HOST = re.compile(r'^[0-9a-fx.]+$', re.IGNORECASE)
_BAD_URL_CHARS = re.compile(r'[\s\\\x00-\x1f\x7f]')


@dataclass(frozen=True)
class ParsedUrl:
    scheme: str
    host: str  # lowercase, no brackets, no trailing dot
    port: int | None
    base_url: str  # the input with any trailing '/' stripped


class UrlSyntaxError(ValueError):
    """The URL is not an acceptable endpoint URL (``vlm_url_invalid``)."""


def _resolve(host: str) -> list[str]:
    """Every address ``host`` resolves to (empty when it does not). The one
    DNS call in this module."""
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return []
    return [str(info[4][0]).split('%', 1)[0] for info in infos]


def strip_userinfo(url: str) -> str:
    """``url`` without any ``user:password@`` and without query / fragment:
    the one redaction for an endpoint URL that is served, logged or used (a
    credential in a URL must never leave the process)."""
    parts = urlsplit(url)
    host = parts.hostname or ''
    if ':' in host:
        host = f'[{host}]'
    netloc = f'{host}:{parts.port}' if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path, '', ''))


def parse_endpoint_url(base_url: str) -> ParsedUrl:
    """Validate ``base_url``'s syntax; raise :class:`UrlSyntaxError`."""
    text = (base_url or '').strip()
    if not text or _BAD_URL_CHARS.search(text):
        raise UrlSyntaxError('base_url must be a plain http(s) URL without whitespace')
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError as exc:
        raise UrlSyntaxError(f'base_url is not a valid URL: {exc}') from exc
    if parts.scheme not in ('http', 'https'):
        raise UrlSyntaxError('base_url must use http or https')
    if '@' in parts.netloc:
        raise UrlSyntaxError('base_url must not carry credentials (userinfo)')
    if parts.query or parts.fragment or '?' in text or '#' in text:
        raise UrlSyntaxError('base_url must not carry a query or fragment')
    # NFKC: a resolver folds full-width letters to ASCII,
    # so the policy must compare the folded name.
    host = unicodedata.normalize('NFKC', parts.hostname or '').lower().rstrip('.')
    if not host or '%' in host:
        raise UrlSyntaxError('base_url needs a host')
    return ParsedUrl(parts.scheme, host, port, text.rstrip('/'))


def _literal_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """``host`` as an IP address when it is one in ANY notation an HTTP
    client's resolver accepts (dotted, decimal, hex, octal, short forms,
    IPv6, IPv4-mapped IPv6); ``None`` for a name."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is None and _NUMERIC_HOST.match(host):
        try:
            ip = ipaddress.IPv4Address(socket.inet_aton(host))
        except (OSError, ValueError):
            ip = None
    return None if ip is None else _unwrap_v4(ip)


def _unwrap_v4(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """The IPv4 address an IPv6 one carries (IPv4-mapped ``::ffff:a.b.c.d``
    and 6to4 ``2002:ab:cd::``), so every policy decision is made on the v4
    address a router would actually deliver to."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return ip.ipv4_mapped
        if ip.sixtofour is not None:
            return ip.sixtofour
    return ip


def _ip_of(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return _unwrap_v4(ipaddress.ip_address(text))
    except ValueError:
        return None


def _never_reachable(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Link-local, cloud-metadata, unspecified, multicast and reserved
    addresses: no VLM lives there, and the metadata ones are the classic
    SSRF target."""
    if ip.is_loopback:
        # Loopback is a legitimate (private) target, e.g. a VLM on the same
        # host; ``::1`` sits inside the reserved ``::/8`` block, so it has to
        # be excluded before the reserved check.
        return False
    return (
        ip in _METADATA_ADDRESSES
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_multicast
        or ip.is_reserved
    )


def _is_private(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_loopback or ip.is_private:
        return True
    return isinstance(ip, ipaddress.IPv4Address) and ip in _CGNAT


def _compose_project() -> str:
    return os.environ.get('COMPOSE_PROJECT_NAME', '').strip() or 'openprocessor'


def _is_denied_service_name(host: str) -> bool:
    if host in DENIED_INTERNAL_SERVICES:
        return True
    project = _compose_project()
    # `${COMPOSE_PROJECT_NAME}-<anything>` is a container of this stack;
    # only the VLM container is a legitimate target.
    return host.startswith(f'{project}-') and host != f'{project}-vlm'


_DENIED_IP_CACHE: tuple[float, frozenset[str]] | None = None
_DENIED_IP_TTL_S = 30.0


def _denied_service_addresses() -> frozenset[str]:
    """The addresses this stack's non-VLM services currently resolve to,
    so a URL that spells one out (or a DNS alias of it) is caught too."""
    global _DENIED_IP_CACHE  # noqa: PLW0603 - short-lived process cache
    now = time.monotonic()
    if _DENIED_IP_CACHE is not None and now - _DENIED_IP_CACHE[0] < _DENIED_IP_TTL_S:
        return _DENIED_IP_CACHE[1]
    found: set[str] = set()
    for name in DENIED_INTERNAL_SERVICES:
        for addr in _resolve(name):
            ip = _ip_of(addr)
            if ip is not None:
                found.add(str(ip))
    _DENIED_IP_CACHE = (now, frozenset(found))
    return _DENIED_IP_CACHE[1]


def reset_policy_caches() -> None:
    """Test-only."""
    global _DENIED_IP_CACHE, _GATEWAY_CACHE  # noqa: PLW0603
    _DENIED_IP_CACHE = None
    _GATEWAY_CACHE = None
    _LOCALITY_CACHE.clear()


_GATEWAY_CACHE: frozenset[str] | None = None


def _docker_gateway_addresses() -> frozenset[str]:
    """The container's default-route gateway (the Docker host's bridge
    address), read from ``/proc/net/route``; empty when unreadable."""
    global _GATEWAY_CACHE  # noqa: PLW0603 - immutable for the container's life
    if _GATEWAY_CACHE is not None:
        return _GATEWAY_CACHE
    found: set[str] = set()
    try:
        with open('/proc/net/route', encoding='ascii') as fh:
            next(fh, None)
            for line in fh:
                cols = line.split()
                if len(cols) >= 3 and cols[1] == '00000000':
                    raw = int(cols[2], 16).to_bytes(4, 'little')
                    found.add(str(ipaddress.IPv4Address(raw)))
    except (OSError, ValueError):
        pass
    _GATEWAY_CACHE = frozenset(found)
    return _GATEWAY_CACHE


@dataclass(frozen=True)
class UrlDenial:
    #: ``vlm_url_denied_internal_service`` (one of this stack's services) or
    #: ``vlm_url_denied_address`` (link-local / metadata / unspecified).
    code: Literal['vlm_url_denied_internal_service', 'vlm_url_denied_address']
    reason: str

    def __str__(self) -> str:
        return self.reason


def url_denial(base_url: str) -> UrlDenial | None:
    """Why this URL must never be used, or ``None``. Syntax errors are
    :func:`parse_endpoint_url`'s job; call it first.

    Checked for the literal host and for every address a name resolves to
    (a name pointing at the metadata service or at one of this stack's
    services is refused however it is spelled)."""
    parsed = parse_endpoint_url(base_url)
    host = parsed.host
    service_code: Literal['vlm_url_denied_internal_service'] = 'vlm_url_denied_internal_service'
    if _is_denied_service_name(host):
        return UrlDenial(
            service_code, f"{host!r} is one of this deployment's own services, not a VLM"
        )
    literal = _literal_ip(host)
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    if literal is not None:
        addresses.append(literal)
    elif host not in _HOST_ALIASES:
        addresses.extend(ip for a in _resolve(host) if (ip := _ip_of(a)) is not None)
    denied_ips = _denied_service_addresses()
    for ip in addresses:
        if _never_reachable(ip):
            return UrlDenial(
                'vlm_url_denied_address',
                f'{host!r} resolves to {ip}, a link-local / metadata / unspecified address',
            )
        if str(ip) in denied_ips:
            return UrlDenial(
                service_code,
                f"{host!r} resolves to {ip}, an address of one of this deployment's own services",
            )
    return None


def compute_locality(base_url: str) -> Locality:
    """Where the endpoint lives, for the external-images acknowledgement.

    ``compose``: a bare service name on the stack network. ``host``: the
    Docker host. ``private``: loopback / RFC 1918 / ULA / CGNAT. ``external``:
    anything else, including a name that ALSO resolves to a public address
    (a DNS answer mixing private and public addresses is external). ``unknown``:
    the name does not resolve -- treated as external by callers.
    """
    host = parse_endpoint_url(base_url).host
    if host in _HOST_ALIASES:
        return 'host'
    literal = _literal_ip(host)
    if literal is not None:
        if str(literal) in _docker_gateway_addresses():
            return 'host'
        return 'private' if _is_private(literal) else 'external'
    addresses = [ip for a in _resolve(host) if (ip := _ip_of(a)) is not None]
    if not addresses:
        return 'unknown'
    if not all(_is_private(ip) for ip in addresses):
        return 'external'
    if any(str(ip) in _docker_gateway_addresses() for ip in addresses):
        return 'host'
    if '.' not in host and host != 'localhost':
        return 'compose'
    return 'private'


def sends_images_externally(locality: Locality) -> bool:
    return locality in ('external', 'unknown')


def external_warning(base_url: str, locality: Locality) -> str | None:
    if not sends_images_externally(locality):
        return None
    host = parse_endpoint_url(base_url).host
    if locality == 'unknown':
        return (
            f'{host} could not be resolved, so crops are treated as sent to a service '
            'outside this deployment.'
        )
    return f'Crops are sent to a service outside this deployment ({host}).'


_LOCALITY_CACHE: dict[str, tuple[float, Locality]] = {}
_LOCALITY_TTL_S = 30.0


async def acompute_locality(base_url: str, *, cached: bool = False) -> Locality:
    """:func:`compute_locality` off the event loop. ``cached=True`` (listings
    only: one lookup per endpoint per few seconds) answers from a short TTL
    cache; every gate (activation, a run, validation) resolves fresh, so a
    DNS answer that changes is never trusted from memory."""
    if not cached:
        return await asyncio.to_thread(compute_locality, base_url)
    now = time.monotonic()
    hit = _LOCALITY_CACHE.get(base_url)
    if hit is not None and now - hit[0] < _LOCALITY_TTL_S:
        return hit[1]
    value = await asyncio.to_thread(compute_locality, base_url)
    if len(_LOCALITY_CACHE) > 256:
        _LOCALITY_CACHE.clear()
    _LOCALITY_CACHE[base_url] = (now, value)
    return value


async def aurl_denial(base_url: str) -> UrlDenial | None:
    return await asyncio.to_thread(url_denial, base_url)


__all__ = [
    'DENIED_INTERNAL_SERVICES',
    'Locality',
    'ParsedUrl',
    'UrlDenial',
    'UrlSyntaxError',
    'acompute_locality',
    'aurl_denial',
    'compute_locality',
    'external_warning',
    'parse_endpoint_url',
    'reset_policy_caches',
    'sends_images_externally',
    'strip_userinfo',
    'url_denial',
]
