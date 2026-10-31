"""Pin a VLM connection to the addresses that passed the URL policy.

The URL policy (``vlm_url_policy.py``) judges the addresses a host name
resolves to; a plain HTTP client would then resolve the name AGAIN to connect,
and a DNS answer that changed in between (rebinding) would land on an address
nobody judged. :class:`PinnedTransport` closes that window: it never resolves
anything. It connects to one of the :class:`AddressPin`'s vetted addresses
and leaves the request untouched -- the ``Host`` header stays the endpoint's
name and, for https, the TLS handshake sends and verifies the original host
name (httpcore starts TLS with the origin host, not the address it dialled),
so certificate checking is exactly what it was.

Fail closed: a connect to any host but the pinned one, or with no vetted
address, is refused (:class:`httpx.ConnectError`) before a socket opens.
A literal-IP endpoint has nothing to re-resolve and is not pinned.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import httpcore
import httpx


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class AddressPin:
    """The vetted addresses of one endpoint host. Mutable on purpose: the
    labeler's periodic re-check swaps in a fresh answer (a restarted
    container gets a new address) -- but only an answer that passed the same
    policy, so the set never holds an unjudged address."""

    def __init__(self, host: str, addresses: Sequence[str]) -> None:
        self.host = host
        self._lock = threading.Lock()
        self._addresses: tuple[str, ...] = tuple(addresses)

    @property
    def addresses(self) -> tuple[str, ...]:
        with self._lock:
            return self._addresses

    def update(self, addresses: Sequence[str]) -> None:
        with self._lock:
            self._addresses = tuple(addresses)


class _PinnedBackend(httpcore.AsyncNetworkBackend):
    """httpcore's network layer with the name lookup replaced by the pin.

    Only the TCP connect is redirected: the request, its ``Host`` header, the
    pool key and the TLS server name (SNI and certificate check) all stay the
    endpoint's name, because httpcore hands this method the origin host and
    starts TLS afterwards with that same name."""

    def __init__(self, pin: AddressPin) -> None:
        self._pin = pin
        self._inner = httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        if host != self._pin.host:
            raise httpcore.ConnectError(f'refused: {host!r} is not the pinned endpoint host')
        failure: Exception | None = None
        for address in self._pin.addresses:
            try:
                return await self._inner.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                failure = exc  # try the next vetted address; never another one
        raise failure or httpcore.ConnectError(f'refused: no vetted address for {host!r}')

    async def connect_unix_socket(
        self,
        path: str,  # noqa: ARG002 - protocol signature
        timeout: float | None = None,  # noqa: ARG002 - protocol signature
        socket_options: Iterable[Any] | None = None,  # noqa: ARG002 - protocol signature
    ) -> httpcore.AsyncNetworkStream:
        raise httpcore.ConnectError('refused: a pinned endpoint is never a unix socket')

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class PinnedTransport(httpx.AsyncHTTPTransport):
    """``httpx.AsyncHTTPTransport`` whose connections go only to ``pin``'s
    addresses. Everything else (limits, TLS verification, HTTP/1.1) is the
    stock transport's; only its connection pool is rebuilt on the pinned
    network backend."""

    def __init__(self, pin: AddressPin, *, limits: httpx.Limits | None = None) -> None:
        limits = limits or httpx.Limits()
        super().__init__(limits=limits)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=httpx.create_ssl_context(),
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            http1=True,
            http2=False,
            network_backend=_PinnedBackend(pin),
        )


__all__ = ['AddressPin', 'PinnedTransport']
