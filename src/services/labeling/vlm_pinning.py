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

import contextlib
import threading
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any

import httpcore
import httpx


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Iterator, Sequence


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
        addresses = self._pin.addresses
        if not addresses:
            raise httpcore.ConnectError(f'refused: no vetted address for {host!r}')
        failure: Exception | None = None
        for address in addresses:
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
            except (OSError, ValueError) as exc:
                # anyio raises a bare ValueError when every attempt was
                # cancelled before recording an error (it builds an empty
                # ExceptionGroup); an unmapped error would escape the retry
                # code's httpx types.
                failure = httpcore.ConnectError(f'could not connect to {address}: {exc}')
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


_EXCEPTION_MAP: dict[type[Exception], type[httpx.HTTPError]] = {
    httpcore.TimeoutException: httpx.TimeoutException,
    httpcore.ConnectTimeout: httpx.ConnectTimeout,
    httpcore.ReadTimeout: httpx.ReadTimeout,
    httpcore.WriteTimeout: httpx.WriteTimeout,
    httpcore.PoolTimeout: httpx.PoolTimeout,
    httpcore.NetworkError: httpx.NetworkError,
    httpcore.ConnectError: httpx.ConnectError,
    httpcore.ReadError: httpx.ReadError,
    httpcore.WriteError: httpx.WriteError,
    httpcore.ProxyError: httpx.ProxyError,
    httpcore.UnsupportedProtocol: httpx.UnsupportedProtocol,
    httpcore.ProtocolError: httpx.ProtocolError,
    httpcore.LocalProtocolError: httpx.LocalProtocolError,
    httpcore.RemoteProtocolError: httpx.RemoteProtocolError,
}


@contextlib.contextmanager
def _map_httpcore_exceptions() -> Iterator[None]:
    """Re-raise an httpcore error as the most specific httpx one, so the retry
    code that matches on httpx types (``_RETRYABLE_HTTPX_EXCEPTIONS``) sees
    what the stock transport would have raised."""
    try:
        yield
    except Exception as exc:
        mapped: type[httpx.HTTPError] | None = None
        for source, target in _EXCEPTION_MAP.items():
            if isinstance(exc, source) and (mapped is None or issubclass(target, mapped)):
                mapped = target
        if mapped is None:
            raise
        raise mapped(str(exc)) from exc


class _ResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream: AsyncIterable[bytes]) -> None:
        self._stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        with _map_httpcore_exceptions():
            async for part in self._stream:
                yield part

    async def aclose(self) -> None:
        close = getattr(self._stream, 'aclose', None)
        if close is not None:
            await close()


class PinnedTransport(httpx.AsyncBaseTransport):
    """An ``httpx`` transport whose connections go only to ``pin``'s
    addresses. Built on public API only: it owns an
    ``httpcore.AsyncConnectionPool`` on the pinned network backend and does
    the request/response translation the stock transport does, so there is no
    path by which a request could reach a socket unpinned. TLS verification
    (system/``SSL_CERT_FILE`` trust, hostname check) and the connection limits
    are the stock defaults; HTTP/1.1 only."""

    def __init__(self, pin: AddressPin, *, limits: httpx.Limits | None = None) -> None:
        limits = limits or httpx.Limits()
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=httpx.create_ssl_context(),
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            http1=True,
            http2=False,
            network_backend=_PinnedBackend(pin),
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if not isinstance(request.stream, httpx.AsyncByteStream):
            raise httpx.RequestError('a pinned transport needs an async request body')
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        with _map_httpcore_exceptions():
            response = await self._pool.handle_async_request(core_request)
        if not isinstance(response.stream, AsyncIterable):
            raise httpx.ReadError('the pinned pool returned a response without a body stream')
        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_ResponseStream(response.stream),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


__all__ = ['AddressPin', 'PinnedTransport']
