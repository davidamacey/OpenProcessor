"""Pick the Triton server for each call (research GPU S-15, S-23).

The servers come from the active serving profile (one server by default). A call goes to the
server with the fewest outstanding requests *in this process* among those whose
``serving_reports`` say the model is ready; the choice and the count happen with no ``await``
between them, so concurrent callers on one event loop spread evenly without a lock. Counts are
per process on purpose: each API worker balances its own calls, and the ``op_triton_router_outstanding``
gauge sums them across workers (prometheus_client multiprocess mode,
https://prometheus.github.io/client_python/multiprocess/).

The image-prep call of an image and every follow-up for that image must hit the same server
(its tensors sit in that card's memory): ``affinity(key)`` opens a scope in which the first
routed call pins the server and the later ones reuse it. A pinned server that stops reporting
the model refuses the call rather than moving it. Failing closed is the rule everywhere: a
profile with servers but no ready report for the model raises ``NoServerError``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from prometheus_client import Gauge


OUTSTANDING = Gauge(
    'op_triton_router_outstanding',
    'Triton calls this process has in flight, by server.',
    ['server'],
    multiprocess_mode='livesum',
)


class NoServerError(RuntimeError):
    """No server reports the model ready (or the pinned server no longer does)."""


@dataclass(frozen=True)
class Route:
    server: str
    url: str


class TritonRouter:
    def __init__(self, servers: Mapping[str, str]) -> None:
        if not servers:
            msg = 'a router needs at least one server'
            raise ValueError(msg)
        self._servers = dict(servers)
        self._outstanding = dict.fromkeys(self._servers, 0)
        # None: no reports are consulted (the single default server, as before the router).
        self._ready: dict[str, frozenset[str]] | None = {}
        self._pins: dict[str, str | None] = {}

    @classmethod
    def single(cls, url: str) -> TritonRouter:
        """The default install: one server, always eligible, exactly the behaviour before routing."""
        router = cls({'default': url})
        router._ready = None
        return router

    def urls(self) -> list[str]:
        """The gRPC address of every server, in profile order."""
        return list(self._servers.values())

    @property
    def follows_reports(self) -> bool:
        """False for the single default server, which is always eligible and needs no reports."""
        return self._ready is not None

    def update_ready(self, ready: Mapping[str, Iterable[str]]) -> None:
        """Replace the per-server ready models (from ``serving_reports``); unknown servers are ignored."""
        self._ready = {server: frozenset(ready.get(server, ())) for server in self._servers}

    def outstanding(self) -> dict[str, int]:
        return dict(self._outstanding)

    def _eligible(self, server: str, model: str) -> bool:
        return self._ready is None or model in self._ready.get(server, frozenset())

    def _pick(self, model: str, key: str | None) -> str:
        pinned = self._pins.get(key) if key is not None else None
        if pinned is not None:
            if not self._eligible(pinned, model):
                msg = f'server {pinned!r} pinned for {key!r} does not report {model!r} ready'
                raise NoServerError(msg)
            return pinned
        candidates = [s for s in self._servers if self._eligible(s, model)]
        if not candidates:
            msg = f'no server reports {model!r} ready'
            raise NoServerError(msg)
        return min(candidates, key=lambda s: self._outstanding[s])

    @contextmanager
    def affinity(self, key: str) -> Iterator[None]:
        """Calls routed with ``affinity=key`` inside this scope share the first call's server."""
        if key in self._pins:
            msg = f'affinity scope {key!r} is already open'
            raise ValueError(msg)
        self._pins[key] = None
        try:
            yield
        finally:
            del self._pins[key]

    @contextmanager
    def route(self, model: str, *, affinity: str | None = None) -> Iterator[Route]:
        """Count one call against the chosen server for as long as the ``with`` body runs."""
        if affinity is not None and affinity not in self._pins:
            msg = f'affinity scope {affinity!r} is not open'
            raise ValueError(msg)
        server = self._pick(model, affinity)
        if affinity is not None:
            self._pins[affinity] = server
        self._outstanding[server] += 1
        OUTSTANDING.labels(server).set(self._outstanding[server])
        try:
            yield Route(server, self._servers[server])
        finally:
            self._outstanding[server] -= 1
            OUTSTANDING.labels(server).set(self._outstanding[server])
