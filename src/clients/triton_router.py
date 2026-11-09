"""Pick the Triton server for each call (research GPU S-15, S-23).

The servers come from the active serving profile (one server by default). A call goes to the
server with the fewest outstanding requests *in this process* among those whose
``serving_reports`` say the model is ready; the choice and the count happen with no ``await``
between them, so concurrent callers on one event loop spread evenly without a lock. Counts are
per process on purpose: each API worker balances its own calls, and the ``op_triton_router_outstanding``
gauge sums them across workers (prometheus_client multiprocess mode,
https://prometheus.github.io/client_python/multiprocess/).

Failing closed is the rule: a profile with servers but no ready report for the model raises
``NoServerError``. Every call carries its whole input today, so no call depends on what an earlier
one left in a server's memory; the per-image pinning that a resident-frame image-prep model needs
(research GPU S-15) comes with that model and its caller.
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

    def _pick(self, model: str) -> str:
        candidates = [s for s in self._servers if self._eligible(s, model)]
        if not candidates:
            msg = f'no server reports {model!r} ready'
            raise NoServerError(msg)
        return min(candidates, key=lambda s: self._outstanding[s])

    @contextmanager
    def route(self, model: str) -> Iterator[Route]:
        """Count one call against the chosen server for as long as the ``with`` body runs."""
        server = self._pick(model)
        self._outstanding[server] += 1
        OUTSTANDING.labels(server).set(self._outstanding[server])
        try:
            yield Route(server, self._servers[server])
        finally:
            self._outstanding[server] -= 1
            OUTSTANDING.labels(server).set(self._outstanding[server])
