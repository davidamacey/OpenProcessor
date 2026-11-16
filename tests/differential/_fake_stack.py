"""A stack that is a function: ``httpx.MockTransport`` under a client, recording what it was sent.

httpx documents MockTransport for exactly this (https://www.python-httpx.org/advanced/transports/#mock-transports);
the runner sends with httpx, so no socket is opened.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx


Answer = Callable[[httpx.Request], httpx.Response]


def json_answer(body: Any, status: int = 200, raw: str | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        content=(raw if raw is not None else json.dumps(body)).encode(),
        headers={'content-type': 'application/json'},
    )


@dataclass
class FakeStack:
    name: str
    answer: Answer
    seen: list[httpx.Request] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        return self.answer(request)

    def client(self) -> httpx.Client:
        return httpx.Client(
            base_url=f'http://{self.name}.test',
            transport=httpx.MockTransport(self.handle),
            timeout=5.0,
        )

    def requests(self) -> list[tuple[str, str, tuple[tuple[str, str], ...], bytes]]:
        """What was sent, without the parts that differ by construction."""
        sent = []
        for request in self.seen:
            headers = tuple(
                sorted(
                    (k, v)
                    for k, v in request.headers.items()
                    if k not in {'host', 'user-agent', 'content-type', 'content-length'}
                )
            )
            sent.append((request.method, request.url.raw_path.decode(), headers, request.content))
        return sent
