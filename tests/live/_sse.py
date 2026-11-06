"""A background Server-Sent Events reader for the live checks (httpx streaming, no extra library).

Spec: https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation
Frames are separated by a blank line; ``:`` lines are comments (keep-alives); ``data`` is JSON here.
"""

from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    from collections.abc import Callable


class SseCollector:
    """Reads one SSE stream on a daemon thread; ``events`` grows as frames arrive."""

    def __init__(self, url: str, *, last_event_id: str | None = None) -> None:
        self._url = url
        self._headers = {'Accept': 'text/event-stream'}
        if last_event_id is not None:
            self._headers['Last-Event-ID'] = last_event_id
        self.events: list[dict[str, Any]] = []
        self.status: int | None = None
        self.error: str | None = None
        self._connected = threading.Event()
        self._closing = threading.Event()
        self._response: httpx.Response | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self, *, connect_timeout_s: float = 30.0) -> SseCollector:
        self._thread.start()
        if not self._connected.wait(connect_timeout_s):
            self.close()
            raise TimeoutError(f'no SSE response from {self._url} in {connect_timeout_s} s')
        if self.status != 200:
            raise RuntimeError(f'SSE {self._url} answered {self.status}: {self.error}')
        return self

    def _run(self) -> None:
        frame: dict[str, Any] = {}
        try:
            with (
                httpx.Client(timeout=httpx.Timeout(10.0, read=60.0)) as client,
                client.stream('GET', self._url, headers=self._headers) as response,
            ):
                self._response = response
                self.status = response.status_code
                if response.status_code != 200:
                    self.error = response.read().decode(errors='replace')[:500]
                self._connected.set()
                if response.status_code != 200:
                    return
                for line in response.iter_lines():
                    if self._closing.is_set():
                        return
                    if line == '':
                        if frame:
                            self.events.append(frame)
                            frame = {}
                    elif not line.startswith(':'):
                        name, _, value = line.partition(':')
                        value = value.removeprefix(' ')
                        if name == 'data':
                            frame['data'] = json.loads(value)
                        elif name in ('id', 'event'):
                            frame[name] = value
        except (httpx.HTTPError, OSError, RuntimeError) as exc:
            if not self._closing.is_set():
                self.error = f'{type(exc).__name__}: {exc}'
        finally:
            self._connected.set()

    def wait_for(self, predicate: Callable[[list[dict[str, Any]]], bool], timeout_s: float) -> bool:
        """True once ``predicate(events)`` holds, False after ``timeout_s``."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate(list(self.events)):
                return True
            time.sleep(0.2)
        return predicate(list(self.events))

    def for_job(self, job_id: str) -> list[dict[str, Any]]:
        return [e for e in list(self.events) if e.get('data', {}).get('job_id') == job_id]

    def close(self) -> None:
        self._closing.set()
        if self._response is not None:
            self._response.close()
        self._thread.join(timeout=5)
