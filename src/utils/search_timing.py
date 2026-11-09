"""Per-request stage timing for the curation kNN search routes (#414).

``stage_timing.py`` is ingest-scoped (closed ingest stage set, ``op_pipeline_stage_seconds``).
Search needs its own closed set and metric. A route binds a :class:`SearchTimer`; the service
and the encoder record spans and annotations into it through module-level helpers that do
nothing when no timer is bound, so other callers of ``encode_text`` are unaffected.

The timer is a mutable object held in a ``ContextVar``: ``asyncio.gather`` children and
``run_in_executor_bound`` threads run on a copy of the context, and a mutable recorder (not a
re-set variable) is what makes their writes visible. The kNN page and the unembedded count
travel in one ``_msearch`` (#384), so ``knn_page`` is the wall span of both.

Nothing is observed or emitted until the route calls :meth:`SearchTimer.finish` on success,
so error responses carry no header and leave the histogram alone.
"""

from __future__ import annotations

import contextlib
import time
from contextvars import ContextVar
from typing import TYPE_CHECKING

from prometheus_client import Histogram


if TYPE_CHECKING:
    from collections.abc import Iterator

SEARCH_STAGES: tuple[str, ...] = (
    'bootstrap',
    'encode',
    'source_vector',
    'filter_build',
    'knn_page',
    'hydrate',
)

_SECONDS = Histogram(
    'op_search_stage_seconds',
    'Wall seconds spent in one stage of a curation kNN search request.',
    labelnames=('route', 'stage'),
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 10.0),
)

_clock = time.perf_counter


class SearchTimer:
    """Stage durations (seconds) and annotations of one search request."""

    __slots__ = ('annotations', 'durations', 'route')

    def __init__(self, route: str) -> None:
        self.route = route
        self.durations: dict[str, float] = {}
        self.annotations: dict[str, int] = {}

    def record(self, stage: str, seconds: float) -> None:
        if stage not in SEARCH_STAGES:
            raise ValueError(f'unknown search stage {stage!r}; expected one of {SEARCH_STAGES}')
        self.durations[stage] = self.durations.get(stage, 0.0) + seconds

    def header(self) -> str:
        """``Server-Timing`` value: one ``name;dur=<ms>`` entry per stage, then annotations."""
        parts = [
            f'{s};dur={self.durations[s] * 1000:.2f}' for s in SEARCH_STAGES if s in self.durations
        ]
        parts.extend(f'{k};dur={v}' for k, v in self.annotations.items())
        return ', '.join(parts)

    def finish(self) -> str:
        """Observe every recorded stage in the histogram and return the header value."""
        for stage, seconds in self.durations.items():
            _SECONDS.labels(route=self.route, stage=stage).observe(seconds)
        return self.header()


_current: ContextVar[SearchTimer | None] = ContextVar('op_search_timer', default=None)


@contextlib.contextmanager
def bind_search_timer(route: str) -> Iterator[SearchTimer]:
    """Bind a fresh timer for the duration of one request."""
    timer = SearchTimer(route)
    token = _current.set(timer)
    try:
        yield timer
    finally:
        _current.reset(token)


@contextlib.contextmanager
def search_span(stage: str) -> Iterator[None]:
    """Record the block's wall time under ``stage``; a no-op with no timer bound."""
    timer = _current.get()
    if timer is None:
        yield
        return
    if stage not in SEARCH_STAGES:
        raise ValueError(f'unknown search stage {stage!r}; expected one of {SEARCH_STAGES}')
    start = _clock()
    try:
        yield
    finally:
        timer.record(stage, _clock() - start)


def search_annotate(key: str, value: int) -> None:
    """Attach an integer fact (``encode_cache_hit``, ``os_took_ms``); a no-op with no timer."""
    timer = _current.get()
    if timer is not None:
        timer.annotations[key] = value
