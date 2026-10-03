"""Per-stage wall-time and byte metrics for the ingest hot path.

One shared timer for every stage boundary so a baseline run can attribute
time without ad-hoc ``perf_counter`` pairs scattered through the code. The
timer only reads a clock twice and updates two prometheus children; it never
changes what the wrapped block does and never swallows its exceptions.

``host_to_device`` is deliberately not a stage here: the copy happens inside
Triton, so it is derived from Triton's own ``compute_input`` counters by the
baseline harness (``scripts/bench/run_baseline.py``).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prometheus_client import Counter, Histogram


if TYPE_CHECKING:
    from types import TracebackType


STAGES: tuple[str, ...] = (
    'decode',
    'crop',
    'jpeg_encode',
    'resize',
    'embed',
    'opensearch_write',
)

_SECONDS = Histogram(
    'op_pipeline_stage_seconds',
    'Wall seconds spent in one ingest pipeline stage call.',
    labelnames=('stage',),
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 10.0),
)
_BYTES = Counter(
    'op_pipeline_stage_bytes_total',
    'Bytes handled by an ingest pipeline stage (input or output, see the stage).',
    labelnames=('stage',),
)

_SECONDS_BY_STAGE = {s: _SECONDS.labels(stage=s) for s in STAGES}
_BYTES_BY_STAGE = {s: _BYTES.labels(stage=s) for s in STAGES}

_clock = time.perf_counter


class _StageTimer:
    __slots__ = ('_nbytes', '_stage', '_start')

    def __init__(self, stage: str, nbytes: int) -> None:
        self._stage = stage
        self._nbytes = nbytes
        self._start = 0.0

    def __enter__(self) -> _StageTimer:
        self._start = _clock()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        _SECONDS_BY_STAGE[self._stage].observe(_clock() - self._start)
        if self._nbytes:
            _BYTES_BY_STAGE[self._stage].inc(self._nbytes)


def add_stage_bytes(stage: str, nbytes: int) -> None:
    """Count ``nbytes`` for ``stage`` when the size is only known after the block."""
    if stage not in _BYTES_BY_STAGE:
        raise ValueError(f'unknown stage {stage!r}; expected one of {STAGES}')
    if nbytes > 0:
        _BYTES_BY_STAGE[stage].inc(nbytes)


def stage_timer(stage: str, *, nbytes: int = 0) -> _StageTimer:
    """Context manager that records the block's wall time (and ``nbytes``) for ``stage``."""
    if stage not in _SECONDS_BY_STAGE:
        raise ValueError(f'unknown stage {stage!r}; expected one of {STAGES}')
    return _StageTimer(stage, nbytes)
