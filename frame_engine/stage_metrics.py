"""The frame engine's Prometheus metrics, served on ``OP_FRAME_ENGINE_METRICS_PORT``.

Per-stage wall timing of one request: two counters, both labelled ``stage``,
``op_frame_engine_stage_seconds_total`` (summed host wall seconds) and
``op_frame_engine_stage_observations_total`` (images that ran the stage). Seconds over
observations is the mean per image. Stage seconds are sums over concurrent requests, so they are
compared across runs and never added up to wall time. The stage names are the ones the Triton
prep model published, so the baseline tables stay comparable:

``queue``        wait for a worker thread (gRPC receipt to the first stage)
``budget``       wait for device memory (``prep_budget``); only photos that route to the GPU
``decode``       header routing, nvJPEG decode, EXIF orientation
``letterbox``    detector letterbox canvas
``detect``       detector call, box decoding
``embed_frame``  whole-frame square for the encoder
``embed_crops``  one square per detection (or stored box) for the encoder
``embed_pe``     the encoder calls over those squares
``crop_copy``    device-to-host copy of each detection's pixels
``crop_jpeg``    JPEG encode, summed over the image's crops (host threads, concurrent)
``blur``         Laplacian sums of the frame and every detection

CUDA launches are asynchronous, so by default a kernel stage ends when its launches are queued and
its GPU time lands in the next stage that waits (usually ``detect`` or ``embed_pe``). With
``OP_FRAME_ENGINE_STAGE_SYNC`` set the clock waits for the stream at the end of those stages so
the time is attributed where the work ran; that removes overlap inside one image, so use it for
attribution runs, not for throughput numbers.

Besides the stages: requests in flight, the shared-memory slab (bytes in use, peak, waits), budget
waits, region registrations, failed model calls by gRPC code, overloads and image outcomes.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from prometheus_client import CollectorRegistry, Counter, Gauge


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


STAGES = (
    'queue',
    'budget',
    'decode',
    'letterbox',
    'detect',
    'embed_frame',
    'embed_crops',
    'embed_pe',
    'crop_copy',
    'crop_jpeg',
    'blur',
)
ASYNC_STAGES = frozenset({'letterbox', 'embed_frame', 'embed_crops', 'blur'})
SECONDS_NAME = 'op_frame_engine_stage_seconds'
OBSERVATIONS_NAME = 'op_frame_engine_stage_observations'


class StageClock:
    """Seconds spent per stage by one request."""

    __slots__ = ('_sync', 'seconds')

    def __init__(self, sync: Callable[[], None] | None = None) -> None:
        self._sync = sync
        self.seconds: dict[str, float] = {}

    def add(self, stage: str, seconds: float) -> None:
        if stage not in STAGES:
            raise ValueError(f'unknown frame engine stage {stage!r}')
        self.seconds[stage] = self.seconds.get(stage, 0.0) + seconds

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            if self._sync is not None and name in ASYNC_STAGES:
                self._sync()
            self.add(name, time.perf_counter() - start)


class EngineMetrics:
    """Every series of one engine process, in its own registry (served by ``main``)."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.stage_seconds = Counter(
            SECONDS_NAME, 'Frame engine stage wall seconds.', ['stage'], registry=r
        )
        self.stage_observations = Counter(
            OBSERVATIONS_NAME, 'Images that ran a frame engine stage.', ['stage'], registry=r
        )
        self.in_flight = Gauge(
            'op_frame_engine_in_flight', 'Requests admitted and not yet answered.', registry=r
        )
        self.slab_in_use = Gauge(
            'op_frame_engine_slab_in_use_bytes', 'Bytes of the region handed out.', registry=r
        )
        self.slab_peak = Gauge(
            'op_frame_engine_slab_peak_bytes', 'Most bytes of the region handed out.', registry=r
        )
        self.slab_waits = Gauge(
            'op_frame_engine_slab_waits', 'Requests that waited for room in the region.', registry=r
        )
        self.budget_waits = Counter(
            'op_frame_engine_budget_waits',
            'Requests that waited for device memory, by outcome (admitted or timeout).',
            ['outcome'],
            registry=r,
        )
        self.registrations = Counter(
            'op_frame_engine_region_registrations',
            'Shared-memory region registrations with the model server, by cause.',
            ['cause'],
            registry=r,
        )
        self.triton_errors = Counter(
            'op_frame_engine_triton_call_errors',
            'Failed model-server calls, by model and gRPC status.',
            ['model', 'code'],
            registry=r,
        )
        self.outcomes = Counter(
            'op_frame_engine_images',
            'Images answered, by status (ok or the reason) and decode route.',
            ['status', 'decode_route'],
            registry=r,
        )
        self.overloads = Counter(
            'op_frame_engine_overloads',
            'Requests answered with the overload status, by cause.',
            ['cause'],
            registry=r,
        )

    def publish(self, clock: StageClock) -> None:
        for stage, secs in clock.seconds.items():
            self.stage_seconds.labels(stage).inc(secs)
            self.stage_observations.labels(stage).inc()

    def watch_slab(self, slab: Any) -> None:
        """Read the slab's counters at scrape time."""
        self.slab_in_use.set_function(lambda: float(slab.in_use))
        self.slab_peak.set_function(lambda: float(slab.peak))
        self.slab_waits.set_function(lambda: float(slab.waits))
