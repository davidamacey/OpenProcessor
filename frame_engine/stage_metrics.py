"""The frame engine's Prometheus metrics, served on ``OP_FRAME_ENGINE_METRICS_PORT``.

Per-stage timing of one request: two counters, both labelled ``stage``,
``op_frame_engine_stage_seconds_total`` and ``op_frame_engine_stage_observations_total`` (images
that ran the stage). Seconds over observations is the mean per image. A stage's seconds are the
wall time the request's worker thread spent in it. A photo's model calls are in flight together
(two rounds, ``engine.py``), so a call stage (``detect``, ``embed_pe``) counts the commit, the send
and the time the worker WAITED for the answer, not the call's latency: a call answered while the
worker was in another stage costs its stage almost nothing. The stages of one request do not
overlap (``crop_jpeg`` excepted), so per image they add up to about the worker's busy time; over
concurrent requests they are sums, compared across runs, never added up to wall time. The stage
names are the ones the Triton prep model published, so the baseline tables stay comparable:

``queue``        wait for a worker thread (gRPC receipt to the first stage)
``budget``       wait for device memory (``prep_budget``); only photos that route to the GPU
``decode``       header routing, nvJPEG decode, EXIF orientation
``letterbox``    detector letterbox canvas (and the wait for room in the region)
``detect``       round 1's one stream wait and sends, the wait for the detector's answer, boxes
``embed_frame``  whole-frame square for the encoder (and the wait for room)
``embed_crops``  one square per detection (or stored box) for the encoder (and the wait for room,
                 which may collect the photo's oldest call: the window of the region)
``embed_pe``     encoder calls: commit and send of each crop call, the waits for the answers
                 (round 1's wait and sends when the plan has no detector)
``crop_copy``    device-to-host copy of each detection's pixels
``crop_jpeg``    JPEG encode, summed over the image's crops (host threads, concurrent)
``blur``         Laplacian sums of the frame and every detection

CUDA launches are asynchronous, so by default a kernel stage ends when its launches are queued and
its GPU time lands in the next stage that waits (usually ``detect`` or ``embed_pe``). With
``OP_FRAME_ENGINE_STAGE_SYNC`` set the clock waits for the stream at the end of those stages so
the time is attributed where the work ran; that removes overlap inside one image, so use it for
attribution runs, not for throughput numbers.

Besides the stages: requests in flight, the shared-memory slab (bytes in use, peak, waits), the
frame budget (bytes in use, waits), region registrations, failed model calls by gRPC code and
reason, overloads, requests answered at their deadline (by stage), image outcomes, lost CUDA
contexts (by cudaError code) and the process series (``process_start_time_seconds``).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from prometheus_client import CollectorRegistry, Counter, Gauge, ProcessCollector


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

    __slots__ = ('_on_enter', '_sync', 'seconds')

    def __init__(
        self,
        sync: Callable[[], None] | None = None,
        on_enter: Callable[[str], None] | None = None,
    ) -> None:
        self._sync = sync
        self._on_enter = on_enter  # told the name of every stage as it starts
        self.seconds: dict[str, float] = {}

    def add(self, stage: str, seconds: float) -> None:
        if stage not in STAGES:
            raise ValueError(f'unknown frame engine stage {stage!r}')
        self.seconds[stage] = self.seconds.get(stage, 0.0) + seconds

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self._on_enter is not None:
            self._on_enter(name)
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
            SECONDS_NAME,
            'Wall seconds the worker thread spent in a frame engine stage; a model call in flight '
            'counts only while the worker waits for its answer (crop_jpeg: host-thread encode '
            'seconds summed over the crops).',
            ['stage'],
            registry=r,
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
        self.budget_in_use = Gauge(
            'op_frame_engine_budget_in_use_bytes',
            'Bytes of the decoded-frame budget held by requests.',
            registry=r,
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
            'Failed model-server calls, by model, gRPC status and reason (deadline: no answer in '
            'time; unavailable: the channel or server went away; client: the client library '
            'failed the call; rejected: the server answered an error).',
            ['model', 'code', 'reason'],
            registry=r,
        )
        self.deadlines = Counter(
            'op_frame_engine_request_deadlines',
            'Requests answered DEADLINE_EXCEEDED at the request deadline, by the stage they were '
            'last in.',
            ['stage'],
            registry=r,
        )
        self.outcomes = Counter(
            'op_frame_engine_images',
            'Images answered, by status (ok or the reason) and decode route.',
            ['status', 'decode_route'],
            registry=r,
        )
        self.context_lost = Counter(
            'op_frame_engine_context_lost',
            'CUDA contexts lost to a sticky CUDA error, by cudaError code; the process then exits '
            'for a restart (a dying process may not be scraped: changes() over '
            'process_start_time_seconds counts the restarts).',
            ['code'],
            registry=r,
        )
        # process_start_time_seconds and the other process series (prometheus_client)
        ProcessCollector(registry=r)
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

    def watch_budget(self, budget: Any) -> None:
        """Read the budget's bytes in use at scrape time."""
        self.budget_in_use.set_function(lambda: float(budget.in_use))

    def watch_slab(self, slab: Any) -> None:
        """Read the slab's counters at scrape time."""
        self.slab_in_use.set_function(lambda: float(slab.in_use))
        self.slab_peak.set_function(lambda: float(slab.peak))
        self.slab_waits.set_function(lambda: float(slab.waits))
