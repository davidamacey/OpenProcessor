"""Per-stage wall timing of one ``gpu_image_prep`` request, exported as Triton custom metrics.

Two counters, both labelled ``stage``: ``op_gpu_prep_stage_seconds`` (summed host wall seconds)
and ``op_gpu_prep_stage_observations`` (images that ran the stage). Seconds over observations is
the mean per image. Stage seconds are sums over concurrent requests, so they are compared across
runs and never added up to wall time.

A request keeps one :class:`StageClock` and publishes it once when it ends: the hot path costs a
few ``perf_counter`` calls and one addition per stage per image (the counters are flushed in
batches, see :class:`StageMetrics`). Stages:

``queue``        wait for a free slot and a worker thread (``execute`` receipt to the first stage)
``decode``       header routing, nvJPEG decode, EXIF orientation
``letterbox``    detector letterbox canvas
``detect``       detector call (BLS), box decoding
``embed_frame``  whole-frame square for PE
``embed_crops``  one square per detection (or stored box) for PE
``embed_pe``     the PE calls (BLS) over those squares
``crop_copy``    device-to-host copy of each detection's pixels
``crop_jpeg``    JPEG encode, summed over the image's crops (host threads, concurrent)
``blur``         Laplacian sums of the frame and every detection

CUDA launches are asynchronous, so by default a kernel stage ends when its launches are queued and
its GPU time lands in the next stage that waits (usually ``detect`` or ``embed_pe``). With
``sync`` set (parameter ``STAGE_SYNC``) the clock waits for the stream at the end of those stages
so the time is attributed where the work ran; that removes overlap inside one image, so use it
for attribution runs, not for throughput numbers.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


STAGES = (
    'queue',
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
SECONDS_NAME = 'op_gpu_prep_stage_seconds'
OBSERVATIONS_NAME = 'op_gpu_prep_stage_observations'


class StageClock:
    """Seconds spent per stage by one request."""

    __slots__ = ('_sync', 'seconds')

    def __init__(self, sync: Callable[[], None] | None = None) -> None:
        self._sync = sync
        self.seconds: dict[str, float] = {}

    def add(self, stage: str, seconds: float) -> None:
        if stage not in STAGES:
            raise ValueError(f'unknown gpu_image_prep stage {stage!r}')
        self.seconds[stage] = self.seconds.get(stage, 0.0) + seconds

    @contextmanager
    def stage(self, name: str) -> Any:
        start = time.perf_counter()
        try:
            yield
        finally:
            if self._sync is not None and name in ASYNC_STAGES:
                self._sync()
            self.add(name, time.perf_counter() - start)


class StageMetrics:
    """The two counter families of one model instance; ``pb_utils`` is the Triton module.

    Triton 2.70's Python backend aborts the server when several threads increment custom metrics
    at once, and deletes a family when its Python object is collected (its series leave /metrics,
    ``increment`` raises); a family that dies before its series crashes it too. So requests only
    add to local sums (``publish``), one flusher thread is the sole caller of ``increment`` (every
    ``FLUSH_SECONDS`` and once at ``close``), and the families live as long as the series.
    """

    FLUSH_SECONDS = 0.5

    def __init__(self, pb_utils: Any) -> None:
        kind = pb_utils.MetricFamily.COUNTER
        self._families = (
            pb_utils.MetricFamily(
                name=SECONDS_NAME, description='gpu_image_prep stage wall seconds', kind=kind
            ),
            pb_utils.MetricFamily(
                name=OBSERVATIONS_NAME,
                description='images that ran a gpu_image_prep stage',
                kind=kind,
            ),
        )
        seconds, seen = self._families
        self._seconds = {s: seconds.Metric(labels={'stage': s}) for s in STAGES}
        self._seen = {s: seen.Metric(labels={'stage': s}) for s in STAGES}
        self._pending_seconds = dict.fromkeys(STAGES, 0.0)
        self._pending_seen = dict.fromkeys(STAGES, 0)
        self._pending_lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._stop = threading.Event()
        self._flusher = threading.Thread(target=self._flush_loop, name='stage-metrics', daemon=True)
        self._flusher.start()

    def publish(self, clock: StageClock) -> None:
        with self._pending_lock:
            for stage, secs in clock.seconds.items():
                self._pending_seconds[stage] += secs
                self._pending_seen[stage] += 1

    def flush(self) -> None:
        with self._flush_lock:
            with self._pending_lock:
                seconds, seen = self._pending_seconds, self._pending_seen
                self._pending_seconds = dict.fromkeys(STAGES, 0.0)
                self._pending_seen = dict.fromkeys(STAGES, 0)
            for stage in STAGES:
                if seen[stage]:
                    self._seconds[stage].increment(seconds[stage])
                    self._seen[stage].increment(seen[stage])

    def _flush_loop(self) -> None:
        while not self._stop.wait(self.FLUSH_SECONDS):
            self.flush()

    def close(self) -> None:
        """Flush, then delete the series before their families (the reverse order crashes)."""
        self._stop.set()
        self._flusher.join()
        self.flush()
        self._seconds = {}
        self._seen = {}
        self._families = ()
