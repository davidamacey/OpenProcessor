"""Per-stage wall timing of one ``gpu_image_prep`` request, exported as Triton custom metrics.

Two counters, both labelled ``stage``: ``op_gpu_prep_stage_seconds`` (summed host wall seconds)
and ``op_gpu_prep_stage_observations`` (images that ran the stage). Seconds over observations is
the mean per image. Stage seconds are sums over concurrent requests, so they are compared across
runs and never added up to wall time.

A request keeps one :class:`StageClock` and publishes it once when it ends: the hot path costs a
few ``perf_counter`` calls and one counter increment per stage per image. Stages:

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
    """The two counter families of one model instance; ``pb_utils`` is the Triton module."""

    def __init__(self, pb_utils: Any) -> None:
        kind = pb_utils.MetricFamily.COUNTER
        seconds = pb_utils.MetricFamily(
            name=SECONDS_NAME, description='gpu_image_prep stage wall seconds', kind=kind
        )
        seen = pb_utils.MetricFamily(
            name=OBSERVATIONS_NAME, description='images that ran a gpu_image_prep stage', kind=kind
        )
        self._seconds = {s: seconds.Metric(labels={'stage': s}) for s in STAGES}
        self._seen = {s: seen.Metric(labels={'stage': s}) for s in STAGES}

    def publish(self, clock: StageClock) -> None:
        for stage, secs in clock.seconds.items():
            self._seconds[stage].increment(secs)
            self._seen[stage].increment(1)
