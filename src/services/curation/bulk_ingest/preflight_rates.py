"""The measured figures a bulk-ingest preview estimates from, and the estimate itself.

Nothing here is a guess: every number is copied from ``docs/PERFORMANCE.md`` (the
section is named beside it, and a test checks the figure and the section are still in
that file). They are served to clients by ``GET /ingest/bulk/config`` so a screen shows
the basis of an estimate instead of an unexplained number. They describe the measured
hardware (one RTX A6000, policy ``all`` on public COCO and Commons photos); another
host or detector profile will differ, which is why the answer is labelled an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DECODE_SECTION = 'Decode once and the GPU decode (#217)'
STORAGE_SECTION = 'Storage per image and per vector'
BASELINE_SECTION = 'v0.5.0 baseline (set A, public COCO, 2,000 images)'


@dataclass(frozen=True)
class Rate:
    route: Literal['cpu_small', 'cpu_large', 'gpu_large']
    images_per_s: float
    source_section: str
    measured_on: str


@dataclass(frozen=True)
class Measured:
    value: float
    unit: str
    source_section: str
    measured_on: str


RATES: tuple[Rate, ...] = (
    Rate(
        'cpu_small',
        29.0,
        DECODE_SECTION,
        'COCO frames (0.3 MP), CPU decode, detect + embed, /ingest/batch, 1 x A6000',
    ),
    Rate(
        'cpu_large',
        27.0,
        DECODE_SECTION,
        'Commons photos (12 to 24 MP), CPU decode, detect + embed, /ingest/batch, 1 x A6000',
    ),
    Rate(
        'gpu_large',
        35.0,
        DECODE_SECTION,
        'Commons photos (12 to 24 MP), GPU decode, detect + embed, /ingest/batch, 1 x A6000',
    ),
)
EMBEDDINGS_PER_IMAGE = Measured(
    7.4, 'embeddings per image', BASELINE_SECTION, 'COCO, policy all (detector + PE encoder)'
)
OPENSEARCH_BYTES_PER_IMAGE = Measured(
    65298,
    'bytes per image',
    STORAGE_SECTION,
    'COCO, policy all, /ingest/batch, merged until the size repeats (images + items indexes)',
)


@dataclass(frozen=True)
class Estimate:
    # ``megapixel_sample``: one rate per image from the sampled sizes. ``rate_range``:
    # sizes unknown, so the span from the slowest to the fastest measured route.
    basis: Literal['megapixel_sample', 'rate_range']
    duration_s_min: float
    duration_s_max: float
    vectors: float
    opensearch_bytes: int


def estimate(*, files: int, share_large: float | None, gpu_decode: bool) -> Estimate:
    """Duration and footprint for ``files`` images. ``share_large`` is the sampled share
    of images at or above the GPU-decode gate, or ``None`` when no sizes are known."""
    rates = {r.route: r.images_per_s for r in RATES}
    if share_large is None:
        fastest, slowest = max(rates.values()), min(rates.values())
        basis: Literal['megapixel_sample', 'rate_range'] = 'rate_range'
        low, high = files / fastest, files / slowest
    else:
        large_rate = rates['gpu_large'] if gpu_decode else rates['cpu_large']
        large = files * share_large
        low = high = (files - large) / rates['cpu_small'] + large / large_rate
        basis = 'megapixel_sample'
    return Estimate(
        basis=basis,
        duration_s_min=low,
        duration_s_max=high,
        vectors=files * EMBEDDINGS_PER_IMAGE.value,
        opensearch_bytes=round(files * OPENSEARCH_BYTES_PER_IMAGE.value),
    )
