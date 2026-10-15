"""Preflight for a bulk ingest: what a folder, manifest or S3 prefix holds and roughly
what ingesting it would cost, with no ingestion and no write of any kind.

The walk is bounded (a file cap and a time cap, both served by ``GET /ingest/bulk/config``)
and stoppable: a source too large to finish inside the caps answers with what was seen
and ``partial: true`` plus the reason, never a silent truncation. An unreadable root
refuses. Image sizes come from a bounded sample of file headers (no pixel is decoded);
an S3 prefix is only listed, so its estimate is a range across the measured routes.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PIL import Image

from src.services.curation.bulk_ingest import preflight_rates
from src.services.curation.bulk_ingest.s3_listing import wanted_extension
from src.services.curation.bulk_ingest.source import (
    BulkSource,
    InvalidSourceError,
    iter_manifest_paths,
    validate_source,
)
from src.services.curation.bulk_ingest.walk import WalkStats, iter_image_paths
from src.services.curation.image_serving import is_servable_image_path


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from src.services.curation.bulk_ingest.object_store import ObjectStore, S3Config

StopReason = Literal['file_cap', 'time_cap', 'cancelled']
# Upper edges (megapixels) of the histogram buckets; the last bucket is open.
HISTOGRAM_EDGES_MP: tuple[float, ...] = (1.0, 3.0, 8.0, 16.0, 24.0)
HEADER_SAMPLE_SEED = 20261010


@dataclass(frozen=True)
class PreviewLimits:
    """Bounds on one preview, served to clients as part of the bulk-ingest config."""

    max_files: int = 200_000
    max_seconds: float = 60.0
    header_sample: int = 300


@dataclass
class ExtensionTotals:
    files: int = 0
    bytes: int = 0


@dataclass(frozen=True)
class MegapixelSample:
    sampled: int = 0
    unreadable: int = 0
    histogram: dict[str, int] = field(default_factory=dict)
    gate_megapixels: float = 0.0
    # Share of the sampled images at or above the GPU-decode gate; None without a sample.
    share_at_or_above_gate: float | None = None


@dataclass
class PreviewResult:
    files: int = 0
    total_bytes: int = 0
    by_extension: dict[str, ExtensionTotals] = field(default_factory=dict)
    megapixels: MegapixelSample = field(default_factory=MegapixelSample)
    estimate: preflight_rates.Estimate = field(
        default_factory=lambda: preflight_rates.estimate(
            files=0, share_large=None, gpu_decode=False
        )
    )
    partial: bool = False
    stop_reason: StopReason | None = None
    unreadable_dirs: int = 0
    unreadable_samples: list[str] = field(default_factory=list)
    symlinks_skipped: int = 0
    # Manifest entries only.
    missing: int = 0
    outside_roots: int = 0
    # Files that could not be statted.
    unreadable_files: int = 0
    gpu_decode: bool = False


class _Budget:
    """The caps and the cancel flag of one preview; remembers which one tripped."""

    def __init__(
        self,
        limits: PreviewLimits,
        should_stop: Callable[[], bool],
        clock: Callable[[], float],
    ) -> None:
        self.limits = limits
        self._should_stop = should_stop
        self._clock = clock
        self._started = clock()
        self.reason: StopReason | None = None

    def stopped(self) -> bool:
        """Cancel or time cap (also the walk's stop poll)."""
        if self.reason is not None:
            return True
        if self._should_stop():
            self.reason = 'cancelled'
        elif self._clock() - self._started > self.limits.max_seconds:
            self.reason = 'time_cap'
        return self.reason is not None

    def full(self, files: int) -> bool:
        if files >= self.limits.max_files:
            self.reason = 'file_cap'
        return self.reason is not None


class _Reservoir:
    """A fixed-size uniform sample of the paths seen (algorithm R, seeded)."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.items: list[str] = []
        self._seen = 0
        self._rng = random.Random(HEADER_SAMPLE_SEED)

    def add(self, item: str) -> None:
        self._seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
            return
        slot = self._rng.randrange(self._seen)
        if slot < self.size:
            self.items[slot] = item


def _bucket_label(megapixels: float) -> str:
    previous = 0.0
    for edge in HISTOGRAM_EDGES_MP:
        if megapixels < edge:
            return f'<{edge:g}' if previous == 0.0 else f'{previous:g}-{edge:g}'
        previous = edge
    return f'{previous:g}+'


def _sample_headers(
    paths: Iterable[str], gate_megapixels: float, stopped: Callable[[], bool]
) -> MegapixelSample:
    histogram: dict[str, int] = {}
    sampled = unreadable = at_gate = 0
    for path in paths:
        if stopped():
            break
        try:
            with Image.open(path) as image:
                width, height = image.size
        except (OSError, ValueError, Image.DecompressionBombError):
            unreadable += 1
            continue
        megapixels = width * height / 1e6
        label = _bucket_label(megapixels)
        histogram[label] = histogram.get(label, 0) + 1
        sampled += 1
        at_gate += megapixels >= gate_megapixels
    return MegapixelSample(
        sampled,
        unreadable,
        histogram,
        gate_megapixels,
        at_gate / sampled if sampled else None,
    )


def _finish(
    result: PreviewResult,
    budget: _Budget,
    sample: MegapixelSample,
    *,
    gpu_decode: bool,
) -> PreviewResult:
    result.megapixels = sample
    result.gpu_decode = gpu_decode
    result.partial = budget.reason is not None
    result.stop_reason = budget.reason
    result.estimate = preflight_rates.estimate(
        files=result.files, share_large=sample.share_at_or_above_gate, gpu_decode=gpu_decode
    )
    return result


def _count(result: PreviewResult, suffix: str, size: int) -> None:
    result.files += 1
    result.total_bytes += size
    totals = result.by_extension.setdefault(suffix, ExtensionTotals())
    totals.files += 1
    totals.bytes += size


def preview_source(
    source: BulkSource,
    *,
    limits: PreviewLimits = PreviewLimits(),  # noqa: B008 - frozen defaults
    should_stop: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
    gpu_decode: bool = True,
    gpu_min_megapixels: float = 3.0,
    walk_workers: int = 8,
) -> PreviewResult:
    """Preview a folder or manifest. Blocking: run it off the event loop. Raises
    :class:`InvalidSourceError` for a path outside the source roots, a missing or
    unreadable root."""
    if source.kind not in {'folder', 'manifest'}:
        raise InvalidSourceError(f'cannot preview a {source.kind} source')
    validate_source(source)
    budget = _Budget(limits, should_stop, clock)
    result = PreviewResult()
    reservoir = _Reservoir(limits.header_sample)
    if source.kind == 'folder':
        _preview_folder(source, result, budget, reservoir, walk_workers)
    else:
        _preview_manifest(source, result, budget, reservoir)
    sample = _sample_headers(reservoir.items, gpu_min_megapixels, should_stop)
    return _finish(result, budget, sample, gpu_decode=gpu_decode)


def _preview_folder(
    source: BulkSource,
    result: PreviewResult,
    budget: _Budget,
    reservoir: _Reservoir,
    walk_workers: int,
) -> None:
    stats = WalkStats()
    walk = iter_image_paths(
        Path(source.path),
        frozenset(source.extensions),
        walk_workers,
        stats=stats,
        should_stop=budget.stopped,
    )
    try:
        for path in walk:
            if budget.stopped() or budget.full(result.files):
                break
            try:
                size = path.stat().st_size
            except OSError:
                result.unreadable_files += 1
                continue
            _count(result, path.suffix.lower(), size)
            reservoir.add(str(path))
    finally:
        walk.close()
    result.unreadable_dirs = stats.unreadable_dirs
    result.unreadable_samples = list(stats.unreadable_samples)
    result.symlinks_skipped = stats.symlinks


def _preview_manifest(
    source: BulkSource, result: PreviewResult, budget: _Budget, reservoir: _Reservoir
) -> None:
    for entry in iter_manifest_paths(Path(source.path)):
        if budget.stopped() or budget.full(result.files + result.missing + result.outside_roots):
            break
        if not is_servable_image_path(entry):
            result.outside_roots += 1
            continue
        try:
            size = Path(entry).stat().st_size
        except FileNotFoundError:
            result.missing += 1
            continue
        except OSError:
            result.unreadable_files += 1
            continue
        _count(result, Path(entry).suffix.lower(), size)
        reservoir.add(entry)


def preview_s3(
    store: ObjectStore,
    config: S3Config,
    bucket: str,
    prefix: str,
    extensions: Iterable[str],
    *,
    limits: PreviewLimits = PreviewLimits(),  # noqa: B008 - frozen defaults
    should_stop: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
    gpu_decode: bool = True,
) -> PreviewResult:
    """Preview a bucket/prefix by listing it (sizes and keys only, no object is read).
    Raises the allow-list errors of :meth:`S3Config.check` and the store's classified
    listing errors. Blocking."""
    config.check(bucket, prefix)
    wanted = frozenset(e.lower() for e in extensions)
    budget = _Budget(limits, should_stop, clock)
    result = PreviewResult()
    token: str | None = None
    while not budget.stopped():
        page = store.list_page(bucket, prefix, token=token, delimiter=None)
        for listed in page.objects:
            if not wanted_extension(listed.ref.key, wanted):
                continue
            if budget.full(result.files):
                break
            dot = listed.ref.key.rfind('.')
            _count(result, listed.ref.key[dot:].lower(), listed.ref.size or 0)
        token = page.next_token
        if token is None or budget.reason is not None:
            break
    return _finish(result, budget, MegapixelSample(), gpu_decode=gpu_decode)


__all__ = [
    'ExtensionTotals',
    'MegapixelSample',
    'PreviewLimits',
    'PreviewResult',
    'preview_s3',
    'preview_source',
]
