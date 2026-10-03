"""The crop region stage's use of the shared segmenter gate.

Tier 1 (the profile's ``parent_classes``) is applied when items are fetched and
seeded, and tier 2 (a vision-model "is a region visible?") is the batched
visibility stage ahead of the segmenter, so what is decided here, per item just
before its segmenter call, is tier 3: a learned hit rate per item class
(:func:`src.services.detection.segmenter_gate.decide`, the one function the
full-image pass also calls). Off unless the region profile sets
``gate_hit_rate``.

A skipped item is stamped, never silently dropped (the caller writes
``region_gate_skip``). An item whose class a human owns or validated is never
skipped (the lock rule). Windows live in this worker's memory, one tracker per
project: a restart re-learns them from the next calls.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING

from src.services.curation.class_write_guard import class_write_locked
from src.services.curation.metrics import (
    OP_REGION_SEGMENTER_CALLS_TOTAL,
    OP_REGION_SEGMENTER_SECONDS_TOTAL,
    OP_SEGMENTER_GATE_DECISIONS_TOTAL,
)
from src.services.detection.open_vocab_set import HitRateGate
from src.services.detection.segmenter_gate import (
    RUN,
    GateDecision,
    GateSubject,
    HitRateTracker,
    decide,
)
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Callable

    from scripts.curation.worker.state import _ItemTask
    from src.config import DetectionProfile


def class_label(t: _ItemTask) -> str:
    """The item's class for the gate and the metrics: its class name, or the
    ingest detector's own label while it has none."""
    return normalize_class_name(t.class_name or t.proposal_name)


def _locked(t: _ItemTask) -> bool:
    return class_write_locked(
        {
            'class_source': t.class_source,
            'class_validated': t.class_validated,
            'test_holdout': t.test_holdout,
        }
    )


class CropGate:
    def __init__(self, rand: Callable[[], float] = random.random) -> None:
        self._rand = rand
        self._trackers: dict[str, HitRateTracker] = {}
        #: ``{class label: {'hit'|'miss'|'skipped'|'sampled': n}}``, for worker stats.
        self.stats: dict[str, Counter[str]] = {}

    def _tracker(self, t: _ItemTask) -> HitRateTracker:
        slug = t.project.slug if t.project is not None else ''
        return self._trackers.setdefault(slug, HitRateTracker())

    def _count(self, label: str, what: str) -> None:
        self.stats.setdefault(label, Counter())[what] += 1

    @property
    def skipped_total(self) -> int:
        return sum(c['skipped'] for c in self.stats.values())

    async def decide(self, t: _ItemTask, profile: DetectionProfile) -> GateDecision:
        label = class_label(t)
        if not profile.gate_hit_rate or not label or _locked(t):
            return RUN
        cfg = HitRateGate(
            enabled=True,
            window=profile.gate_hit_window,
            miss_threshold=profile.gate_hit_miss_threshold,
            sample_floor=profile.gate_hit_sample_floor,
        )
        decision = await decide(
            enabled=True,
            parent_classes=(),
            subject=GateSubject(),
            hit_rate=(self._tracker(t), f'{profile.name}:{label}', cfg),
            rand=self._rand,
        )
        kind = 'sample' if decision.sampled else 'run' if decision.run else 'skip'
        OP_SEGMENTER_GATE_DECISIONS_TOTAL.labels(
            scope='crop', decision=kind, tier=str(decision.tier or ''), reason=decision.reason or ''
        ).inc()
        if not decision.run:
            self._count(label, 'skipped')
        elif decision.sampled:
            self._count(label, 'sampled')
        return decision

    def observe(
        self, t: _ItemTask, profile: DetectionProfile, *, hit: bool, seconds: float
    ) -> None:
        """Record one real segmenter call: feeds the window and the metrics."""
        label = class_label(t) or 'unlabeled'
        outcome = 'hit' if hit else 'miss'
        OP_REGION_SEGMENTER_CALLS_TOTAL.labels(profile.name, label, outcome).inc()
        OP_REGION_SEGMENTER_SECONDS_TOTAL.labels(profile.name, label, outcome).inc(seconds)
        self._count(label, outcome)
        if profile.gate_hit_rate and class_label(t):
            self._tracker(t).record(
                f'{profile.name}:{class_label(t)}', hit=hit, window=profile.gate_hit_window
            )


__all__ = ['CropGate', 'class_label']
