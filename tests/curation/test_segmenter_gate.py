"""Wave 6: the one segmenter gate (tiers 1-3), pure parts and ordering."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.detection.open_vocab_set import HitRateGate
from src.services.detection.segmenter_gate import (
    GateDecision,
    GateSubject,
    HitRateTracker,
    decide,
    hit_rate_decision,
    registry_decision,
)


CFG = HitRateGate(enabled=True, window=10, miss_threshold=4, sample_floor=0.25)


def _subject(*names: str | None) -> GateSubject:
    return GateSubject(names=tuple(names))


async def _decide(**kw: Any) -> GateDecision:
    base: dict[str, Any] = {'enabled': True, 'parent_classes': (), 'subject': _subject()}
    base.update(kw)
    return await decide(**base)


def test_tier1_disabled_target_is_skipped() -> None:
    d = registry_decision(enabled=False, parent_classes=(), subject=_subject())
    assert d == GateDecision(run=False, tier=1, reason='disabled')
    assert d.label == 'tier1_disabled'


@pytest.mark.parametrize(
    ('parents', 'names', 'runs'),
    [
        ((), ('anything',), True),
        ((), (), True),
        (('car',), ('Car', 'tree'), True),
        (('car',), ('tree',), False),
        (('car',), (None, ''), False),
        (('car',), (), False),
        ((' CAR ',), ('car',), True),
    ],
)
def test_tier1_parent_classes_match_like_the_region_stage(
    parents: tuple[str, ...], names: tuple[str | None, ...], runs: bool
) -> None:
    d = registry_decision(enabled=True, parent_classes=parents, subject=_subject(*names))
    assert (d is None) is runs
    if d is not None:
        assert (d.tier, d.reason) == (1, 'no_parent_class')


def test_tier3_off_or_below_threshold_never_objects() -> None:
    tracker = HitRateTracker()
    for _ in range(3):
        tracker.record('k', hit=False, window=10)
    assert hit_rate_decision(tracker, 'k', CFG, lambda: 0.99) is None
    off = HitRateGate(enabled=False)
    for _ in range(20):
        tracker.record('k', hit=False, window=10)
    assert hit_rate_decision(tracker, 'k', off, lambda: 0.99) is None


def test_tier3_on_a_miss_streak_samples_at_the_floor() -> None:
    tracker = HitRateTracker()
    for _ in range(4):
        tracker.record('k', hit=False, window=10)
    skip = hit_rate_decision(tracker, 'k', CFG, lambda: 0.25)
    assert skip == GateDecision(run=False, tier=3, reason='hit_rate')
    sample = hit_rate_decision(tracker, 'k', CFG, lambda: 0.2499)
    assert sample == GateDecision(run=True, sampled=True)


def test_tier3_recovers_when_hits_push_the_misses_out_of_the_window() -> None:
    tracker = HitRateTracker()
    for _ in range(10):
        tracker.record('k', hit=False, window=10)
    assert hit_rate_decision(tracker, 'k', CFG, lambda: 1.0) is not None
    for _ in range(7):
        tracker.record('k', hit=True, window=10)
    assert tracker.misses('k') == 3
    assert hit_rate_decision(tracker, 'k', CFG, lambda: 1.0) is None


def test_tracker_window_is_bounded_and_round_trips() -> None:
    tracker = HitRateTracker()
    for i in range(25):
        tracker.record('k', hit=i % 2 == 0, window=10)
    assert len(tracker.windows['k']) == 10
    again = HitRateTracker.load(tracker.dump())
    assert again.dump() == tracker.dump()
    tracker.record('k', hit=True, window=5)  # a smaller window shrinks it
    assert len(tracker.windows['k']) == 5
    assert HitRateTracker.load(None).dump() == {}


@pytest.mark.asyncio
async def test_tier1_ends_the_decision_before_the_vision_model_is_asked() -> None:
    asked = []

    async def vlm() -> bool:
        asked.append(1)
        return True

    d = await _decide(enabled=False, vlm_visible=vlm)
    assert (d.tier, asked) == (1, [])


@pytest.mark.asyncio
@pytest.mark.parametrize(('answer', 'runs'), [(True, True), (False, False), (None, True)])
async def test_tier2_only_a_no_skips(answer: bool | None, runs: bool) -> None:
    async def vlm() -> bool | None:
        return answer

    d = await _decide(vlm_visible=vlm)
    assert d.run is runs
    if not runs:
        assert (d.tier, d.reason) == (2, 'vlm_no')


@pytest.mark.asyncio
async def test_a_vision_model_error_is_not_a_no() -> None:
    async def vlm() -> bool:
        raise RuntimeError('endpoint down')

    assert (await _decide(vlm_visible=vlm)).run is True


@pytest.mark.asyncio
async def test_tier2_no_ends_the_decision_before_tier3_counts_anything() -> None:
    tracker = HitRateTracker()

    async def vlm() -> bool:
        return False

    d = await _decide(vlm_visible=vlm, hit_rate=(tracker, 'k', CFG))
    assert d.tier == 2


@pytest.mark.asyncio
async def test_tier3_runs_last_and_samples_through_decide() -> None:
    tracker = HitRateTracker()
    for _ in range(10):
        tracker.record('k', hit=False, window=10)
    skipped = await _decide(hit_rate=(tracker, 'k', CFG), rand=lambda: 0.9)
    assert (skipped.tier, skipped.reason) == (3, 'hit_rate')
    sampled = await _decide(hit_rate=(tracker, 'k', CFG), rand=lambda: 0.0)
    assert sampled.run
    assert sampled.sampled


@pytest.mark.asyncio
async def test_default_is_to_run() -> None:
    assert await _decide() == GateDecision(run=True)
