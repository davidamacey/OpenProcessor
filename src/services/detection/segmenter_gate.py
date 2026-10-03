"""One gate deciding whether to SPEND a segmenter call: shared by the crop
region stage and the full-image open-vocabulary pass.

Three tiers, in order, each able to end the decision:

1. Registry rules (free): the target is enabled, and when it names parent
   classes the subject (a crop's own class/proposal names, or the names of
   every item on an image) holds one of them. The predicate is the one the
   region stage already uses (:func:`~src.services.curation.region_scope.in_parent_classes`).
2. An optional vision-model yes/no ("is a <prompt> visible?"), only when the
   caller supplies one. A "no" skips; an ERROR or an unparseable reply is not
   a "no": the call runs (logged by the caller).
3. A learned hit rate per target: after ``miss_threshold`` misses in the last
   ``window`` runs the target is only SAMPLED at ``sample_floor``, so it can
   recover; default off.

Invariants: the gate only decides whether to spend a call. It never edits a
label, a box or a lock; infrastructure failure is never a skip (the caller's
segmenter outage handling is separate); every decision names its tier and
reason so skips can be counted and shown.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from src.core.logging import get_logger
from src.services.curation.region_scope import in_parent_classes


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from src.services.detection.open_vocab_set import HitRateGate

logger = get_logger(__name__)

GateReason = Literal['disabled', 'no_parent_class', 'vlm_no', 'hit_rate']


@dataclass(frozen=True)
class GateDecision:
    run: bool
    tier: int | None = None
    reason: GateReason | None = None
    #: A tier-3 recovery sample: it runs although the target is on a miss streak.
    sampled: bool = False

    @property
    def label(self) -> str:
        """Counter key for a skip: ``tier<N>_<reason>``."""
        return f'tier{self.tier}_{self.reason}'


RUN = GateDecision(run=True)


@dataclass(frozen=True)
class GateSubject:
    """What the gate is deciding for. ``names`` are the class / proposal names
    of the subject: one crop's, or every item on the image."""

    names: tuple[str | None, ...] = ()


@dataclass
class HitRateTracker:
    """Rolling hit/miss windows per target key (``window`` most recent runs).
    Persisted by the caller (:meth:`dump` / :meth:`load`) so a restart does not
    forget what a target has been doing."""

    windows: dict[str, deque[bool]] = field(default_factory=dict)

    def record(self, key: str, *, hit: bool, window: int) -> None:
        recent = self.windows.setdefault(key, deque(maxlen=window))
        if recent.maxlen != window:
            recent = self.windows[key] = deque(recent, maxlen=window)
        recent.append(hit)

    def misses(self, key: str) -> int:
        return sum(1 for hit in self.windows.get(key, ()) if not hit)

    def dump(self) -> dict[str, list[bool]]:
        return {k: list(v) for k, v in self.windows.items()}

    @classmethod
    def load(cls, raw: dict[str, list[bool]] | None) -> HitRateTracker:
        return cls({k: deque(v) for k, v in (raw or {}).items()})


def registry_decision(
    *, enabled: bool, parent_classes: Iterable[str], subject: GateSubject
) -> GateDecision | None:
    """Tier 1. ``None`` = no objection."""
    if not enabled:
        return GateDecision(run=False, tier=1, reason='disabled')
    parents = tuple(parent_classes)
    if parents and not any(
        in_parent_classes(parents, class_name=name, proposal_name=None) for name in subject.names
    ):
        return GateDecision(run=False, tier=1, reason='no_parent_class')
    return None


def hit_rate_decision(
    tracker: HitRateTracker, key: str, cfg: HitRateGate, rand: Callable[[], float]
) -> GateDecision | None:
    """Tier 3. ``None`` = no objection (gate off, or not on a miss streak)."""
    if not cfg.enabled or tracker.misses(key) < cfg.miss_threshold:
        return None
    if rand() < cfg.sample_floor:
        return GateDecision(run=True, sampled=True)
    return GateDecision(run=False, tier=3, reason='hit_rate')


async def decide(
    *,
    enabled: bool,
    parent_classes: Iterable[str],
    subject: GateSubject,
    vlm_visible: Callable[[], Awaitable[bool | None]] | None = None,
    hit_rate: tuple[HitRateTracker, str, HitRateGate] | None = None,
    rand: Callable[[], float] = random.random,
) -> GateDecision:
    """The three tiers in order; the first that objects ends it. ``vlm_visible``
    answers True/False, or ``None`` when it could not (then the call runs)."""
    objection = registry_decision(enabled=enabled, parent_classes=parent_classes, subject=subject)
    if objection is not None:
        return objection
    if vlm_visible is not None:
        try:
            visible = await vlm_visible()
        except Exception as exc:
            # An error is not a "no": the call runs.
            logger.warning('segmenter_gate_vlm_failed', error=str(exc))
            visible = None
        if visible is False:
            return GateDecision(run=False, tier=2, reason='vlm_no')
    if hit_rate is not None:
        tracker, key, cfg = hit_rate
        objection = hit_rate_decision(tracker, key, cfg, rand)
        if objection is not None:
            return objection
    return RUN


__all__ = [
    'RUN',
    'GateDecision',
    'GateReason',
    'GateSubject',
    'HitRateTracker',
    'decide',
    'hit_rate_decision',
    'registry_decision',
]
