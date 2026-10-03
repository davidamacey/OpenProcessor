"""W8.4: select up to N region candidates per item.

One function, used by every candidate-producing leg (detector, segmenter,
text-hint re-pass) and by ``POST /region_profiles/test`` (the ``selected``
/ ``drop_reason`` flags): :func:`select_region_candidates`. Regions are
always a list per item (W8.0) -- N=1 is a list of one, not a separate
code path.

Pure, no I/O (stdlib + the existing :class:`RegionCandidate` dataclass
only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.services.detection.cascade_detect import RegionCandidate
from src.services.detection.geometry import iou as box_iou


if TYPE_CHECKING:
    from collections.abc import Sequence


DropReason = Literal['below_min_score', 'nms', 'over_max']


@dataclass(frozen=True)
class SelectionResult:
    """The outcome of :func:`select_region_candidates`.

    ``dropped`` records every candidate NOT selected, paired with the
    reason it was cut -- used by ``POST /region_profiles/test`` to show an
    operator why a candidate didn't make it (§5.2).
    """

    selected: list[RegionCandidate]
    dropped: list[tuple[RegionCandidate, DropReason]]


def select_region_candidates(
    cands: Sequence[RegionCandidate],
    *,
    min_score: float,
    iou: float,
    max_n: int,
) -> SelectionResult:
    """Floor, deterministic sort, greedy class-agnostic NMS, then cap.

    1. Drop candidates with ``score < min_score`` (``below_min_score``).
    2. Sort by ``(-score, x1, y1)`` -- a deterministic tie-break so two
       equal-score candidates always order the same way.
    3. Greedy NMS: keep a candidate unless its IoU (crop frame) with an
       already-kept one is ``> iou`` (``nms``).
    4. Keep the first ``max_n`` (``over_max``).

    There is no containment rule -- a box enclosing two smaller boxes is
    resolved by the VLM and the sanity gate, not by geometry heuristics.
    """
    dropped: list[tuple[RegionCandidate, DropReason]] = []
    floor_passed: list[RegionCandidate] = []
    for c in cands:
        if c.score < min_score:
            dropped.append((c, 'below_min_score'))
        else:
            floor_passed.append(c)

    ordered = sorted(floor_passed, key=lambda c: (-c.score, c.bbox_norm[0], c.bbox_norm[1]))

    kept: list[RegionCandidate] = []
    for c in ordered:
        if any(box_iou(c.bbox_norm, k.bbox_norm) > iou for k in kept):
            dropped.append((c, 'nms'))
            continue
        kept.append(c)

    selected = kept[:max_n]
    dropped.extend((c, 'over_max') for c in kept[max_n:])

    return SelectionResult(selected=selected, dropped=dropped)


__all__ = ['DropReason', 'RegionCandidate', 'SelectionResult', 'select_region_candidates']
