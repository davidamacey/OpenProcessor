"""Selection and dedup for the full-image SAM 3 pass: one pure function.

:func:`select_open_vocab_hits` is used by the runner that writes items, by
the per-image test route and by the reprocess dry run's accounting, so what
a test shows is what a run keeps. Per target it applies the score floor, the
min/max box area, NMS and the instance cap (the same
:func:`~src.services.detection.region_candidates.select_region_candidates`
the crop stage uses), then a class-agnostic NMS across targets that share a
class name, then the dedup against boxes already on the image:

* a LOCKED existing item (human or imported, see the lock rule) overlapping
  a hit at IoU >= :data:`LOCKED_OVERLAP_IOU`, whatever its class, makes the
  hit ``skipped_locked``: never written, the locked item never edited;
* an existing item of the SAME class name at IoU >= ``dedup_iou`` wins
  (``agree_existing``): no duplicate is created;
* any other overlap (a different class name on a machine item) keeps both.

Pure: no I/O, deterministic for equal inputs. Class identity is by NAME
(:func:`~src.utils.class_names.normalize_class_name`, the one name-equality
rule), never by registry id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.services.detection.geometry import iou
from src.services.detection.region_candidates import select_region_candidates
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.services.detection.cascade_detect import RegionCandidate

#: A locked box this much covered by a hit blocks the hit, whatever its class.
LOCKED_OVERLAP_IOU = 0.8

DropReason = Literal[
    'below_min_score',
    'too_small',
    'too_large',
    'nms',
    'over_max',
    'cross_target_nms',
    'agree_existing',
    'skipped_locked',
]


@dataclass(frozen=True)
class TargetRules:
    """What one target keeps (the numeric part of an ``open_vocab`` target)."""

    prompt: str
    class_name: str
    min_score: float
    min_area_frac: float
    max_area_frac: float
    max_instances: int


@dataclass(frozen=True)
class ExistingBox:
    """A box already on the image. ``locked`` is the lock rule's verdict
    (:func:`~src.services.curation.reprocess_locks.item_locked`)."""

    bbox_norm: tuple[float, float, float, float]
    class_name: str | None
    locked: bool


@dataclass(frozen=True)
class Hit:
    """One surviving SAM 3 candidate with the target that produced it."""

    candidate: RegionCandidate
    prompt: str
    class_name: str


@dataclass(frozen=True)
class OpenVocabSelection:
    kept: list[Hit]
    dropped: list[tuple[Hit, DropReason]]


def _norm(name: str | None) -> str:
    return normalize_class_name(name or '')


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _per_target(
    rules: TargetRules, cands: Sequence[RegionCandidate], dedup_iou: float
) -> tuple[list[Hit], list[tuple[Hit, DropReason]]]:
    def hit(c: RegionCandidate) -> Hit:
        return Hit(candidate=c, prompt=rules.prompt, class_name=rules.class_name)

    dropped: list[tuple[Hit, DropReason]] = []
    sized: list[RegionCandidate] = []
    for c in cands:
        area = _area(c.bbox_norm)
        if area < rules.min_area_frac:
            dropped.append((hit(c), 'too_small'))
        elif area > rules.max_area_frac:
            dropped.append((hit(c), 'too_large'))
        else:
            sized.append(c)
    sel = select_region_candidates(
        sized, min_score=rules.min_score, iou=dedup_iou, max_n=rules.max_instances
    )
    dropped.extend((hit(c), reason) for c, reason in sel.dropped)
    return [hit(c) for c in sel.selected], dropped


def select_open_vocab_hits(
    per_target: Sequence[tuple[TargetRules, Sequence[RegionCandidate]]],
    existing: Sequence[ExistingBox],
    *,
    dedup_iou: float,
) -> OpenVocabSelection:
    """Floor, size, NMS, cap per target; cross-target NMS; dedup vs ``existing``."""
    dropped: list[tuple[Hit, DropReason]] = []
    pooled: list[Hit] = []
    for rules, cands in per_target:
        kept, drops = _per_target(rules, cands, dedup_iou)
        pooled.extend(kept)
        dropped.extend(drops)

    pooled.sort(key=lambda h: (-h.candidate.score, *h.candidate.bbox_norm[:2], h.prompt))
    survivors: list[Hit] = []
    for h in pooled:
        key = _norm(h.class_name)
        if any(
            _norm(k.class_name) == key
            and iou(h.candidate.bbox_norm, k.candidate.bbox_norm) > dedup_iou
            for k in survivors
        ):
            dropped.append((h, 'cross_target_nms'))
            continue
        survivors.append(h)

    kept_hits: list[Hit] = []
    for h in survivors:
        box = h.candidate.bbox_norm
        key = _norm(h.class_name)
        if any(e.locked and iou(box, e.bbox_norm) >= LOCKED_OVERLAP_IOU for e in existing):
            dropped.append((h, 'skipped_locked'))
        elif any(
            _norm(e.class_name) == key and iou(box, e.bbox_norm) >= dedup_iou for e in existing
        ):
            dropped.append((h, 'agree_existing'))
        else:
            kept_hits.append(h)
    return OpenVocabSelection(kept=kept_hits, dropped=dropped)


__all__ = [
    'LOCKED_OVERLAP_IOU',
    'DropReason',
    'ExistingBox',
    'Hit',
    'OpenVocabSelection',
    'TargetRules',
    'select_open_vocab_hits',
]
