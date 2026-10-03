"""The ingest detector's label vocabulary and the plan for seeding a registry from it.

Class identity is by NAME: the plan compares slugified detector labels with
registry class names and never aligns or assumes class ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from src.config.ingest_profiles import load_label_names
from src.utils.class_names import get_class_names, normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.config import DetectionProfile


SkipReason = Literal['exists', 'deprecated']
ConflictReason = Literal['duplicate_slug', 'unnamed_label']


@dataclass(frozen=True)
class DetectorLabel:
    """One detector class: its id, raw label and registry slug."""

    class_id: int
    name: str
    slug: str


@dataclass(frozen=True)
class SkippedLabel:
    label: DetectorLabel
    reason: SkipReason


@dataclass(frozen=True)
class ConflictLabel:
    label: DetectorLabel
    reason: ConflictReason


@dataclass
class SeedPlan:
    create: list[DetectorLabel]
    skipped: list[SkippedLabel]
    conflicts: list[ConflictLabel]
    unknown: list[str] = field(default_factory=list)


def detector_labels(profile: DetectionProfile) -> list[DetectorLabel]:
    """The profile's labels in class-id order, resolved the way ingest does
    (``labels_path``, else the model directory's own ``labels.txt``). Empty
    when the model has none. A configured but unreadable ``labels_path``
    raises ``OSError`` (a deployment error, not something to guess)."""
    if profile.labels_path:
        raw = dict(enumerate(load_label_names(profile.labels_path)))
    else:
        raw = get_class_names(profile.detector_model)
    if not raw:
        return []
    return [
        DetectorLabel(i, raw.get(i, ''), normalize_class_name(raw.get(i, '')))
        for i in range(max(raw) + 1)
    ]


def plan_seed(
    labels: Sequence[DetectorLabel],
    existing: Mapping[str, bool],
    requested: Sequence[str] | None,
) -> SeedPlan:
    """What to create, skip and report for a seed call.

    ``existing`` maps registry class names to their ``deprecated`` flag.
    ``requested`` limits the plan to those names (matched by slug); names the
    detector does not have are returned in ``unknown`` and nothing else is
    planned for them. A deprecated class is never resurrected or duplicated.
    """
    wanted: set[str] | None = None
    unknown: list[str] = []
    if requested is not None:
        wanted = {normalize_class_name(n) for n in requested}
        known = {label.slug for label in labels}
        unknown = [n for n in requested if normalize_class_name(n) not in known]
    plan = SeedPlan([], [], [], unknown)
    seen: set[str] = set()
    for label in labels:
        if wanted is not None and label.slug not in wanted:
            continue
        if not label.slug:
            plan.conflicts.append(ConflictLabel(label, 'unnamed_label'))
        elif label.slug in seen:
            plan.conflicts.append(ConflictLabel(label, 'duplicate_slug'))
        elif label.slug in existing:
            seen.add(label.slug)
            plan.skipped.append(
                SkippedLabel(label, 'deprecated' if existing[label.slug] else 'exists')
            )
        else:
            seen.add(label.slug)
            plan.create.append(label)
    return plan
