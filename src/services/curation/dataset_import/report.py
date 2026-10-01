"""The counters an import accumulates, additive per chunk so a resumed run
can rebuild them by summing ``chunks_done.jsonl``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


MAX_DISAGREEMENT_SAMPLES = 1000

_COUNTERS = (
    'images_created',
    'images_reused',
    'images_failed',
    'images_skipped',
    'items_created',
    'items_updated',
    'items_noop',
    'labels_written',
    'boxes_written',
    'standalone_regions',
    'negatives',
    'unlabeled',
    'parents_detected',
    'proposals_created',
    'proposals_merged',
    'holdout_frozen',
    'label_conflicts_locked',
    'items_reconciled_removed',
)


@dataclass
class ImportReport:
    images_created: int = 0
    images_reused: int = 0
    images_failed: int = 0
    images_skipped: int = 0
    items_created: int = 0
    items_updated: int = 0
    items_noop: int = 0
    labels_written: int = 0
    boxes_written: int = 0
    standalone_regions: int = 0
    negatives: int = 0
    unlabeled: int = 0
    parents_detected: int = 0
    proposals_created: int = 0
    proposals_merged: int = 0
    holdout_frozen: int = 0
    label_conflicts_locked: int = 0
    items_reconciled_removed: int = 0
    disagreement_counts: dict[str, int] = field(
        default_factory=lambda: {'mismatches': 0, 'missed_labels': 0, 'unmatched_detections': 0}
    )
    disagreement_samples: list[dict[str, Any]] = field(default_factory=list)

    def add(self, other: ImportReport) -> None:
        for name in _COUNTERS:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for key, value in other.disagreement_counts.items():
            self.disagreement_counts[key] = self.disagreement_counts.get(key, 0) + value
        room = MAX_DISAGREEMENT_SAMPLES - len(self.disagreement_samples)
        if room > 0:
            self.disagreement_samples.extend(other.disagreement_samples[:room])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ImportReport:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


__all__ = ['MAX_DISAGREEMENT_SAMPLES', 'ImportReport']
