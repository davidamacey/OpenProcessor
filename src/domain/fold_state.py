"""What the fold writes: the ``item_state`` row of an item and the ``box_state`` rows of its boxes.

Columns are the ones 15a section 2.4 lists, plus the wire-relevant values the wire serializer
derives from them (the class name, the lock flags). The class name is looked up from the
registry, never stored; it is on the value so the wire layer and the tests read one object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from datetime import datetime

    from src.domain.claim_payloads import Box4

GeometrySource = Literal['detector', 'human', 'import']


@dataclass(frozen=True)
class FoldSettings:
    """Deployment values the fold needs and must not read from the environment."""

    primary_detector_name: str
    precedence: tuple[str, ...]
    human_detector_name: str = 'human'
    human_detector_version: str = '1'
    confusables: tuple[tuple[str, str], ...] = ()
    """The active profile's confusable character map, as pairs (``text_fold`` of typed text)."""


@dataclass(frozen=True)
class BoxState:
    box_id: str
    bbox: Box4
    state: str
    ordinal: int
    source: str | None = None
    score: float | None = None
    detector: str | None = None
    detector_version: str | None = None
    bbox_correct: bool | None = None
    confidence: str | None = None
    rejection_reason: str | None = None
    text: str | None = None
    text_source: str | None = None
    text_choice: str | None = None
    text_confidence: float | None = None
    text_disagreement: bool | None = None
    text_fold: str | None = None
    detected_at: str | None = None
    from_claims_only: bool = False
    """The box exists only through a claim: no machine box matched it (``box_key`` NULL)."""
    locked: bool = False
    """Wire ``locked``: today's ``is_locked_box`` on this folded box."""


@dataclass(frozen=True)
class ItemState:
    crop_id: str
    photo_key: int
    bbox: Box4
    area: float
    rank: int
    geometry_source: GeometrySource
    geometry_revision: int
    class_id: int | None
    class_name: str | None
    class_source: str | None
    label_source: str | None
    class_validated: bool
    class_unresolved: bool
    class_labeled_at: datetime | None
    class_confidence: float | None
    class_claim_id: int | None
    class_op_id: str | None
    class_opinion_source: str | None
    class_labeler: str | None
    class_detector: str | None
    class_detector_version: str | None
    vlm_suggestion_class_id: int | None
    vlm_suggestion_name: str | None
    vlm_suggestion_dismissed: bool
    needs_new_class: bool
    probe_dismissed_class: str | None
    excluded: bool
    excluded_reason: str | None
    excluded_at: datetime | None
    excluded_by_human: bool
    holdout: bool
    review_dismissed_at: datetime | None
    audit_sample: bool
    audit_outcome: str | None
    combine_conflict: bool
    import_ids: tuple[str, ...]
    region_status: str | None
    region_validated: bool
    region_verifier: str | None
    region_label_source: str | None
    region_count: int
    region_rejected_count: int
    region_max_score: float | None
    region_set_complete: bool | None
    region_revision: int
    region_seen_seq: int
    """The ``region_seq`` a person's next box or region claim records: the machine boxes shown."""
    moved_partition_id: str | None
    moved_cluster_uid: str | None
    created_at: datetime
    updated_at: datetime
    label_locked: bool
    """Wire ``label_locked``: today's ``is_locked_item`` on the folded state."""
    geometry_locked: bool
    """Wire ``geometry_locked``: today's ``is_editor_locked`` on the folded state."""


@dataclass(frozen=True)
class Conflict:
    """An import or combine claim that lost to a person's (``fold.conflicts``)."""

    crop_id: str
    kind: Literal['locked_conflict', 'combine_conflict']
    claim_id: int


@dataclass(frozen=True)
class ItemFold:
    state: ItemState
    boxes: tuple[BoxState, ...]
    conflicts: tuple[Conflict, ...]
