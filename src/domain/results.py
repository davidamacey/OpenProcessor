"""What the fold reads besides claims: machine results and the class registry, as plain values.

A repository loads these rows; the fold never sees a table. Every result names the analyser that
made it (R4); a row that carries ``op_id`` or ``import_id`` is not rebuildable (R2): a re-run
keeps it and only the undo of its operation or import removes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from datetime import datetime

    from src.domain.claim_payloads import Box4
    from src.domain.claims import Claim

AnalyserKind = Literal['detector', 'open_vocab', 'import']
OpinionSource = Literal[
    'classifier',
    'vlm',
    'assist_propagated',
    'import_suggestion',
    'adopted_detector',
    'open_vocab_target',
]
OpinionOutcome = Literal[
    'matched', 'unmatched', 'new_class_pending', 'reclassified', 'empty', 'low_conf', 'model'
]

# Plan 15 section 14, answer 5: the order in which machine opinions decide the effective class.
DEFAULT_PRECEDENCE: tuple[str, ...] = (
    'open_vocab_target',
    'vlm',
    'import_suggestion',
    'assist_propagated',
    'classifier',
    'adopted_detector',
    'detection',
)


@dataclass(frozen=True)
class ClassRef:
    class_id: int
    name: str


@dataclass(frozen=True)
class RegistryClass:
    class_id: int
    name: str
    active: bool = True
    merged_into: int | None = None


@dataclass(frozen=True)
class RegistrySnapshot:
    """The project's class registry at one moment (identity is the name; the id is a hint)."""

    classes: tuple[RegistryClass, ...] = ()

    def _end(self, cls: RegistryClass) -> RegistryClass:
        by_id = {c.class_id: c for c in self.classes}
        seen = {cls.class_id}
        while cls.merged_into is not None and cls.merged_into in by_id:
            cls = by_id[cls.merged_into]
            if cls.class_id in seen:
                break
            seen.add(cls.class_id)
        return cls

    def _reaches(self, start: RegistryClass, target_id: int) -> bool:
        by_id = {c.class_id: c for c in self.classes}
        seen = {start.class_id}
        cur = start
        while cur.merged_into is not None and cur.merged_into in by_id:
            if cur.merged_into == target_id:
                return True
            cur = by_id[cur.merged_into]
            if cur.class_id in seen:
                return False
            seen.add(cur.class_id)
        return False

    def resolve(self, name: str, hint: int | None = None) -> ClassRef | None:
        """The class a claim's ``name`` means now (15a section 3.3, M20); ``None`` is unresolved.

        The hint counts only when its class has the claim's name, or the named class was merged
        into the hint's class; a hint from anywhere else never selects a class.
        """
        key = normalize_class_name(name)
        named = [c for c in self.classes if normalize_class_name(c.name) == key]
        active = next((c for c in named if c.active and c.merged_into is None), None)
        chosen: RegistryClass | None = None
        hinted = next((c for c in self.classes if c.class_id == hint), None)
        if hinted is not None and (
            normalize_class_name(hinted.name) == key
            or any(self._reaches(c, hinted.class_id) for c in named)
        ):
            chosen = self._end(hinted)
        if chosen is None and active is not None:
            chosen = active
        if chosen is None and named:
            chosen = self._end(named[0])
        return None if chosen is None else ClassRef(chosen.class_id, chosen.name)


@dataclass(frozen=True)
class DetectionResult:
    analyser_kind: AnalyserKind
    analyser_name: str
    analyser_version: str
    bbox: Box4
    created_at: datetime
    class_name: str | None = None
    confidence: float | None = None
    proposal_name: str | None = None
    import_id: str | None = None
    analyser_id: int = 0


@dataclass(frozen=True)
class ClassOpinion:
    """One machine class opinion per item and source (``class_opinions``)."""

    source: OpinionSource
    analyser_name: str
    outcome: OpinionOutcome
    created_at: datetime
    class_name: str | None = None
    class_id_hint: int | None = None
    confidence: float | None = None
    confidence_cat: str | None = None
    proposed_class: str | None = None
    raw_class: str | None = None
    empty_reason: str | None = None
    op_id: str | None = None
    import_id: str | None = None
    analyser_id: int = 0


@dataclass(frozen=True)
class MachineText:
    """The reader consensus of a box (VLM and OCR readings, resolved by the text stage).

    The consensus function still lives in ``src/services/detection/region_text.py``; the lane
    that wires the fold moves it here. The fold takes its answer as an input.
    """

    text: str | None
    source: str | None = None
    choice: str | None = None
    disagreement: bool | None = None
    confidence: float | None = None
    fold: str | None = None


@dataclass(frozen=True)
class MachineBox:
    """One box of the region stage (or an import's proposed box) with its reading."""

    box_id: str
    bbox: Box4
    state: str
    region_seq: int
    score: float | None = None
    detector: str | None = None
    detector_version: str | None = None
    source: str | None = None
    bbox_correct: bool | None = None
    confidence: str | None = None
    rejection_reason: str | None = None
    detected_at: str | None = None
    text: MachineText | None = None


@dataclass(frozen=True)
class RegionResult:
    """The region stage's item-level outcome (``item_region_results``)."""

    status: str
    verifier: str | None = None
    set_complete: bool | None = None
    detected_at: datetime | None = None


@dataclass(frozen=True)
class ItemRow:
    """The identity row of an item and the sequences machine writes keep."""

    crop_id: str
    photo_key: int
    created_at: datetime
    box_seq: int = 0
    geometry_seq: int = 0
    region_seq: int = 0


@dataclass(frozen=True)
class ItemInputs:
    """Everything the fold needs for one item. ``claims`` holds every claim attached to it by
    :mod:`src.domain.claim_match`, withdraws included: the holdout rule reads withdrawn ones."""

    item: ItemRow
    frame: int
    detections: tuple[DetectionResult, ...] = ()
    opinions: tuple[ClassOpinion, ...] = ()
    region: RegionResult | None = None
    boxes: tuple[MachineBox, ...] = ()
    claims: tuple[Claim, ...] = field(default=())
