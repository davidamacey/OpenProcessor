"""The payload of each claim kind (15a section 3.3): typed, validated on write and on read.

``extra='forbid'`` and frozen: a payload says exactly what the person decided, nothing a writer
happened to carry along. The bbox of a payload is in image coordinates (``x1, y1, x2, y2``, each
in ``[0, 1]``) in the frame of the photo's decode (``subject_frame`` on the claim).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.domain.claim_kinds import ClaimKind
from src.domain.region_boxes import BOX_STATES


Box4 = tuple[float, float, float, float]

# The strings a person's class write can carry (class_sources.HUMAN_LABEL_SOURCES plus the bulk
# accept's and the import's). The services module keeps its own copy until the claim-writing lane
# makes it import this one.
HUMAN_LABEL_SOURCES: tuple[str, ...] = ('human', 'human_confirmed', 'new_class_proposal')
CLASS_LABEL_SOURCES: tuple[str, ...] = (*HUMAN_LABEL_SOURCES, 'human_bulk_accept', 'import')
CLASS_SET_VIA = Literal['single', 'batch', 'bulk_accept', 'resolve', 'import', 'combine']


def _checked_box(value: Box4) -> Box4:
    x1, y1, x2, y2 = value
    if not all(0.0 <= v <= 1.0 for v in value):
        msg = f'bbox {value} is outside [0, 1]'
        raise ValueError(msg)
    if x2 <= x1 or y2 <= y1:
        msg = f'bbox {value} is degenerate'
        raise ValueError(msg)
    return value


class Payload(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class BoxPayload(Payload):
    bbox: Box4

    @field_validator('bbox')
    @classmethod
    def _bbox_in_image(cls, value: Box4) -> Box4:
        return _checked_box(value)


class Origin(Payload):
    project: str
    crop_id: str
    image_id: str


class MachineClass(Payload):
    """The machine class that would have won when an item was frozen."""

    class_name: str
    source: str
    confidence: float | None = None


class ClassProfile(Payload):
    """``class_detector`` and ``class_detector_version`` of the wire item (M19)."""

    detector: str | None = None
    detector_version: str | None = None


class ItemDrawn(BoxPayload):
    request_id: str | None = None
    standalone_region: bool = False
    origin: Origin | None = None


class GeometrySet(BoxPayload):
    pass


class ClassSet(Payload):
    class_name: str = Field(min_length=1)
    class_id_hint: int | None = None
    label_source: str
    via: CLASS_SET_VIA
    labeler: str
    profile: ClassProfile = ClassProfile()

    @field_validator('label_source')
    @classmethod
    def _known_label_source(cls, value: str) -> str:
        if value not in CLASS_LABEL_SOURCES:
            msg = f'label_source {value!r} is not one of {CLASS_LABEL_SOURCES}'
            raise ValueError(msg)
        return value


class Empty(Payload):
    """class_cleared, restored, review_dismissed, review_restored, withdraw."""


class Excluded(Payload):
    reason: str


class NeedsNewClass(Payload):
    note: str | None = None


class ClassMoved(Payload):
    partition_id: str
    cluster_uid: str


class VlmDismissed(Payload):
    class_name: str | None = None
    class_id_hint: int | None = None


class ProbeDismissed(Payload):
    class_name: str
    probe_analyser_id: int


class HoldoutFrozen(Payload):
    freeze_id: str
    # Required, nullable: ``None`` is "frozen as no class" (M1). A missing key is refused.
    machine_class: MachineClass | None


class AuditSampled(Payload):
    batch_id: str
    label_name: str
    label_source: str


class RegionStatusSet(Payload):
    status: Literal['no_region_visible']
    label_source: str
    region_seq: int = Field(ge=0)


class BoxCreated(BoxPayload):
    state: str
    text: str | None = None
    region_seq: int = Field(ge=0)

    @field_validator('state')
    @classmethod
    def _known_state(cls, value: str) -> str:
        return _box_state(value)


class BoxMoved(BoxPayload):
    bbox_correct: bool | None = None
    region_seq: int = Field(ge=0)


class BoxStateSet(Payload):
    state: str
    rejection_reason: str | None = None
    region_seq: int = Field(ge=0)

    @field_validator('state')
    @classmethod
    def _known_state(cls, value: str) -> str:
        return _box_state(value)


class BoxTextSet(Payload):
    text: str  # an explicit empty text is a reading of "no text"
    region_seq: int = Field(ge=0)


class BoxDeleted(Payload):
    region_seq: int = Field(ge=0)


class CombineConflict(Payload):
    origins: tuple[str, ...]
    merged_origins: tuple[str, ...]


class DuplicateVerdict(Payload):
    level: Literal['image', 'crop']
    decision: Literal['same', 'different']


def _box_state(value: str) -> str:
    if value not in BOX_STATES:
        msg = f'box state {value!r} is not one of {BOX_STATES}'
        raise ValueError(msg)
    return value


PAYLOAD_MODELS: dict[ClaimKind, type[Payload]] = {
    ClaimKind.ITEM_DRAWN: ItemDrawn,
    ClaimKind.GEOMETRY_SET: GeometrySet,
    ClaimKind.CLASS_SET: ClassSet,
    ClaimKind.CLASS_CLEARED: Empty,
    ClaimKind.CLASS_MOVED: ClassMoved,
    ClaimKind.HOLDOUT_FROZEN: HoldoutFrozen,
    ClaimKind.EXCLUDED: Excluded,
    ClaimKind.RESTORED: Empty,
    ClaimKind.REVIEW_DISMISSED: Empty,
    ClaimKind.REVIEW_RESTORED: Empty,
    ClaimKind.VLM_DISMISSED: VlmDismissed,
    ClaimKind.PROBE_DISMISSED: ProbeDismissed,
    ClaimKind.NEEDS_NEW_CLASS: NeedsNewClass,
    ClaimKind.AUDIT_SAMPLED: AuditSampled,
    ClaimKind.COMBINE_CONFLICT: CombineConflict,
    ClaimKind.REGION_STATUS_SET: RegionStatusSet,
    ClaimKind.BOX_CREATED: BoxCreated,
    ClaimKind.BOX_MOVED: BoxMoved,
    ClaimKind.BOX_STATE_SET: BoxStateSet,
    ClaimKind.BOX_TEXT_SET: BoxTextSet,
    ClaimKind.BOX_DELETED: BoxDeleted,
    ClaimKind.DUPLICATE_VERDICT: DuplicateVerdict,
    ClaimKind.WITHDRAW: Empty,
}


class PayloadError(ValueError):
    """A payload that does not fit its kind (the row is never written)."""


def parse_payload(kind: ClaimKind, data: dict[str, Any]) -> Payload:
    """The typed payload of ``kind`` from its stored JSON, or :class:`PayloadError`."""
    try:
        return PAYLOAD_MODELS[kind].model_validate(data)
    except ValidationError as exc:
        msg = f'{kind.value}: {exc}'
        raise PayloadError(msg) from exc


def payload_json(payload: Payload) -> dict[str, Any]:
    """The JSON object stored in ``claims.payload``."""
    return payload.model_dump(mode='json', exclude_none=False)
