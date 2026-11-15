"""Everything the fold reads for a set of items, one query per table family, in key order.

``load_item_inputs`` returns the typed values :func:`src.domain.fold.fold_item` takes (plan 15a
section 4.1): the identity row with the photo's decode frame, the detections and class opinions with
the analysers that made them, the region result and machine boxes with their readings, and the item's
claims with the withdraws that name them. Nothing is decided here: a reading consensus is the
caller's function (:data:`TextConsensus`), because it lives in ``src/services/detection`` until the
wiring lane moves it to ``src/domain`` (15a, fold lane decision Q12).

Every function takes a session already inside ``project_tx``. The caller holds the item lock
(:func:`src.db.repos.items.lock_items`) before it reads, so a second writer waits and folds the first
one's committed inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import select

from src.db.models.analysis import Analyser
from src.db.models.boxes import BoxText, RegionBox
from src.db.models.classes import ProjectClass
from src.db.models.items import ClassOpinion, Detection, Item, ItemRegionResult
from src.db.models.photos import Photo
from src.db.repos.claims import load_claims_for_items
from src.db.repos.upserts import any_of
from src.domain.results import (
    AnalyserKind,
    ClassOpinion as OpinionResult,
    DetectionResult,
    ItemInputs,
    ItemRow,
    MachineBox,
    MachineText,
    RegionResult,
    RegistryClass,
    RegistrySnapshot,
)
from src.domain.wire_ids import to_wire


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.domain.claim_payloads import Box4


@dataclass(frozen=True)
class BoxReading:
    """One reader's reading of one box, with the analyser that made it (a ``box_texts`` row)."""

    reader: str
    analyser_name: str
    analyser_version: str
    text: str | None
    text_raw: str | None
    text_frame: str | None
    invalid: str | None
    confidence: float | None
    conf_char_min: float | None
    jurisdiction: str | None
    jurisdiction_conf: float | None


TextConsensus = Callable[[Sequence[BoxReading]], MachineText | None]
"""The resolved reading of a box from its readers (VLM and OCR); ``None`` is no reading."""


@dataclass(frozen=True)
class LoadedItem:
    inputs: ItemInputs
    box_keys: dict[str, int]
    """``region_boxes.box_key`` by machine box id: ``box_state.box_key`` points at it."""


def _box(values: Sequence[float]) -> Box4:
    a, b, c, d = values
    return (float(a), float(b), float(c), float(d))


async def load_registry(session: AsyncSession, project_id: uuid.UUID) -> RegistrySnapshot:
    rows = await session.execute(
        select(
            ProjectClass.class_id,
            ProjectClass.name,
            ProjectClass.deprecated,
            ProjectClass.merged_into,
        )
        .where(ProjectClass.project_id == project_id)
        .order_by(ProjectClass.class_id)
    )
    return RegistrySnapshot(
        tuple(RegistryClass(r.class_id, r.name, not r.deprecated, r.merged_into) for r in rows)
    )


async def _by_item(session: AsyncSession, statement: Any) -> dict[int, list[Any]]:
    grouped: dict[int, list[Any]] = {}
    for row in await session.execute(statement):
        grouped.setdefault(row.item_key, []).append(row)
    return grouped


def _detection(r: Any) -> DetectionResult:
    return DetectionResult(
        analyser_kind=cast('AnalyserKind', r.analyser_kind),
        analyser_name=r.analyser_name,
        analyser_version=r.analyser_version,
        bbox=_box(r.bbox),
        created_at=r.created_at,
        class_name=r.class_name,
        confidence=r.confidence,
        proposal_name=r.proposal_name,
        import_id=None if r.import_id is None else str(r.import_id),
        analyser_id=r.analyser_id,
    )


def _opinion(r: Any) -> OpinionResult:
    return OpinionResult(
        source=r.source,
        analyser_name=r.analyser_name,
        outcome=r.outcome,
        created_at=r.created_at,
        class_name=r.class_name,
        class_id_hint=r.class_id_hint,
        confidence=r.confidence,
        confidence_cat=r.confidence_cat,
        proposed_class=r.proposed_class,
        raw_class=r.raw_class,
        empty_reason=r.empty_reason,
        op_id=None if r.op_id is None else str(r.op_id),
        import_id=None if r.import_id is None else str(r.import_id),
        analyser_id=r.analyser_id,
    )


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


async def load_item_inputs(
    session: AsyncSession, item_keys: Iterable[int], *, text_consensus: TextConsensus
) -> dict[int, LoadedItem]:
    """The fold inputs of each item that exists, by ``item_key`` (an absent key was deleted)."""
    keys = sorted(set(item_keys))
    if not keys:
        return {}
    items = (
        await session.execute(
            select(
                Item.item_key, Item.crop_id, Item.photo_key, Item.created_at, Item.box_seq,
                Item.geometry_seq, Item.region_seq, Photo.frame,
            )
            .join(Photo, Photo.photo_key == Item.photo_key)
            .where(any_of(Item.item_key, keys))
            .order_by(Item.item_key)
        )
    ).all()  # fmt: skip
    detections = await _by_item(
        session,
        select(
            Detection.item_key, Detection.analyser_id, Detection.bbox, Detection.created_at,
            Detection.class_name, Detection.confidence, Detection.proposal_name,
            Detection.import_id, Analyser.kind.label('analyser_kind'),
            Analyser.name.label('analyser_name'), Analyser.version.label('analyser_version'),
        )
        .join(Analyser, Analyser.analyser_id == Detection.analyser_id)
        .where(any_of(Detection.item_key, keys))
        .order_by(Detection.item_key, Detection.detection_key),
    )  # fmt: skip
    opinions = await _by_item(
        session,
        select(
            ClassOpinion.item_key, ClassOpinion.source, ClassOpinion.analyser_id,
            ClassOpinion.outcome, ClassOpinion.created_at, ClassOpinion.class_name,
            ClassOpinion.class_id_hint, ClassOpinion.confidence, ClassOpinion.confidence_cat,
            ClassOpinion.proposed_class, ClassOpinion.raw_class, ClassOpinion.empty_reason,
            ClassOpinion.op_id, ClassOpinion.import_id, Analyser.name.label('analyser_name'),
        )
        .join(Analyser, Analyser.analyser_id == ClassOpinion.analyser_id)
        .where(any_of(ClassOpinion.item_key, keys))
        .order_by(ClassOpinion.item_key, ClassOpinion.source),
    )  # fmt: skip
    regions = await _by_item(
        session,
        select(
            ItemRegionResult.item_key, ItemRegionResult.status, ItemRegionResult.verifier,
            ItemRegionResult.set_complete, ItemRegionResult.detected_at,
        )
        .where(any_of(ItemRegionResult.item_key, keys))
        .order_by(ItemRegionResult.item_key),
    )  # fmt: skip
    boxes = await _by_item(
        session,
        select(
            RegionBox.item_key, RegionBox.box_key, RegionBox.box_id, RegionBox.region_seq,
            RegionBox.bbox, RegionBox.state, RegionBox.score, RegionBox.detector,
            RegionBox.detector_version, RegionBox.source, RegionBox.confidence,
            RegionBox.rejection_reason, RegionBox.detected_at,
        )
        .where(any_of(RegionBox.item_key, keys))
        .order_by(RegionBox.item_key, RegionBox.box_key),
    )  # fmt: skip
    readings: dict[int, list[BoxReading]] = {}
    for r in await session.execute(
        select(
            BoxText.box_key, BoxText.reader, BoxText.text, BoxText.text_raw, BoxText.text_frame,
            BoxText.invalid, BoxText.confidence, BoxText.conf_char_min, BoxText.jurisdiction,
            BoxText.jurisdiction_conf, Analyser.name.label('analyser_name'),
            Analyser.version.label('analyser_version'),
        )
        .join(RegionBox, RegionBox.box_key == BoxText.box_key)
        .join(Analyser, Analyser.analyser_id == BoxText.analyser_id)
        .where(any_of(RegionBox.item_key, keys))
        .order_by(BoxText.box_key, BoxText.reader)
    ):  # fmt: skip
        readings.setdefault(r.box_key, []).append(
            BoxReading(
                r.reader, r.analyser_name, r.analyser_version, r.text, r.text_raw, r.text_frame,
                r.invalid, r.confidence, r.conf_char_min, r.jurisdiction, r.jurisdiction_conf,
            )
        )  # fmt: skip
    claims = await load_claims_for_items(session, keys)

    loaded: dict[int, LoadedItem] = {}
    for it in items:
        key = it.item_key
        region = regions.get(key, [None])[0]
        machine = tuple(
            MachineBox(
                box_id=b.box_id,
                bbox=_box(b.bbox),
                state=b.state,
                region_seq=b.region_seq,
                score=b.score,
                detector=b.detector,
                detector_version=b.detector_version,
                source=b.source,
                confidence=b.confidence,
                rejection_reason=b.rejection_reason,
                detected_at=_iso(b.detected_at),
                text=text_consensus(readings[b.box_key]) if b.box_key in readings else None,
            )
            for b in boxes.get(key, [])
        )
        inputs = ItemInputs(
            item=ItemRow(
                crop_id=to_wire(it.crop_id),
                photo_key=it.photo_key,
                created_at=it.created_at,
                box_seq=it.box_seq,
                geometry_seq=it.geometry_seq,
                region_seq=it.region_seq,
            ),
            frame=it.frame,
            detections=tuple(_detection(d) for d in detections.get(key, [])),
            opinions=tuple(_opinion(o) for o in opinions.get(key, [])),
            region=None
            if region is None
            else RegionResult(
                region.status, region.verifier, region.set_complete, region.detected_at
            ),
            boxes=machine,
            claims=tuple(claims[key]),
        )
        loaded[key] = LoadedItem(inputs, {b.box_id: b.box_key for b in boxes.get(key, [])})
    return loaded
