"""Builders shared by the fold tests: a clock, a registry, results, and a log that commits drafts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from src.config.ingest_profiles import ingest_primary_name
from src.domain.claim_builders import ItemView
from src.domain.claims import Claim, ClaimDraft, commit
from src.domain.fold import fold_item
from src.domain.fold_state import FoldSettings
from src.domain.results import (
    DEFAULT_PRECEDENCE,
    ClassOpinion,
    DetectionResult,
    ItemInputs,
    ItemRow,
    MachineBox,
    RegionResult,
    RegistryClass,
    RegistrySnapshot,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from src.domain.claim_payloads import Box4
    from src.domain.fold_state import ItemFold

BASE = datetime(2026, 1, 1, tzinfo=UTC)
PRIMARY = ingest_primary_name()
SETTINGS = FoldSettings(primary_detector_name=PRIMARY, precedence=DEFAULT_PRECEDENCE)
REGISTRY = RegistrySnapshot(
    (
        RegistryClass(1, 'car'),
        RegistryClass(2, 'truck'),
        RegistryClass(3, 'bus'),
        RegistryClass(4, 'van'),
    )
)
SOURCE_ANALYSER = {
    'classifier': 10,
    'vlm': 20,
    'assist_propagated': 30,
    'import_suggestion': 40,
    'adopted_detector': 50,
    'open_vocab_target': 60,
}
BOX: Box4 = (0.10, 0.10, 0.50, 0.50)
CROP = 'c1'
PHOTO = 7
FRAME = 1
VIEW = ItemView(crop_id=CROP, photo_key=PHOTO, frame=FRAME, bbox=BOX, machine_bbox=BOX)


def t(n: float) -> datetime:
    return BASE + timedelta(seconds=n)


def detection(
    *,
    bbox: Box4 = BOX,
    at: float = 0,
    kind: str = 'detector',
    class_name: str | None = 'car',
    import_id: str | None = None,
    analyser_id: int = 1,
) -> DetectionResult:
    return DetectionResult(
        analyser_kind=kind,  # type: ignore[arg-type]
        analyser_name=PRIMARY,
        analyser_version='1',
        bbox=bbox,
        created_at=t(at),
        class_name=class_name,
        import_id=import_id,
        analyser_id=analyser_id,
    )


def opinion(
    source: str,
    class_name: str | None,
    *,
    at: float = 1,
    outcome: str = 'matched',
    **extra: object,
) -> ClassOpinion:
    extra.setdefault('analyser_id', SOURCE_ANALYSER[source])
    return ClassOpinion(
        source=source,  # type: ignore[arg-type]
        analyser_name=PRIMARY if source == 'classifier' else source,
        outcome=outcome,  # type: ignore[arg-type]
        created_at=t(at),
        class_name=class_name,
        **extra,  # type: ignore[arg-type]
    )


def machine_box(
    box_id: str,
    bbox: Box4,
    *,
    state: str = 'proposed',
    seq: int = 1,
    **extra: object,
) -> MachineBox:
    return MachineBox(box_id=box_id, bbox=bbox, state=state, region_seq=seq, **extra)  # type: ignore[arg-type]


@dataclass
class Log:
    """The claim log of one item: drafts are committed with a rising id and time."""

    claims: list[Claim] = field(default_factory=list)
    clock: float = 10

    def add(self, drafts: Iterable[ClaimDraft]) -> list[Claim]:
        out = []
        for draft in drafts:
            self.clock += 1
            claim = commit(draft, claim_id=len(self.claims) + 1, at=t(self.clock))
            self.claims.append(claim)
            out.append(claim)
        return out


def inputs(
    log: Sequence[Claim] = (),
    *,
    detections: Sequence[DetectionResult] | None = None,
    opinions: Sequence[ClassOpinion] = (),
    boxes: Sequence[MachineBox] = (),
    region: RegionResult | None = None,
    item: ItemRow | None = None,
) -> ItemInputs:
    return ItemInputs(
        item=item or ItemRow(crop_id=CROP, photo_key=PHOTO, created_at=t(0)),
        frame=FRAME,
        detections=(detection(),) if detections is None else tuple(detections),
        opinions=tuple(opinions),
        region=region,
        boxes=tuple(boxes),
        claims=tuple(log),
    )


def folded(inp: ItemInputs, registry: RegistrySnapshot = REGISTRY) -> ItemFold:
    result = fold_item(inp, registry, SETTINGS)
    assert result is not None
    return result
