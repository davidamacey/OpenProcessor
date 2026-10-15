"""Cohort images, region records and the index reads behind the region evaluator.

Turns an item's ``region_boxes`` list into scoreable :class:`RegionRecord`
rows and reads them (and the images-index ids) for a ground-truth cohort.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config.region_rejection import REJECT_REASON_SANITY_PREFIX
from src.config.region_state import RegionStatus
from src.domain.region_boxes import RegionBox, read_boxes
from src.services.curation.export_support import scroll_hits
from src.services.curation.region_eval_geometry import Box, as_box


if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.config.curation import CurationConfig
    from src.config.region_fields import RegionFields


NO_STATUS = '(none)'
NO_DETECTOR = '(none)'
TERMS_CHUNK = 1000


@dataclass
class CohortImage:
    """One ground-truth frame. ``key`` is the local path the dataset is read from."""

    key: str
    server_path: str
    gt: list[Box]
    split: str = ''
    image_id: str | None = None

    @property
    def positive(self) -> bool:
        return bool(self.gt)


@dataclass
class RegionRecord:
    crop_id: str
    status: str
    detector: str
    score: float | None
    box: Box | None  # source frame; None when the item carries no region box


def normalize_status(raw: Any) -> str:
    if raw is None or raw == '':
        return NO_STATUS
    return str(raw.value if isinstance(raw, RegionStatus) else raw)


# Per-box ``state`` -> the RegionStatus-shaped vocabulary this
# evaluator has always scored against. ``proposed`` (awaiting verification)
# is treated as a pending value, same as the item-level pending statuses.
_BOX_STATE_TO_STATUS: dict[str, str] = {
    'accepted': RegionStatus.DETECTED.value,
    'rejected': RegionStatus.VERIFY_REJECTED.value,
    RegionStatus.FALSE_POSITIVE.value: RegionStatus.FALSE_POSITIVE.value,
    'proposed': RegionStatus.PENDING_VERIFICATION.value,
}


def _box_status(box: RegionBox) -> str:
    """The status a box is scored under. A box a sanity gate rejected
    (``sanity_reject:<why>``) is an item-level ``detection_failed``, not a
    verifier ``verify_rejected`` -- the two are different failures."""
    if box.state == 'rejected' and (box.rejection_reason or '').startswith(
        REJECT_REASON_SANITY_PREFIX
    ):
        return RegionStatus.DETECTION_FAILED.value
    return _BOX_STATE_TO_STATUS.get(box.state, box.state)


def region_records(
    src: dict[str, Any], fields: RegionFields, doc_id: str = ''
) -> list[RegionRecord]:
    """One :class:`RegionRecord` per box in the item's ``region_boxes`` list.

    An item with no boxes at all (still pending, or a terminal box-less
    status like ``no_region_visible`` / ``detection_failed``) falls back to
    a single box-less record built from the item-level
    ``RegionFields.status`` field -- the only case where a region status
    exists without any entry in the box list.
    """
    crop_id = str(src.get('crop_id') or doc_id)
    boxes = read_boxes(src, fields)
    if not boxes:
        return [
            RegionRecord(
                crop_id=crop_id,
                status=normalize_status(src.get(fields.status)),
                detector=NO_DETECTOR,
                score=None,
                box=None,
            )
        ]
    return [
        RegionRecord(
            crop_id=crop_id,
            status=_box_status(b),
            detector=str(b.detector or NO_DETECTOR),
            score=b.score,
            box=as_box(list(b.bbox_norm)),
        )
        for b in boxes
    ]


def _chunks(values: list[str], n: int = TERMS_CHUNK) -> Iterable[list[str]]:
    for i in range(0, len(values), n):
        yield values[i : i + n]


async def resolve_image_ids(
    opensearch: Any, cohort: list[CohortImage], *, config: CurationConfig
) -> None:
    """Fill ``image_id`` from the images index for entries that lack one."""
    todo = sorted({c.server_path for c in cohort if not c.image_id})
    found: dict[str, str] = {}
    for chunk in _chunks(todo):
        hits = await scroll_hits(
            opensearch,
            index=config.images_index,
            query={'terms': {'image_path': chunk}},
            source=['image_id', 'image_path'],
        )
        for hit in hits:
            src = hit.get('_source') or {}
            if src.get('image_path') and src.get('image_id'):
                found[str(src['image_path'])] = str(src['image_id'])
    for c in cohort:
        if not c.image_id:
            c.image_id = found.get(c.server_path)


async def fetch_regions(
    opensearch: Any,
    image_ids: Iterable[str],
    *,
    config: CurationConfig,
    fields: RegionFields,
) -> dict[str, list[RegionRecord]]:
    """Every item on the given frames, as region records keyed by image_id."""
    source = [
        'crop_id',
        'image_id',
        'image_path',
        'bbox_norm',
        fields.status,
        fields.boxes,
    ]
    out: dict[str, list[RegionRecord]] = defaultdict(list)
    for chunk in _chunks(sorted(set(image_ids))):
        hits = await scroll_hits(
            opensearch,
            index=config.items_index,
            query={'terms': {'image_id': chunk}},
            source=source,
        )
        for hit in hits:
            src = hit.get('_source') or {}
            out[str(src.get('image_id'))].extend(region_records(src, fields, str(hit.get('_id'))))
    return dict(out)
