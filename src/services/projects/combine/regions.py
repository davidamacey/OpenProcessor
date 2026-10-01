"""A source class mapped to ``region`` (W10.7 semantics, for a project
source): its items stop being items and become region boxes on the item that
contains them, or a standalone region item when nothing does."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.region_source import CANDIDATE_IMPORT
from src.config.region_state import RegionStatus
from src.services.curation.ingest_class_sources import (
    LABEL_IMPORT_CLASS_SOURCE,
    LABEL_SOURCE_IMPORT,
)
from src.services.curation.region_boxes import (
    RegionBox,
    boxes_write_fields,
    derive_status,
    next_box_id,
    read_boxes,
)
from src.services.detection.geometry import crop_id as make_crop_id


if TYPE_CHECKING:
    from src.config.region_fields import RegionFields

DETECTOR = 'combine'
Box = tuple[float, float, float, float]


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _contained(inner: Box, outer: Box) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = _area(inner)
    return (ix * iy) / area if area > 0 else 0.0


def parent_of(box: Box, parents: list[dict[str, Any]], containment: float) -> dict[str, Any] | None:
    """The smallest parent that holds at least ``containment`` of ``box``."""

    def frame(doc: dict[str, Any]) -> Box:
        x1, y1, x2, y2 = doc['bbox_norm']
        return (x1, y1, x2, y2)

    fits = [p for p in parents if _contained(box, frame(p)) >= containment]
    if not fits:
        return None
    return min(fits, key=lambda p: (_area(frame(p)), p['crop_id']))


def _state_fields(
    boxes: list[RegionBox], current: dict[str, Any], F: RegionFields, job_id: str, now: str
) -> dict[str, Any]:
    fields = boxes_write_fields(boxes, current_src=current, F=F, set_complete=True)
    fields[F.status] = derive_status(boxes, empty_status=RegionStatus.NO_REGION_VISIBLE).value
    fields[F.validated] = all(b.state == 'accepted' for b in boxes)
    fields[F.label_source] = 'import'
    fields[F.verifier] = DETECTOR
    fields[F.verifier_version] = job_id
    fields[F.verified_at] = now
    return fields


def _box(source_item: dict[str, Any], box_id: str, job_id: str, now: str) -> RegionBox:
    validated = bool(source_item.get('class_validated'))
    return RegionBox(
        box_id=box_id,
        bbox_norm=tuple(source_item['bbox_norm']),  # type: ignore[arg-type]
        state='accepted' if validated else 'proposed',
        score=1.0,
        detector=DETECTOR,
        detector_version=job_id,
        source=CANDIDATE_IMPORT,
        detected_at=now,
    )


def attach_regions(
    parents: list[dict[str, Any]],
    region_items: list[dict[str, Any]],
    *,
    image_id: str,
    image_path: str,
    fields: RegionFields,
    job_id: str,
    origin_project: str,
    containment: float,
    now: str,
) -> tuple[int, list[dict[str, Any]]]:
    """Attach each region item to its parent (mutating the parent docs in
    place) and return ``(boxes attached, standalone region docs)``."""
    attached = 0
    standalone: list[dict[str, Any]] = []
    for item in region_items:
        bbox = tuple(item['bbox_norm'])
        parent = parent_of(bbox, parents, containment)  # type: ignore[arg-type]
        if parent is None:
            doc = _standalone(item, image_id, image_path, fields, job_id, origin_project, now)
            standalone.append(doc)
            continue
        stored = read_boxes(parent, fields)
        seq = int(parent.get(fields.box_seq) or 0)
        parent.update(
            _state_fields(
                [*stored, _box(item, next_box_id(stored, seq=seq), job_id, now)],
                parent,
                fields,
                job_id,
                now,
            )
        )
        attached += 1
    return attached, standalone


def _standalone(
    item: dict[str, Any],
    image_id: str,
    image_path: str,
    fields: RegionFields,
    job_id: str,
    origin_project: str,
    now: str,
) -> dict[str, Any]:
    bbox = list(item['bbox_norm'])
    doc: dict[str, Any] = {
        'crop_id': make_crop_id(image_id, bbox),
        'image_id': image_id,
        'image_path': image_path,
        'bbox_norm': bbox,
        'confidence': 1.0,
        'class_source': LABEL_IMPORT_CLASS_SOURCE,
        'label_source': LABEL_SOURCE_IMPORT,
        'class_validated': False,
        'test_holdout': bool(item.get('test_holdout')),
        'source': item.get('source'),
        'created_at': now,
        'updated_at': now,
        'import_ids': [job_id],
        'origin_project': origin_project,
        'origin_item_id': str(item.get('crop_id')),
        'origin_image_id': str(item.get('image_id')),
        'import_standalone_region': True,
    }
    doc.update(_state_fields([_box(item, 'b1', job_id, now)], {}, fields, job_id, now))
    return {k: v for k, v in doc.items() if v is not None}


__all__ = ['attach_regions', 'parent_of']
