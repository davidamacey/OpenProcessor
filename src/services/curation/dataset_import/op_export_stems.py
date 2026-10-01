"""Resolve OpenProcessor-export stems to docs the project already holds
(W10.2.2), so a re-import into the project that exported the dataset
attaches labels to the original image/item instead of ingesting the
exporter's resized copy.

* ``frame`` stems (multi-class and whole-frame single-class exports) are
  ``image_id`` values: the images doc with that id.
* ``item_crop`` stems are ``crop_id`` values: the items doc with that id; its
  boxes are in the crop's frame and are projected into the source frame with
  :func:`~src.services.detection.cascade_detect.crop_norm_to_source_norm`.

A stem is only ever used as a document id here, never as a path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from src.services.curation.dataset_import.scan import LabelBox
from src.services.detection.cascade_detect import crop_norm_to_source_norm


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.services.curation.dataset_import.issues import IssueCollector
    from src.services.curation.dataset_import.scan import ScanEntry

_MGET_CHUNK = 500


@dataclass(frozen=True)
class StemResolution:
    kind: Literal['frame', 'item_crop']
    image_id: str
    image_path: str
    crop_id: str | None = None
    crop_bbox_norm: tuple[float, float, float, float] | None = None
    """``item_crop``: the item's own box in the source frame (the parent the
    crop was cut from)."""


async def _mget_sources(
    opensearch: Any, ids: Sequence[str], *, index: str, fields: list[str]
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), _MGET_CHUNK):
        chunk = ids[start : start + _MGET_CHUNK]
        resp = await opensearch.mget(
            body={'docs': [{'_id': i, '_index': index, '_source': fields} for i in chunk]},
            index=index,
        )
        for doc in resp.get('docs') or []:
            if doc.get('found') and isinstance(doc.get('_source'), dict):
                found[str(doc['_id'])] = doc['_source']
    return found


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


async def resolve_stems(
    opensearch: Any,
    entries: Sequence[ScanEntry],
    *,
    images_index: str,
    items_index: str,
    issues: IssueCollector | None = None,
) -> dict[str, StemResolution]:
    """``{source_stem: StemResolution}`` for every entry whose stem names a
    doc this project already holds. Unresolved stems are simply absent (the
    caller ingests that file as a new image). With ``issues``, records
    ``images_reused_by_stem`` (info) and ``item_crop_export_imported_as_frames``
    (warning, an unresolved ``item_crop`` stem)."""
    frame_stems = sorted({e.source_stem for e in entries if e.stem_kind == 'frame'})
    crop_stems = sorted({e.source_stem for e in entries if e.stem_kind == 'item_crop'})
    resolved: dict[str, StemResolution] = {}

    images = await _mget_sources(
        opensearch, frame_stems, index=images_index, fields=['image_id', 'image_path']
    )
    for stem, src in images.items():
        if src.get('image_id') == stem and src.get('image_path'):
            resolved[stem] = StemResolution('frame', stem, str(src['image_path']))

    items = await _mget_sources(
        opensearch,
        crop_stems,
        index=items_index,
        fields=['crop_id', 'image_id', 'image_path', 'bbox_norm'],
    )
    for stem, src in items.items():
        bbox = _bbox(src.get('bbox_norm'))
        if src.get('crop_id') == stem and src.get('image_id') and src.get('image_path') and bbox:
            resolved[stem] = StemResolution(
                'item_crop', str(src['image_id']), str(src['image_path']), stem, bbox
            )

    if issues is not None:
        for entry in entries:
            if entry.source_stem in resolved:
                issues.add('images_reused_by_stem', file=entry.rel_path)
            elif entry.stem_kind == 'item_crop':
                issues.add('item_crop_export_imported_as_frames', file=entry.rel_path)
    return resolved


def project_item_crop_boxes(
    boxes: Sequence[LabelBox], resolution: StemResolution
) -> list[LabelBox]:
    """Project an ``item_crop`` entry's crop-frame boxes into the source
    frame of the item the crop was cut from. Boxes of any other resolution
    kind are returned unchanged."""
    if resolution.kind != 'item_crop' or resolution.crop_bbox_norm is None:
        return list(boxes)
    parent = resolution.crop_bbox_norm
    return [LabelBox(b.dataset_class, crop_norm_to_source_norm(b.bbox_norm, parent)) for b in boxes]


__all__ = ['StemResolution', 'project_item_crop_boxes', 'resolve_stems']
