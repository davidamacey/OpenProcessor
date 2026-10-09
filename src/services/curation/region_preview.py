"""Preview helpers for the test-on-crop routes (W5, any_domain_plan.md §5).

Pure: no I/O. A leg's raw candidates become complete ``RegionBox`` wire
elements (so one renderer draws candidates and stored boxes alike) and an
item's preview is :func:`~src.services.curation.wire.serialize_item` of the
stored source with the write the worker would make laid over it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.curation.region_boxes import RegionBox
from src.services.curation.wire import region_box_to_wire, serialize_item
from src.services.detection.cascade_detect.candidate import RegionCandidate
from src.services.detection.cascade_detect.sanity import crop_norm_to_source_norm
from src.services.detection.region_candidates import select_region_candidates


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.config import DetectionProfile
    from src.services.detection.region_candidates import SelectionResult

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class LegCandidate:
    """One raw candidate of a leg, in the item-crop frame. ``mask_polygon``
    is the segmenter's mask outline (crop-frame points) when asked for."""

    bbox_norm: BBox
    score: float
    mask_iou: float | None = None
    mask_polygon: list[tuple[float, float]] | None = None


@dataclass(frozen=True)
class LegSelection:
    """The worker's own selection over a leg's raw candidates."""

    raw: list[LegCandidate]
    #: ``raw`` as the selection's objects, same order (the wire keys a
    #: candidate by its position in ``raw``).
    candidates: list[RegionCandidate]
    selection: SelectionResult


def select_leg_candidates(
    raw: Sequence[LegCandidate], *, source: str, profile: DetectionProfile, min_score: float
) -> LegSelection:
    """Floor / sort / NMS / cap ``raw`` exactly as the worker does
    (:func:`select_region_candidates` with the profile's knobs)."""
    candidates = [
        RegionCandidate(
            bbox_norm=c.bbox_norm, score=c.score, source=source, rectangularity=c.mask_iou
        )
        for c in raw
    ]
    selection = select_region_candidates(
        candidates,
        min_score=min_score,
        iou=profile.region_nms_iou,
        max_n=profile.max_regions_per_item,
    )
    return LegSelection(raw=list(raw), candidates=candidates, selection=selection)


def _project_polygon(polygon: Sequence[tuple[float, float]], parent: BBox) -> list[list[float]]:
    out: list[list[float]] = []
    for x, y in polygon:
        sx1, sy1, _sx2, _sy2 = crop_norm_to_source_norm((x, y, x, y), parent)
        out.append([sx1, sy1])
    return out


def candidate_wires(
    source: dict[str, Any],
    leg: LegSelection,
    *,
    detector: str,
    detector_version: str,
    box_source: str,
) -> list[dict[str, Any]]:
    """Every raw candidate as a complete box wire element plus
    ``candidate_index`` / ``selected`` / ``drop_reason`` / mask fields.

    ``source`` is the stored item doc (its ``bbox_norm`` is the frame the
    candidates are projected into). ``bbox_norm`` is the source-image frame,
    ``bbox_in_parent`` the item-crop frame the leg answered in, and the mask
    polygon is given in both.
    """
    parent = tuple(float(v) for v in source.get('bbox_norm') or (0.0, 0.0, 1.0, 1.0))
    parent_box: BBox = (parent[0], parent[1], parent[2], parent[3])
    selected_ids = {id(c) for c in leg.selection.selected}
    dropped = {id(c): reason for c, reason in leg.selection.dropped}
    wires: list[dict[str, Any]] = []
    for index, raw in enumerate(leg.raw):
        rc = leg.candidates[index]
        box = RegionBox(
            box_id='',
            bbox_norm=crop_norm_to_source_norm(raw.bbox_norm, parent_box),
            state='proposed',
            score=raw.score,
            detector=detector,
            detector_version=detector_version,
            source=box_source,
        )
        wire = region_box_to_wire(source, box)
        wire['box_id'] = None
        polygon = raw.mask_polygon
        wire.update(
            candidate_index=index,
            selected=id(rc) in selected_ids,
            drop_reason=dropped.get(id(rc)),
            mask_iou=raw.mask_iou,
            mask_polygon=_project_polygon(polygon, parent_box) if polygon else None,
            mask_polygon_in_parent=[[x, y] for x, y in polygon] if polygon else None,
        )
        wires.append(wire)
    return wires


def preview_item(source: dict[str, Any], write_doc: dict[str, Any], crop_id: str) -> dict[str, Any]:
    """The item wire as it would be after the write: the stored ``source``
    with ``write_doc`` laid over it (an OpenSearch partial update: keys the
    write does not carry keep their stored value), serialized by the one
    item serializer."""
    return serialize_item({**source, **write_doc}, crop_id)


__all__ = [
    'LegCandidate',
    'LegSelection',
    'candidate_wires',
    'preview_item',
    'select_leg_candidates',
]
