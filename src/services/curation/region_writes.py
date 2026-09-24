"""The update documents every human region writer applies.

``PUT /crops/{id}/region``, ``PUT /crops/batch_region``, ``PATCH
/crops/{id}/region_meta`` and ``POST /regions/batch_status`` all build
their writes here, so the region state invariants hold whichever writer a
client picks:

- a status whose lifecycle entry ``clears_box`` also clears the box and
  score (``REGION_STATUS_INFO`` in ``src/config/region_state.py``);
- ``region_verified`` follows the status: true for the confirm status,
  false for every other human-written status. Clients never send it. A
  write that re-asserts the stored status re-derives nothing (verified and
  the region-cluster placement stay as stored);
- the confirm status needs a box to confirm (:class:`RegionWriteError`);
- a false-positive mark parks the region in the permanent FP cluster,
  and moving off it releases the region for re-clustering.

Each writer returns the post-write item built by
:func:`post_write_item`, so a client adopts the server's state instead
of re-deriving it.
"""

from __future__ import annotations

from typing import Any

from src.config import get_region_fields
from src.config.region_state import CONFIRM_STATUS, REGION_STATUS_INFO, RegionStatus
from src.services.curation.wire import serialize_item
from src.services.detection.cascade_detect import crop_norm_to_source_norm, region_provenance
from src.services.detection.profile_registry import region_profile_or_neutral


class RegionWriteError(ValueError):
    """A requested human region write would leave the region inconsistent."""


def validate_bbox_norm(bbox: tuple[float, float, float, float] | list[float]) -> None:
    """Raise :class:`RegionWriteError` for an out-of-range or degenerate box."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    for name, v in (('x1', x1), ('y1', y1), ('x2', x2), ('y2', y2)):
        if not 0.0 <= v <= 1.0:
            raise RegionWriteError(f'region bbox {name}={v} out of [0, 1] range')
    if x2 <= x1 or y2 <= y1:
        raise RegionWriteError(f'region bbox is degenerate: ({x1}, {y1}, {x2}, {y2})')


def parent_to_source_bbox(
    region_in_parent: tuple[float, float, float, float] | list[float],
    parent_bbox_norm: Any,
) -> list[float]:
    """Project a box drawn in the item-crop frame into the source frame.

    ``parent_bbox_norm`` is the item's own stored ``bbox_norm`` (source
    frame). Raises :class:`RegionWriteError` when the item has no usable
    box to project through.
    """
    validate_bbox_norm(region_in_parent)
    if not isinstance(parent_bbox_norm, list | tuple) or len(parent_bbox_norm) != 4:
        raise RegionWriteError('item has no bbox_norm to project a parent-frame region through')
    try:
        px1, py1, px2, py2 = (float(v) for v in parent_bbox_norm)
    except (TypeError, ValueError) as exc:
        raise RegionWriteError('item bbox_norm is not numeric') from exc
    if px2 <= px1 or py2 <= py1:
        raise RegionWriteError('item bbox_norm is degenerate')
    x1, y1, x2, y2 = (float(v) for v in region_in_parent)
    return list(crop_norm_to_source_norm((x1, y1, x2, y2), (px1, py1, px2, py2)))


def fp_cluster_fields(region_status: str | None) -> dict[str, Any]:
    """Region-cluster side effects of a human status write."""
    from src.services.curation.clustering.orchestrator import FALSE_POSITIVE_REGION_CLUSTER_ID

    F = get_region_fields()
    if region_status == RegionStatus.FALSE_POSITIVE:
        return {
            F.cluster_id: FALSE_POSITIVE_REGION_CLUSTER_ID,
            F.cluster_subid: None,
            F.cluster_distance: 0.0,
        }
    return {F.cluster_id: None, F.cluster_subid: None}


def human_status_fields(region_status: str, current: dict[str, Any]) -> dict[str, Any]:
    """Fields a human write of ``region_status`` sets, given the stored doc.

    Raises :class:`RegionWriteError` when confirming a region that has no box.
    """
    F = get_region_fields()
    status = RegionStatus(region_status)
    info = REGION_STATUS_INFO[status]
    if status == CONFIRM_STATUS and not current.get(F.bbox_norm):
        raise RegionWriteError(
            f'cannot mark {status.value!r} without a region box; set one with PUT region'
        )
    doc: dict[str, Any] = {F.status: status.value}
    if current.get(F.status) == status.value:
        # Re-asserting the stored status (a bulk write over a mixed
        # selection) changes nothing derived from it: verified and the
        # region-cluster placement stay as stored. Confirming is the one
        # exception — it is an explicit verification.
        if status == CONFIRM_STATUS:
            doc[F.verified] = True
    else:
        doc[F.verified] = status == CONFIRM_STATUS
        doc.update(fp_cluster_fields(status.value))
    if info.clears_box:
        doc[F.bbox_norm] = None
        doc[F.score] = None
    return doc


def region_box_doc(
    region_bbox_norm: list[float] | None, *, label_source: str, now: str
) -> dict[str, Any]:
    """Update doc for a human box write (``None`` = "no region visible").

    ``region_bbox_norm`` is already in the source frame and validated.
    """
    F = get_region_fields()
    human = region_profile_or_neutral()
    if region_bbox_norm is None:
        from src.config.region_state import REJECT_STATUS

        doc = human_status_fields(REJECT_STATUS.value, {})
        doc.update(
            {
                F.label_source: label_source,
                F.detector: human.human_detector_name,
                F.detector_version: human.human_detector_version,
                F.verifier: human.human_detector_name,
                F.verifier_version: human.human_detector_version,
                F.verified_at: now,
                F.detected_at: now,
                F.bbox_frame: 'source',
                F.validated: True,
                'updated_at': now,
            }
        )
        return doc
    return {
        F.bbox_norm: list(region_bbox_norm),
        F.score: 1.0,  # a human-drawn box is ground truth
        F.status: CONFIRM_STATUS.value,
        F.label_source: label_source,
        F.verified: True,
        F.validated: True,
        **region_provenance(
            detector=human.human_detector_name,
            detector_version=human.human_detector_version,
            bbox_frame='source',
            verifier=human.human_detector_name,
            verifier_version=human.human_detector_version,
            detected_at=now,
            verified_at=now,
        ),
        'updated_at': now,
    }


def post_write_item(
    current: dict[str, Any], update: dict[str, Any], crop_id: str
) -> dict[str, Any]:
    """The wire item as stored after ``update`` is merged onto ``current``."""
    return serialize_item({**current, **update}, crop_id)


__all__ = [
    'RegionWriteError',
    'fp_cluster_fields',
    'human_status_fields',
    'parent_to_source_bbox',
    'post_write_item',
    'region_box_doc',
    'validate_bbox_norm',
]
