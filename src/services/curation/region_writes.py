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
- a box PUT that equals the stored box is a confirmation: detector,
  version, score and detection time are kept (:func:`region_box_write`);
- a false-positive mark parks the region in the permanent FP cluster,
  and moving off it releases the region for re-clustering.

Each writer returns the post-write item built by
:func:`post_write_item`, so a client adopts the server's state instead
of re-deriving it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from src.config import get_region_fields
from src.config.region_rejection import REJECT_REASON_NO_VERDICT, REJECT_REASON_VERIFIER
from src.config.region_state import CONFIRM_STATUS, REGION_STATUS_INFO, RegionStatus
from src.services.curation.region_boxes import (
    boxes_with_status,
    boxes_write_fields,
    derive_status,
    read_boxes,
)
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


def candidate_box(current: dict[str, Any]) -> list[float] | None:
    """The verifier-rejected candidate box stored on ``current``, or ``None``."""
    F = get_region_fields()
    box = current.get(F.candidate_bbox_norm)
    if not isinstance(box, list | tuple) or len(box) != 4:
        return None
    try:
        return [float(v) for v in box]
    except (TypeError, ValueError):
        return None


def candidate_promotion(current: dict[str, Any]) -> dict[str, Any]:
    """Fields that turn ``current``'s rejected candidate into its region box.

    The candidate's detector, version, score and source become the
    region's provenance (the detector *found* the box; a human accepting it
    doesn't change that) and the candidate + rejection fields are cleared.
    Empty when there is no candidate or the item already has a box.
    """
    F = get_region_fields()
    box = candidate_box(current)
    if box is None or current.get(F.bbox_norm):
        return {}
    return {
        F.bbox_norm: box,
        F.bbox_frame: 'source',
        F.score: current.get(F.candidate_score),
        F.detector: current.get(F.candidate_detector),
        F.detector_version: current.get(F.candidate_detector_version),
        F.source: current.get(F.candidate_source),
        F.rejection_reason: None,
        **candidate_clear_fields(),
    }


def candidate_clear_fields() -> dict[str, Any]:
    """Update-doc entries clearing every rejected-candidate field."""
    F = get_region_fields()
    return dict.fromkeys(
        (
            F.candidate_bbox_norm,
            F.candidate_score,
            F.candidate_detector,
            F.candidate_detector_version,
            F.candidate_source,
        )
    )


# A human accepting a rejected candidate as a region (confirm) or keeping it
# as a known bad detection (false positive) promotes it into the region box.
_PROMOTING_STATUSES = frozenset({CONFIRM_STATUS, RegionStatus.FALSE_POSITIVE})


def human_status_fields(region_status: str, current: dict[str, Any]) -> dict[str, Any]:
    """Fields a human write of ``region_status`` sets, given the stored doc.

    Confirming (or marking a false positive on) an item whose only box is a
    verifier-rejected candidate promotes that candidate to the region box
    (:func:`candidate_promotion`). Raises :class:`RegionWriteError` when
    confirming a region that has no box and no candidate.
    """
    F = get_region_fields()
    status = RegionStatus(region_status)
    info = REGION_STATUS_INFO[status]
    promotion = candidate_promotion(current) if status in _PROMOTING_STATUSES else {}
    if status == CONFIRM_STATUS and not (current.get(F.bbox_norm) or promotion):
        raise RegionWriteError(
            f'cannot mark {status.value!r} without a region box; set one with PUT region'
        )
    doc: dict[str, Any] = {F.status: status.value, **promotion}
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


def human_status_box_write(
    region_status: str, current: dict[str, Any], *, rejection_reason: str | None = None
) -> dict[str, Any]:
    """W8-cleanup: the ``region_boxes``-based whole-set status write backing
    ``PATCH /crops/{id}/region_meta`` and ``POST /regions/batch_status``.

    Replaces :func:`human_status_fields` (which built the pre-W8 single
    ``region_bbox_norm``/``region_score`` doc) for these two routes only --
    ``PUT /crops/{id}/region`` and its batch form still build on the old
    single-box shape via :func:`region_box_write` until they're removed
    (see the W8-cleanup plan's Item 2/3).

    Delegates the actual box-list transition to
    :func:`~src.services.curation.region_boxes.boxes_with_status` (raises
    :class:`~src.services.curation.region_boxes.RegionBoxWriteError` for
    the same invariant violations ``human_status_fields`` used to raise
    :class:`RegionWriteError` for -- confirming with no box to confirm).
    ``rejection_reason`` overrides the default
    (:data:`~src.config.region_rejection.REJECT_REASON_HUMAN`) on every
    box that transition just rejected -- the closest per-box equivalent of
    the old item-level ``region_rejection_reason`` PATCH field.

    Pre-W8, confirming (or marking false-positive) a verifier-rejected
    candidate promoted it via :func:`candidate_promotion` --
    :func:`~src.services.curation.region_boxes.boxes_with_status` never
    revives an already-``rejected`` box on its own (by design: a
    whole-set confirm must not override a per-box decision that already
    settled a box). A human CONFIRM is the one deliberate exception to
    that rule -- it is explicitly reversing the earlier rejection -- so a
    box list with nothing ``proposed``/``accepted`` to confirm has its
    ``rejected`` box(es) reopened to ``proposed`` (reason cleared) right
    before the transition. ``false_positive`` needs no such step:
    :func:`~src.services.curation.region_boxes.boxes_with_status` already
    force-sets every box (including a rejected one) to
    ``false_positive``.

    Only a box the *verifier* rejected (:data:`REJECT_REASON_VERIFIER` /
    :data:`REJECT_REASON_NO_VERDICT`) is reopened by a whole-set CONFIRM
    (W8-cleanup M3) -- a human's own earlier per-box rejection or a
    sanity-gate reject must never be silently overridden by a later
    whole-set confirm. When nothing is reopenable,
    :func:`~src.services.curation.region_boxes.boxes_with_status` raises
    ``no_accepted_box`` (422), matching pre-W8 behavior for e.g. a
    ``detection_failed`` item whose only box is sanity-rejected.

    The legacy per-item mirror fields
    (``bbox_norm``/``score``/``detector``/``detector_version``/``source``/
    ``bbox_frame``/``rejection_reason``, plus clearing the retired
    ``candidate_*`` fields) that :func:`~src.services.curation.wire.
    region_to_wire` still serves additively are maintained by
    :func:`~src.services.curation.region_boxes.boxes_write_fields` itself
    now (W8-cleanup M2) -- every box writer refreshes them the same way:
    ``bbox_norm``/``score``/``detector``/``detector_version``/``source``
    mirror the highest-scoring accepted-or-false_positive box, cleared to
    ``None`` when there is none (never a rejected box's coordinates --
    ``bbox_norm`` is an accepted region to every reader). ``rejection_reason``
    mirrors the highest-scoring *rejected* box independently, since
    showing a reason never makes a box look accepted.
    """
    F = get_region_fields()
    status = RegionStatus(region_status)
    boxes = read_boxes(current, F)
    if status == CONFIRM_STATUS and not any(b.state in ('proposed', 'accepted') for b in boxes):
        reopenable_reasons = (REJECT_REASON_VERIFIER, REJECT_REASON_NO_VERDICT)
        boxes = [
            dataclasses.replace(b, state='proposed', rejection_reason=None)
            if b.state == 'rejected' and b.rejection_reason in reopenable_reasons
            else b
            for b in boxes
        ]
    new_boxes = boxes_with_status(status.value, boxes)
    if rejection_reason is not None and status == RegionStatus.VERIFY_REJECTED:
        new_boxes = [
            dataclasses.replace(b, rejection_reason=rejection_reason)
            if b.state == 'rejected'
            else b
            for b in new_boxes
        ]
    doc: dict[str, Any] = dict(boxes_write_fields(new_boxes, current_src=current))
    # `empty_status=status`: only NO_REGION_VISIBLE ever leaves `new_boxes`
    # empty (boxes_with_status returns `[]` for it) -- every other status
    # is reflected by derive_status's own precedence over the now-uniform
    # box list, so this only matters for that one case.
    doc[F.status] = derive_status(new_boxes, empty_status=status).value

    if not new_boxes and REGION_STATUS_INFO[status].wants_reason and rejection_reason is not None:
        # M1(a): a box-less status (only NO_REGION_VISIBLE) has no box left
        # to carry the reason -- boxes_write_fields's mirror derivation
        # cleared it to None above; store it on the item directly, the one
        # case the mirror can't cover.
        doc[F.rejection_reason] = rejection_reason

    if current.get(F.status) == status.value:
        # Re-asserting the stored status (a bulk write over a mixed
        # selection) changes nothing derived from it beyond the box
        # states above: verified and the region-cluster placement stay as
        # stored. Confirming is the one exception -- it is an explicit
        # verification.
        if status == CONFIRM_STATUS:
            doc[F.verified] = True
    else:
        doc[F.verified] = status == CONFIRM_STATUS
        doc.update(fp_cluster_fields(status.value))
    return doc


def reason_only_box_write(current: dict[str, Any], reason: str | None) -> dict[str, Any]:
    """W8-cleanup N3: a reason-only PATCH (no status change) over the
    current box list.

    Patches every currently-rejected box's ``rejection_reason``, same as
    ``human_status_box_write``'s per-status reason update. For a box-less
    item (``no_region_visible``), there is no rejected box for
    :func:`~src.services.curation.region_boxes.boxes_write_fields` to
    derive a mirror reason from, so it always clears the item-level
    ``rejection_reason`` to ``None`` -- restore the reason this request
    is setting in that case, matching M1(a)'s box-less mirror.
    """
    F = get_region_fields()
    boxes = read_boxes(current, F)
    new_boxes = [
        dataclasses.replace(b, rejection_reason=reason) if b.state == 'rejected' else b
        for b in boxes
    ]
    doc = dict(boxes_write_fields(new_boxes, current_src=current))
    if not any(b.state == 'rejected' for b in new_boxes):
        doc[F.rejection_reason] = reason
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


BOX_MATCH_TOLERANCE = 1e-4
"""Max per-coordinate difference (normalized source frame) for a PUT box
to count as the stored box: well under a pixel at any practical image
size, well over the float noise of a frame projection round trip."""


def same_box(a: Any, b: Any, *, tol: float = BOX_MATCH_TOLERANCE) -> bool:
    """True when ``a`` and ``b`` are both 4-number boxes within ``tol``."""
    if not isinstance(a, list | tuple) or not isinstance(b, list | tuple):
        return False
    if len(a) != 4 or len(b) != 4:
        return False
    try:
        return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b, strict=True))
    except (TypeError, ValueError):
        return False


def region_confirm_doc(current: dict[str, Any], *, label_source: str, now: str) -> dict[str, Any]:
    """Update doc for a human confirming the stored box unchanged.

    Records the confirmation only — status, verified, validated and the
    human verifier. The detector, its version, score and detection time
    describe who *found* the box, which a confirmation doesn't change.
    """
    F = get_region_fields()
    human = region_profile_or_neutral()
    doc = human_status_fields(CONFIRM_STATUS.value, current)
    doc.update(
        {
            # The stored box, not the request's noisy copy of it.
            F.bbox_norm: list(current[F.bbox_norm]),
            F.label_source: label_source,
            F.validated: True,
            F.verifier: human.human_detector_name,
            F.verifier_version: human.human_detector_version,
            F.verified_at: now,
            'updated_at': now,
        }
    )
    return doc


def region_box_write(
    current: dict[str, Any],
    region_bbox_norm: list[float] | None,
    *,
    label_source: str,
    now: str,
) -> dict[str, Any]:
    """Update doc for ``PUT region``: a confirmation when the box equals the
    stored one or the stored rejected candidate (:func:`same_box`), else a
    human box write (:func:`region_box_doc`)."""
    F = get_region_fields()
    if region_bbox_norm is not None and same_box(region_bbox_norm, current.get(F.bbox_norm)):
        return region_confirm_doc(current, label_source=label_source, now=now)
    promotion = candidate_promotion(current)
    if (
        region_bbox_norm is not None
        and promotion
        and same_box(region_bbox_norm, promotion[F.bbox_norm])
    ):
        # The box a human PUTs back is the rejected candidate: a confirmation
        # of the detector's box, provenance kept.
        promoted = {**current, **promotion}
        return {**promotion, **region_confirm_doc(promoted, label_source=label_source, now=now)}
    doc = region_box_doc(region_bbox_norm, label_source=label_source, now=now)
    if region_bbox_norm is not None:
        # A human-drawn box replaces the rejected candidate.
        doc.update({F.rejection_reason: None, **candidate_clear_fields()})
    return doc


def post_write_item(
    current: dict[str, Any], update: dict[str, Any], crop_id: str
) -> dict[str, Any]:
    """The wire item as stored after ``update`` is merged onto ``current``."""
    return serialize_item({**current, **update}, crop_id)


__all__ = [
    'BOX_MATCH_TOLERANCE',
    'RegionWriteError',
    'candidate_box',
    'candidate_clear_fields',
    'candidate_promotion',
    'fp_cluster_fields',
    'human_status_box_write',
    'human_status_fields',
    'parent_to_source_bbox',
    'post_write_item',
    'region_box_doc',
    'region_box_write',
    'region_confirm_doc',
    'same_box',
    'validate_bbox_norm',
]
