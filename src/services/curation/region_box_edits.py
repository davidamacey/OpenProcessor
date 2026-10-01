"""Human edits over the per-item region-box list (W8.7 / W8.8).

Pure (no I/O). Every human writer -- ``PUT /crops/{id}/regions`` and its
batch form, ``PATCH /crops/{id}/regions/{box_id}``, ``POST
/regions/batch_box_state``, ``PATCH region_meta`` and ``POST
/regions/batch_status`` -- builds its new box list through the functions
here, so each invariant has exactly one implementation:

- a box's state transition (:func:`with_state`): the rejection reason and
  the false-positive cluster placement follow the state, whichever writer
  made the transition;
- a human-moved box is human geometry (:func:`human_geometry`);
- a human-typed text carries the human stamps (:func:`human_text_fields`);
- a whole-set status over the list (:func:`boxes_with_status`);
- bbox validation (:func:`validate_bbox_norm`) and same-box detection
  (:func:`same_box`).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from src.config.region_fields import RegionFields, get_region_fields
from src.config.region_rejection import (
    REJECT_REASON_HUMAN,
    REJECT_REASON_NO_VERDICT,
    REJECT_REASON_VERIFIER,
)
from src.config.region_state import BOX_STATE_ROUTES, RegionStatus
from src.services.curation.cluster_ids import FALSE_POSITIVE_REGION_CLUSTER_ID
from src.services.curation.region_boxes import (
    RegionBox,
    RegionBoxWriteError,
    next_box_id,
    read_boxes,
)
from src.services.detection.region_text import TEXT_CHOICE_HUMAN


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_FALSE_POSITIVE = RegionStatus.FALSE_POSITIVE.value

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


def validate_bbox_norm(bbox: Sequence[float]) -> None:
    """Raise :class:`RegionBoxWriteError` for an out-of-range or degenerate box."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    for name, v in (('x1', x1), ('y1', y1), ('x2', x2), ('y2', y2)):
        if not 0.0 <= v <= 1.0:
            msg = f'region bbox {name}={v} out of [0, 1] range'
            raise RegionBoxWriteError(msg)
    if x2 <= x1 or y2 <= y1:
        msg = f'region bbox is degenerate: ({x1}, {y1}, {x2}, {y2})'
        raise RegionBoxWriteError(msg)


_BOX_STATE_ROUTES_BY_NAME = {r.route: r.states for r in BOX_STATE_ROUTES}


def validate_box_state(route: str, state: str | None) -> None:
    """W8c: enforce ``BOX_STATE_ROUTES`` (``src/config/region_state.py``) on
    write, not just serve it on ``GET .../regions/statuses``.

    ``route`` is the exact ``BoxStateRoute.route`` string (e.g. ``'PATCH
    /crops/{crop_id}/regions/{box_id}'``); a route this table doesn't know
    about is a programming error (``ValueError``), never a client-facing
    422. ``state=None`` (untouched / no state in this write) is always
    fine -- the caller may not be setting a state at all.
    """
    if state is None:
        return
    try:
        allowed = _BOX_STATE_ROUTES_BY_NAME[route]
    except KeyError as exc:
        msg = f'no BOX_STATE_ROUTES entry for route {route!r}'
        raise ValueError(msg) from exc
    if state not in allowed:
        msg = f'state must be one of {sorted(allowed)} for {route}; got {state!r}'
        raise RegionBoxWriteError(msg)


def with_state(box: RegionBox, state: str, *, rejection_reason: str | None = None) -> RegionBox:
    """The one human state transition of a single box.

    - ``rejected``: carries ``rejection_reason`` (default
      :data:`REJECT_REASON_HUMAN`, which also marks the box human-owned,
      :func:`~src.services.curation.region_boxes.is_human_owned`). A
      re-reject re-stamps the reason: a human reject over a verifier
      reject is a human decision now.
    - every other state clears the reason, so a box never carries a stale
      verdict from an earlier state (W8-cleanup m5).
    - ``false_positive``: parked in the permanent FP cluster; leaving it
      releases the box for re-clustering. Re-asserting the box's current
      non-rejected state changes nothing (cluster placement stays).
    """
    if state == box.state and state != 'rejected':
        return box
    changes: dict[str, Any] = {
        'state': state,
        'rejection_reason': (rejection_reason or REJECT_REASON_HUMAN)
        if state == 'rejected'
        else None,
    }
    if state == _FALSE_POSITIVE:
        changes.update(
            cluster_id=FALSE_POSITIVE_REGION_CLUSTER_ID, cluster_subid=None, cluster_distance=0.0
        )
    elif box.state == _FALSE_POSITIVE:
        changes.update(cluster_id=None, cluster_subid=None, cluster_distance=None)
    return dataclasses.replace(box, **changes)


def human_geometry(
    box: RegionBox, bbox: Sequence[float], *, detector: str, detector_version: str, now: str | None
) -> RegionBox:
    """``box`` moved to ``bbox`` by a human.

    Within :data:`BOX_MATCH_TOLERANCE` of the stored geometry this is a
    confirmation, not an edit: the stored box (its exact coordinates and
    provenance) is kept. A different box is human geometry: the human is
    the detector (found ``now``), the score is ``1.0``, and the verdict
    keys that judged the old geometry are cleared. State, rejection
    reason and text stay.
    """
    if same_box(bbox, box.bbox_norm):
        return box
    return dataclasses.replace(
        box,
        bbox_norm=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
        detector=detector,
        detector_version=detector_version,
        score=1.0,
        source='human',
        bbox_correct=None,
        confidence=None,
        detected_at=now,
    )


def human_text_fields(text: str) -> dict[str, Any]:
    """The text attributes a human-typed ``text`` sets on a box: the human
    reading is ground truth, so the text readers know not to overwrite it
    (``text_source == 'human'`` also marks the box human-owned)."""
    return {
        'text': text,
        'text_source': 'human',
        'text_confidence': 1.0 if text else None,
        'text_choice': TEXT_CHOICE_HUMAN,
    }


def apply_put_boxes(
    current: dict[str, Any],
    requested: Sequence[dict[str, Any]],
    *,
    frame: str,
    F: RegionFields | None = None,
    project_parent_to_source: Callable[[Sequence[float], Any], Sequence[float]] | None = None,
    human_detector: str = 'human',
    human_detector_version: str = '1',
    now: str | None = None,
) -> list[RegionBox]:
    """Sibling-preserving merge for ``PUT /crops/{crop_id}/regions``.

    ``requested`` is the full list, in display order (any_domain_plan.md
    W8.8): an element with only ``box_id`` keeps its stored box untouched;
    one with ``box_id`` plus other keys patches just those keys onto the
    stored box (:func:`human_geometry`, :func:`with_state`,
    :func:`human_text_fields`); ``box_id: None`` (or omitted) is a new
    human box, assigned the next id (``bbox_norm`` required) and
    defaulting to ``accepted`` when ``state`` is omitted (W8 pin 2).
    Omitting a stored box from ``requested`` deletes it. A duplicate
    ``box_id`` is an error.

    ``now`` stamps ``detected_at`` on the boxes a human draws or moves.
    ``project_parent_to_source(bbox, item_bbox_norm)`` is supplied by the
    caller for ``frame == 'parent'`` (W8 pin 1; the item crop's own
    ``bbox_norm`` is the frame to project through); this module stays pure
    and does no frame geometry itself.
    """
    F = F or get_region_fields()
    existing = {b.box_id: b for b in read_boxes(current, F)}
    seq = int(current.get(F.box_seq) or 0)
    result: list[RegionBox] = []
    seen: set[str] = set()

    for element in requested:
        box_id = element.get('box_id')
        bbox = element.get('bbox_norm')
        if bbox is not None:
            validate_bbox_norm(bbox)
            if frame == 'parent':
                if project_parent_to_source is None:
                    msg = "frame='parent' requires project_parent_to_source"
                    raise RegionBoxWriteError(msg)
                bbox = project_parent_to_source(bbox, current.get('bbox_norm'))
        if box_id is None:
            if bbox is None:
                msg = 'bbox_required: a new box needs a bbox_norm'
                raise RegionBoxWriteError(msg)
            new_id = next_box_id([*existing.values(), *result], seq=seq)
            box = RegionBox(
                box_id=new_id,
                bbox_norm=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
                state='accepted',
                score=1.0,
                detector=human_detector,
                detector_version=human_detector_version,
                source='human',
                detected_at=now,
            )
        else:
            if box_id in seen:
                msg = f'duplicate_box_id: {box_id!r}'
                raise RegionBoxWriteError(msg)
            seen.add(box_id)
            stored = existing.get(box_id)
            if stored is None:
                msg = f'unknown box_id: {box_id!r}'
                raise RegionBoxWriteError(msg)
            box = stored
            if bbox is not None:
                box = human_geometry(
                    box,
                    bbox,
                    detector=human_detector,
                    detector_version=human_detector_version,
                    now=now,
                )
        if element.get('state') is not None:
            box = with_state(box, element['state'])
        if element.get('text') is not None:
            box = dataclasses.replace(box, **human_text_fields(element['text']))
        result.append(box)

    return result


def _is_verifier_rejected(box: RegionBox) -> bool:
    return box.state == 'rejected' and box.rejection_reason in (
        REJECT_REASON_VERIFIER,
        REJECT_REASON_NO_VERDICT,
    )


def boxes_with_status(
    status: str, boxes: Sequence[RegionBox], *, rejection_reason: str | None = None
) -> list[RegionBox]:
    """Whole-set human status transition over the list (W8.7 table).

    A whole-set confirm never overrides a per-box decision that already
    settled a box; it only settles the undecided ones. ``rejection_reason``
    (a reviewer's note) replaces the default human reason on the boxes a
    ``verify_rejected`` write rejects.
    """
    if status == RegionStatus.DETECTED.value:
        if not boxes:
            msg = 'no_boxes'
            raise RegionBoxWriteError(msg)
        result = [with_state(b, 'accepted') if b.state == 'proposed' else b for b in boxes]
        if not any(b.state == 'accepted' for b in result):
            # Nothing was proposed or accepted: this confirm is a human
            # reversing the verifier's rejection of every box. Reopen only
            # the boxes the VERIFIER rejected (W8-cleanup M3) -- a human's
            # own per-box reject or a sanity-gate reject is never silently
            # overridden. A whole-set HUMAN reject followed by a confirm
            # therefore 422s: a per-box and a whole-set human reject share
            # the same reason and can't be told apart, and `POST
            # .../region/undo` is the documented way back.
            result = [with_state(b, 'accepted') if _is_verifier_rejected(b) else b for b in boxes]
        if not any(b.state == 'accepted' for b in result):
            msg = 'no_accepted_box'
            raise RegionBoxWriteError(msg)
        return result
    if status == RegionStatus.FALSE_POSITIVE.value:
        return [with_state(b, _FALSE_POSITIVE) for b in boxes]
    if status == RegionStatus.VERIFY_REJECTED.value:
        return [with_state(b, 'rejected', rejection_reason=rejection_reason) for b in boxes]
    if status == RegionStatus.NO_REGION_VISIBLE.value:
        return []
    msg = f'unsupported whole-set status: {status!r}'
    raise RegionBoxWriteError(msg)


__all__ = [
    'BOX_MATCH_TOLERANCE',
    'apply_put_boxes',
    'boxes_with_status',
    'human_geometry',
    'human_text_fields',
    'same_box',
    'validate_bbox_norm',
    'validate_box_state',
    'with_state',
]
