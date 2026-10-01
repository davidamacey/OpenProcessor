"""Curation router sub-module — multi-box region-box list edit routes.

``PUT /crops/{id}/regions`` (+ the batch form), ``PATCH
/crops/{id}/regions/{box_id}`` and ``POST /regions/batch_box_state``.
These are the only human routes that write box geometry or per-box
state; the box transitions live in
:mod:`src.services.curation.region_box_edits` and the item-level update
document in :func:`src.services.curation.region_writes.human_box_write`.

Split into its own module (not appended to ``regions_edit.py``) to stay
under the repo's 700-LOC-per-module ceiling.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.config import get_region_fields
from src.config.curation import get_curation_config
from src.config.region_state import RegionStatus
from src.routers.curation._common import OpenSearchDep, RegionProfileDep, _now_iso, router
from src.routers.curation.regions_edit import _batch_write, _Recorder, _write_error
from src.services.curation.region_box_edits import (
    apply_put_boxes,
    boxes_with_status,
    human_text_fields,
    validate_box_state,
    with_state,
)
from src.services.curation.region_boxes import RegionBoxWriteError, read_boxes
from src.services.curation.region_rows import as_row
from src.services.curation.region_writes import (
    human_box_write,
    parent_to_source_bbox,
    post_write_item,
)


class BoxWriteElement(BaseModel):
    """One element of a ``PUT /crops/{crop_id}/regions`` boxes list.

    ``box_id: None`` (or omitted) is a new box; a stored box referenced by
    ``box_id`` alone (no other keys) is left untouched (sibling-preserving,
    any_domain_plan.md §7.7)."""

    model_config = {'extra': 'forbid'}

    box_id: str | None = None
    bbox_norm: list[float] | None = None
    state: str | None = None
    text: str | None = None


class ItemRegionsRequest(BaseModel):
    """Body for ``PUT /crops/{crop_id}/regions``."""

    model_config = {'extra': 'forbid'}

    boxes: list[BoxWriteElement] = Field(default_factory=list)
    frame: Literal['source', 'parent'] = 'source'
    region_status: str | None = None
    region_label_source: str = 'human'
    expected_region_revision: int | None = None


class ItemBatchRegionsRequest(BaseModel):
    """Body for ``PUT /crops/batch_regions``."""

    model_config = {'extra': 'forbid'}

    crop_ids: list[str]
    boxes: list[BoxWriteElement] = Field(default_factory=list)
    region_status: str | None = None
    region_label_source: str = 'human'

    def model_post_init(self, __context: Any) -> None:
        for b in self.boxes:
            if b.box_id is not None:
                raise HTTPException(
                    status_code=422,
                    detail={'error': 'box_id_in_batch', 'message': 'ids are per item'},
                )


class BoxPatchRequest(BaseModel):
    """Body for ``PATCH /crops/{crop_id}/regions/{box_id}``."""

    model_config = {'extra': 'forbid'}

    state: str | None = None
    text: str | None = None
    region_label_source: str = 'human'
    expected_region_revision: int | None = None


class BatchBoxStateTarget(BaseModel):
    model_config = {'extra': 'forbid'}

    crop_id: str
    box_id: str


class BatchBoxStateRequest(BaseModel):
    """Body for ``POST /regions/batch_box_state``."""

    model_config = {'extra': 'forbid'}

    targets: list[BatchBoxStateTarget]
    state: str
    region_label_source: str = 'human'
    expected_region_revisions: dict[str, int] | None = None


class RegionConflictError(Exception):
    """A stale ``expected_region_revision`` (409 ``region_conflict``)."""

    def __init__(self, current: dict[str, Any]) -> None:
        self.current = current


def _check_text_allowed(elements: list[BoxWriteElement] | list[Any], profile: Any) -> None:
    """422 ``region_text_disabled`` when any element sets ``text`` on a
    profile that doesn't read text (W8.8; moved off ``region_meta``)."""
    if profile.reads_text:
        return
    if any(getattr(e, 'text', None) is not None for e in elements):
        raise HTTPException(status_code=422, detail={'error': 'region_text_disabled'})


def _check_box_states(route: str, elements: list[Any]) -> None:
    """W8c: 422 ``region_box_edits.validate_box_state`` per element instead
    of only serving ``BOX_STATE_ROUTES`` on ``GET .../regions/statuses``
    without ever enforcing it on write."""
    for e in elements:
        try:
            validate_box_state(route, getattr(e, 'state', None))
        except RegionBoxWriteError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


def _too_many_boxes_check(n_boxes: int) -> None:
    limit = get_curation_config().region_max_boxes_per_write
    if n_boxes > limit:
        raise HTTPException(
            status_code=422,
            detail={'error': 'too_many_boxes', 'limit': limit, 'requested': n_boxes},
        )


def _check_put_status(payload: ItemRegionsRequest | ItemBatchRegionsRequest) -> None:
    """``region_status: no_region_visible`` with boxes is contradictory (W8.8)."""
    if payload.region_status == RegionStatus.NO_REGION_VISIBLE.value and payload.boxes:
        raise HTTPException(
            status_code=422,
            detail={
                'error': 'validation_failed',
                'message': 'region_status no_region_visible cannot carry boxes',
            },
        )


def _check_revision(current: dict[str, Any], expected: int | None) -> None:
    if expected is None:
        return
    F = get_region_fields()
    current_rev = int(current.get(F.revision) or 0)
    if current_rev != expected:
        raise RegionConflictError(current)


def _region_conflict_detail(crop_id: str, current: dict[str, Any]) -> dict[str, Any]:
    F = get_region_fields()
    boxes = read_boxes(current, F)
    return {
        'error': 'region_conflict',
        'current_region_revision': int(current.get(F.revision) or 0),
        'current_box_ids': [b.box_id for b in boxes],
        'item': post_write_item(current, {}, crop_id),
    }


async def _write_one_boxes(opensearch: Any, crop_id: str, rec: _Recorder, writer_id: str) -> None:
    """Like :func:`_write_one`, but a :class:`RegionConflictError` from the
    merger becomes 409 ``region_conflict`` instead of falling through to
    the generic 404 handler."""
    try:
        await occ_update_one(
            opensearch, doc_id=crop_id, merger=rec, refresh='wait_for', writer_id=writer_id
        )
    except OCCFinalConflictError:
        raise
    except RegionConflictError as exc:
        raise HTTPException(
            status_code=409, detail=_region_conflict_detail(crop_id, exc.current)
        ) from exc
    except RegionBoxWriteError as exc:
        raise _write_error(exc) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f'crop not found: {crop_id}: {exc}') from exc


def _put_boxes_build(
    payload: ItemRegionsRequest | ItemBatchRegionsRequest,
    profile: Any,
    *,
    frame: str,
    check_revision: int | None = None,
) -> Any:
    """Merger body shared by the single and batch PUT: build the new box
    list from the request, apply the optional whole-set status, and write
    it through :func:`human_box_write` (one revision bump)."""
    now = _now_iso()
    requested = [b.model_dump() for b in payload.boxes]

    def _build(current: dict[str, Any]) -> dict[str, Any]:
        F = get_region_fields()
        _check_revision(current, check_revision)
        boxes = apply_put_boxes(
            current,
            requested,
            frame=frame,
            F=F,
            project_parent_to_source=parent_to_source_bbox,
            human_detector=profile.human_detector_name,
            human_detector_version=profile.human_detector_version,
            now=now,
        )
        if payload.region_status is not None:
            boxes = boxes_with_status(payload.region_status, boxes)
        return human_box_write(
            current,
            boxes,
            label_source=payload.region_label_source,
            now=now,
            human_name=profile.human_detector_name,
            human_version=profile.human_detector_version,
        )

    return _build


@router.put('/crops/{crop_id}/regions')
async def set_crop_regions(
    crop_id: str,
    payload: ItemRegionsRequest,
    opensearch: OpenSearchDep,
    profile: RegionProfileDep,
) -> dict[str, Any]:
    """Set the full per-item box list.

    The full list, in display order; omitting a stored box deletes it.
    An element is ``{box_id}`` alone for an untouched box, ``{box_id,
    bbox_norm}`` for a moved box (keeps state; a moved box is human
    geometry, an unmoved one a confirmation), ``{box_id, state}`` /
    ``{box_id, bbox_norm, state}`` to change state too, and ``{box_id:
    null, bbox_norm, state?}`` for a new box (default ``accepted``, W8
    pin 2). ``frame: "parent"`` projects into the source frame
    server-side (W8 pin 1). Optional ``region_status`` applies a
    whole-set status to the built list in the same write (W8 pin 3);
    an empty list is the human "no region visible".
    A stale ``expected_region_revision`` is 409 ``region_conflict``.
    Over ``region_profile.limits.max_boxes_per_write`` is 422
    ``too_many_boxes``. A ``text`` element on a text-free profile is 422
    ``region_text_disabled``. An out-of-range or degenerate ``bbox_norm``
    is 422.
    """
    _too_many_boxes_check(len(payload.boxes))
    _check_text_allowed(payload.boxes, profile)
    _check_box_states('PUT /crops/{crop_id}/regions', payload.boxes)
    _check_put_status(payload)
    build = _put_boxes_build(
        payload, profile, frame=payload.frame, check_revision=payload.expected_region_revision
    )
    rec = _Recorder(build, 'human:set_crop_regions')
    await _write_one_boxes(opensearch, crop_id, rec, 'human:set_crop_regions')
    return {'crop_id': crop_id, 'item': rec.item(crop_id)}


@router.put('/crops/batch_regions')
async def batch_set_crop_regions(
    payload: ItemBatchRegionsRequest,
    opensearch: OpenSearchDep,
    profile: RegionProfileDep,
) -> dict[str, Any]:
    """Replace each crop's box list with the same **new** boxes
    (typically ``boxes: []`` = "none visible"). Every element must
    have ``box_id: null`` (422 ``box_id_in_batch``): ids are per item. A
    ``text`` element on a text-free profile is 422 ``region_text_disabled``.
    """
    # Validate BEFORE the empty-`crop_ids` early return -- an empty batch
    # is trivially valid (0 items to write), but an invalid
    # `state`/oversized/text-disabled payload must still 422 even when it
    # would touch nothing, not silently short-circuit to 200.
    _too_many_boxes_check(len(payload.boxes))
    _check_text_allowed(payload.boxes, profile)
    _check_box_states('PUT /crops/batch_regions', payload.boxes)
    _check_put_status(payload)
    if not payload.crop_ids:
        return {'updated': 0, 'conflicts': [], 'invalid': [], 'items': []}
    build = _put_boxes_build(payload, profile, frame='source')
    return await _batch_write(opensearch, payload.crop_ids, build, 'human:batch_set_crop_regions')


@router.patch('/crops/{crop_id}/regions/{box_id}')
async def patch_crop_region_box(
    crop_id: str,
    box_id: str,
    payload: BoxPatchRequest,
    opensearch: OpenSearchDep,
    profile: RegionProfileDep,
) -> dict[str, Any]:
    """Per-box state/text patch -- the review panel's per-box
    accept/reject action. Every other box in the item's list is left
    untouched (per-box states persist independently). ``text`` on a
    text-free profile is 422 ``region_text_disabled``."""
    if payload.state is None and payload.text is None:
        raise HTTPException(status_code=400, detail='at least one of state, text is required')
    _check_text_allowed([payload], profile)
    _check_box_states('PATCH /crops/{crop_id}/regions/{box_id}', [payload])
    now = _now_iso()

    def _build(current: dict[str, Any]) -> dict[str, Any]:
        F = get_region_fields()
        _check_revision(current, payload.expected_region_revision)
        boxes = read_boxes(current, F)
        if not any(b.box_id == box_id for b in boxes):
            msg = f'unknown box_id: {box_id!r}'
            raise RegionBoxWriteError(msg)

        def _patched(b: Any) -> Any:
            if payload.state is not None:
                b = with_state(b, payload.state)
            if payload.text is not None:
                b = dataclasses.replace(b, **human_text_fields(payload.text))
            return b

        new_boxes = [_patched(b) if b.box_id == box_id else b for b in boxes]
        return human_box_write(
            current,
            new_boxes,
            label_source=payload.region_label_source,
            now=now,
            human_name=profile.human_detector_name,
            human_version=profile.human_detector_version,
        )

    rec = _Recorder(_build, 'human:patch_crop_region_box')
    await _write_one_boxes(opensearch, crop_id, rec, 'human:patch_crop_region_box')
    return {'crop_id': crop_id, 'box_id': box_id, 'item': rec.item(crop_id)}


@router.post('/regions/batch_box_state')
async def batch_set_region_box_state(
    payload: BatchBoxStateRequest,
    opensearch: OpenSearchDep,
    profile: RegionProfileDep,
) -> dict[str, Any]:
    """One state on many boxes across items (region-gallery triage).
    Flips only the named box on each targeted item -- never its
    siblings (contrast ``POST /regions/batch_status``, which flips every
    box of each item). ``items`` are rows: the post-write wire item once
    per targeted box, with that box as ``region_box_id``."""
    # Validate BEFORE the empty-`targets` early return -- see
    # `batch_set_crop_regions`'s matching comment for why.
    try:
        validate_box_state('POST /regions/batch_box_state', payload.state)
    except RegionBoxWriteError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not payload.targets:
        return {'updated': 0, 'conflicts': [], 'invalid': [], 'items': []}

    by_crop: dict[str, list[str]] = {}
    for t in payload.targets:
        by_crop.setdefault(t.crop_id, []).append(t.box_id)
    expected_revisions = payload.expected_region_revisions or {}
    now = _now_iso()

    def _build_for(crop_id: str, box_ids: list[str]) -> Any:
        def _build(current: dict[str, Any]) -> dict[str, Any]:
            F = get_region_fields()
            _check_revision(current, expected_revisions.get(crop_id))
            boxes = read_boxes(current, F)
            known_ids = {b.box_id for b in boxes}
            missing = set(box_ids) - known_ids
            if missing:
                msg = f'unknown box_id(s): {sorted(missing)!r}'
                raise RegionBoxWriteError(msg)
            new_boxes = [with_state(b, payload.state) if b.box_id in box_ids else b for b in boxes]
            return human_box_write(
                current,
                new_boxes,
                label_source=payload.region_label_source,
                now=now,
                human_name=profile.human_detector_name,
                human_version=profile.human_detector_version,
            )

        return _build

    updated = 0
    conflicts: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for crop_id, box_ids in by_crop.items():
        rec = _Recorder(_build_for(crop_id, box_ids), 'human:batch_set_region_box_state')
        try:
            await _write_one_boxes(opensearch, crop_id, rec, 'human:batch_set_region_box_state')
        except HTTPException as exc:
            if exc.status_code == 409:
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                conflicts.append({'crop_id': crop_id, **detail})
            else:
                invalid.append({'crop_id': crop_id, 'detail': exc.detail})
            continue
        updated += 1
        item = rec.item(crop_id)
        items.extend(as_row(item, box_id) for box_id in box_ids)
    return {'updated': updated, 'conflicts': conflicts, 'invalid': invalid, 'items': items}
