"""Request models for the crop label, move, exclude, discard, status and region-meta writes."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from src.config.region_state import HUMAN_WRITABLE_STATUSES

# Runtime import: pydantic resolves the Literal annotation from module globals.
from src.routers.curation._selection import SelectionTargets
from src.services.curation.class_sources import HumanLabelSource


class CropLabelRequest(BaseModel):
    class_id: int
    # A human source only; the server writes class_source='human' itself.
    label_source: HumanLabelSource = 'human'


class CropBatchLabelRequest(SelectionTargets):
    class_id: int
    label_source: HumanLabelSource = 'human'


class CropMoveRequest(SelectionTargets):
    cluster_uid: uuid.UUID


class CropExcludeRequest(SelectionTargets):
    """Exclude crops from training + clustering (reversible).

    Blurry / unidentifiable / partial crops the human doesn't want in
    the training set. ``reason`` defaults to ``'ignore'``; the UI can
    pass a more specific tag (``'blurry'``, ``'unidentifiable'``,
    ``'wrong_class'``, ``'partial_crop'``) when the operator wants to
    record why (e.g. a whole cluster of blurry items).
    """

    reason: str = 'ignore'


class CropUnexcludeRequest(SelectionTargets):
    """Reverse an exclusion (the labeler's Undo path for Ignore)."""


class CropUndoBatchRequest(BaseModel):
    """Undo the most recent human class write on each crop."""

    crop_ids: list[str] = Field(..., max_length=5000)


class CropDiscardRequest(BaseModel):
    """Discard an item: clear its class (it doesn't belong where it is),
    dismiss it from every review queue, or both. Recorded like a label
    write, so ``POST /crops/{id}/label/undo`` reverses it."""

    model_config = {'extra': 'forbid'}

    clear_class: bool = True
    dismiss_from_review: bool = False


class CropDiscardBatchRequest(CropDiscardRequest):
    crop_ids: list[str] = Field(..., max_length=5000)


# Region status values an operator may write — the lifecycle entries marked
# human_writable in src/config/region_state.py (stdlib-only, so the TS
# contract codegen exports the same set without importing the app).
HUMAN_REGION_STATUS_VALUES = HUMAN_WRITABLE_STATUSES


class CropBatchStatusRequest(BaseModel):
    """Bulk-set ``region_status`` over many items (the cluster-view triage op).

    Lets an operator select an outlier sub-cluster and mark every region
    ``false_positive`` / ``no_region_visible`` in one call, or bulk-confirm
    good regions (``region_status='detected'``).
    ``region_status`` must be one of ``HUMAN_REGION_STATUS_VALUES``.
    ``region_verified`` is accepted but ignored: the server derives it
    from ``region_status``.
    """

    model_config = {'extra': 'forbid'}

    crop_ids: list[str] = Field(..., max_length=5000)
    region_status: str
    region_verified: bool | None = Field(
        default=None, deprecated=True, description='Ignored; derived from region_status.'
    )
    region_label_source: str = 'human'


class ItemRegionMetaRequest(BaseModel):
    """Patch region metadata without touching any box geometry.

    Use this for operator corrections like changing the status to
    ``verify_rejected``. To edit a box's text, use ``PATCH
    /crops/{crop_id}/regions/{box_id}`` (per-box) -- ``region_text``
    is REMOVED from this request (D decision, 2026-09-26): it was a
    per-box value on an item-level route. A stale client sending it gets
    422 (``extra='forbid'``). To set or clear a box's geometry, use ``PUT
    /crops/{crop_id}/regions``.

    Every field is optional; only the provided ones are written. ``None``
    on ``region_rejection_reason`` clears the reason. ``region_status``
    must be one of ``HUMAN_REGION_STATUS_VALUES`` when present.
    """

    # extra='forbid' so a stale key (incl. the removed region_text) 422s
    # instead of silently no-opping.
    model_config = {'extra': 'forbid'}

    region_status: str | None = None
    region_rejection_reason: str | None = None
    region_label_source: str = 'human'


class CropFlagNewClassRequest(BaseModel):
    """Marks crops as needing a class that doesn't exist in the registry
    yet — for batch curator review (typically weekly)."""

    crop_ids: list[str] = Field(..., max_length=5000)
    note: str = ''
