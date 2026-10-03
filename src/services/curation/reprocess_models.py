"""Request/response shapes of the unified reprocess (W10.13).

Pydantic, in the service layer: the same models are the HTTP bodies, the
persisted job request and the ``suggested_reprocess`` a profile-activation
impact serves (``region_impact``), so a client can POST that value to
``/reprocess`` verbatim. Every request model is ``extra='forbid'``.

The *shape* rules that pydantic cannot express as one structured error
(exactly one target form, a non-empty filter) live in
:func:`~src.services.curation.reprocess.validate_targets`, so the route
reports them as ``reprocess_targets_invalid`` like every other 4xx.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ReprocessScope = Literal['detect', 'open_vocab', 'region', 'vlm', 'embed']
RegionMode = Literal['redetect', 'reverify']

MAX_TARGET_IDS = 5000


class ReprocessFilter(BaseModel):
    """The W4 requeue selection, moved here, plus provenance selectors.

    ``profile_not`` / ``profile_revision_below`` select items not produced
    by the profile@revision named (both given: not produced by that exact
    profile@revision); ``include_detected`` widens a region selection to
    ``detected`` items and is only valid together with a profile selector.
    """

    model_config = ConfigDict(extra='forbid')

    region_status: list[str] = Field(default_factory=list)
    detector: list[str] = Field(default_factory=list)
    reason: list[str] = Field(default_factory=list)
    profile_not: str | None = None
    profile_revision_below: int | None = Field(default=None, ge=0)
    include_detected: bool = False
    missing_status: bool = False
    missing_provenance: bool = False
    import_id: str | None = None
    source: str | None = None
    class_id: int | None = None
    dataset_split: str | None = None
    all_images: bool = False
    open_vocab_status: list[str] = Field(default_factory=list)
    """Image-level selectors (``all_images``, ``open_vocab_status``) select
    from the images index, so they also reach images with no item yet; they
    only combine with the image-unit scopes and not with the item selectors."""

    def is_empty(self) -> bool:
        return self == ReprocessFilter()


class ReprocessTargets(BaseModel):
    """Exactly one of ``image_ids`` / ``crop_ids`` / ``filter``."""

    model_config = ConfigDict(extra='forbid')

    image_ids: list[str] | None = None
    crop_ids: list[str] | None = None
    filter: ReprocessFilter | None = None


class ReprocessRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    targets: ReprocessTargets
    scopes: list[ReprocessScope] = Field(min_length=1)
    region_mode: RegionMode = 'redetect'
    dry_run: bool = True


class ReprocessOneRequest(BaseModel):
    """The body of ``POST /images/{id}/reprocess`` and
    ``POST /crops/{id}/reprocess``: the target is the path."""

    model_config = ConfigDict(extra='forbid')

    scopes: list[ReprocessScope] = Field(min_length=1)
    region_mode: RegionMode = 'redetect'
    dry_run: bool = False


class BreakdownRow(BaseModel):
    detector: str
    reason: str
    count: int


class ReprocessScopeResult(BaseModel):
    scope: ReprocessScope
    selected: int = 0
    locked_skipped: int = 0
    queued: int = 0
    not_found: int = 0
    failed: int = 0
    breakdown: list[BreakdownRow] = Field(default_factory=list)
    detail: dict[str, int] = Field(default_factory=dict)
    """Scope-specific counters (``detect``: merged/refreshed/replaced/
    created/removed; ``open_vocab``: see ``reprocess_open_vocab``; ``vlm``:
    restored/cleared; ``embed``: images/items)."""


class ReprocessJobInfo(BaseModel):
    job_id: str
    status: str
    scopes: list[ReprocessScope] = Field(default_factory=list)
    images_total: int = 0
    images_done: int = 0
    images_failed: int = 0
    error: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None
    results: list[ReprocessScopeResult] = Field(default_factory=list)
    poll_after_s: int | None = None


class ReprocessResponse(BaseModel):
    dry_run: bool
    scopes: list[ReprocessScopeResult]
    job: ReprocessJobInfo | None = None


__all__ = [
    'MAX_TARGET_IDS',
    'BreakdownRow',
    'RegionMode',
    'ReprocessFilter',
    'ReprocessJobInfo',
    'ReprocessOneRequest',
    'ReprocessRequest',
    'ReprocessResponse',
    'ReprocessScope',
    'ReprocessScopeResult',
    'ReprocessTargets',
]
