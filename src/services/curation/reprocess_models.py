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

from src.services.curation.item_filter import ItemFilter
from src.services.curation.item_selection import Sample  # noqa: TC001 - pydantic resolves it
from src.services.curation.open_vocab_fields import OpenVocabStatus  # noqa: TC001 - pydantic


ReprocessScope = Literal['detect', 'open_vocab', 'region', 'vlm', 'embed']
RegionMode = Literal['redetect', 'reverify']

MAX_TARGET_IDS = 5000


class ReprocessFilter(ItemFilter):
    """The item filter every list route takes, plus the requeue and provenance
    selectors only a reprocess needs.

    ``profile_not`` / ``profile_revision_below`` select items not produced
    by the profile@revision named (both given: not produced by that exact
    profile@revision); ``include_detected`` widens a region selection to
    ``detected`` items and is only valid together with a profile selector.
    """

    region_status: list[str] = Field(default_factory=list)
    detector: list[str] = Field(default_factory=list)
    reason: list[str] = Field(default_factory=list)
    profile_not: str | None = None
    profile_revision_below: int | None = Field(default=None, ge=0)
    include_detected: bool = False
    missing_status: bool = False
    missing_provenance: bool = False
    all_images: bool = False
    open_vocab_status: list[OpenVocabStatus] = Field(default_factory=list)
    """Image-level selectors (``all_images``, ``open_vocab_status``) select
    from the images index, so they also reach images with no item yet; they
    only combine with the image-unit scopes and not with the item selectors."""


class ReprocessTargets(BaseModel):
    """Exactly one of ``image_ids`` / ``crop_ids`` / ``filter``.

    ``limit`` / ``sample`` / ``seed`` cap a ``filter`` selection to its
    ``limit`` largest boxes or a seeded random draw of that size (see
    :mod:`~src.services.curation.item_selection`).
    """

    model_config = ConfigDict(extra='forbid')

    image_ids: list[str] | None = None
    crop_ids: list[str] | None = None
    filter: ReprocessFilter | None = None
    limit: int | None = Field(default=None, ge=1)
    sample: Sample | None = None
    seed: int = 0


EmbedPart = Literal['crop', 'frame', 'region']


class EmbedOptions(BaseModel):
    """How the ``embed`` scope runs. ``only_missing`` skips crop vectors an item
    already has (and, with the default parts, leaves the frame vector alone);
    without it every selected vector is rewritten, which is how an embedding
    model change is re-embedded. ``parts`` defaults to crop and region when
    ``only_missing`` and to all three otherwise."""

    model_config = ConfigDict(extra='forbid')

    only_missing: bool = False
    parts: list[EmbedPart] | None = None


class ReprocessRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    targets: ReprocessTargets
    scopes: list[ReprocessScope] = Field(min_length=1)
    region_mode: RegionMode = 'redetect'
    embed: EmbedOptions = Field(default_factory=EmbedOptions)
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
    detail: dict[str, int | float | bool | str] = Field(default_factory=dict)
    """Scope-specific facts: counters (``detect``: merged/refreshed/replaced/
    created/removed; ``vlm``: restored/cleared; ``embed``: images/items and a
    float ``estimated_vector_kb``), and for ``open_vocab`` also a float
    ``estimated_minutes`` and a bool ``segmenter_reachable`` (see
    ``reprocess_open_vocab``)."""

    def add_count(self, key: str, n: int) -> None:
        """Add ``n`` to the integer counter ``key`` (absent counts as 0)."""
        current = self.detail.get(key, 0)
        self.detail[key] = (current if isinstance(current, int) else 0) + n


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
    'EmbedOptions',
    'EmbedPart',
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
