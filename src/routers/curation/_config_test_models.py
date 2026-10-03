"""Wire models for the test-on-crop routes (W5): ``POST /prompt_packs/test``
and ``POST /region_profiles/test``. Both run a draft or saved pack / profile
(and, for the VLM, a draft or saved endpoint) against stored crops of the
bound project and return the preview. Neither writes anything.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.routers.curation._config_common_models import (
    ValidationReport,  # noqa: TC001 - pydantic field type
)
from src.routers.curation._item_models import ItemDoc  # noqa: TC001 - pydantic field type
from src.routers.curation._prompt_pack_models import (
    PromptPackBody,  # noqa: TC001 - pydantic field type
)
from src.routers.curation._region_profile_models import (
    RegionProfileBody,  # noqa: TC001 - pydantic field type
)
from src.services.labeling.vlm_endpoint_body import (
    VlmEndpointBody,  # noqa: TC001 - pydantic field type
)
from src.services.labeling.vlm_probe import ProbeCall  # noqa: TC001 - pydantic field type


UseRegionBox = Literal['current', 'none']
DropReason = Literal['below_min_score', 'nms', 'over_max']


class VlmSourceFields(BaseModel):
    """Which VLM endpoint a test uses; at most one of ``vlm_name`` /
    ``vlm_draft`` (none = the project's active endpoint). A draft or saved
    endpoint outside this deployment needs ``acknowledge_external``, because
    a real crop is sent."""

    model_config = ConfigDict(extra='forbid')

    vlm_name: str | None = None
    vlm_revision: int | None = None
    vlm_draft: VlmEndpointBody | None = None
    acknowledge_external: bool = False


class PackTestRequest(VlmSourceFields):
    """Run ``call`` of a pack over stored crops. Pack source: exactly one of
    ``pack_name`` (+ ``pack_revision``) or ``draft``; none = the active pack."""

    pack_name: str | None = None
    pack_revision: int | None = None
    draft: PromptPackBody | None = None
    call: ProbeCall
    crop_ids: list[str] = Field(default_factory=list)
    #: ``combined`` only: ``current`` draws every stored box as the numbered
    #: overlay, ``none`` sends the crop with no boxes.
    use_region_box: UseRegionBox = 'current'
    #: Class catalog override; default = the project's registry.
    class_names: list[str] | None = None
    #: Region-text gating follows this profile (default = the active one).
    profile_name: str | None = None


class PackTestPackRef(BaseModel):
    name: str | None
    revision: int | None
    draft: bool


class PackTestVlmRef(BaseModel):
    """The endpoint that answered: ``endpoint`` is its ``name@revision`` ref
    (``_draft@<sha>`` for a draft, ``env@<sha>`` for the built-in)."""

    endpoint: str
    model: str
    name: str | None
    revision: int | None
    draft: bool


class PackTestPrompt(BaseModel):
    system: str | None
    user_text: str | None


class PackTestCropResult(BaseModel):
    crop_id: str
    #: ``region_verify``: the stored box this verdict is about.
    box_id: str | None = None
    #: The production parser's reading of this crop's answer; ``None`` = no
    #: usable answer for it.
    parsed: dict[str, Any] | bool | None = None
    #: The item as the production write would leave it (``ItemDoc`` wire);
    #: ``None`` for ``region_visible`` (no write) or a skipped item.
    preview_item: ItemDoc | None = None
    #: Why no preview is shown (e.g. ``class_locked``).
    skipped: str | None = None


class PackTestResponse(BaseModel):
    call: ProbeCall
    pack: PackTestPackRef
    vlm: PackTestVlmRef
    prompt: PackTestPrompt
    raw_reply: str | None
    reasoning: str | None
    latency_ms: float
    #: False when the reply did not parse (a result, not an error).
    parse_ok: bool
    parse_error: str | None
    results: list[PackTestCropResult]
    validation: ValidationReport


class RegionTestRequest(VlmSourceFields):
    """Run a region profile's legs over one stored crop. Profile source: at
    most one of ``profile_name`` (+ ``profile_revision``) or ``draft``; none
    = the active profile."""

    crop_id: str
    profile_name: str | None = None
    profile_revision: int | None = None
    draft: RegionProfileBody | None = None
    #: The quick "try this prompt" override of ``segmenter_text_prompt``.
    segmenter_text_prompt: str | None = None
    #: Also run the combined VLM call over the selected boxes. The pack is
    #: ``prompt_pack_name`` (+ revision), ``prompt_pack_draft`` or the active one.
    verify: bool = False
    prompt_pack_name: str | None = None
    prompt_pack_revision: int | None = None
    prompt_pack_draft: PromptPackBody | None = None


class RegionTestCandidate(BaseModel):
    """One raw candidate of a leg, as a complete region-box wire element
    (``box_id`` is ``null``: ids are assigned on write) plus what the
    selection did with it."""

    box_id: str | None
    bbox_norm: list[float]
    state: Literal['proposed']
    score: float | None
    detector: str | None
    detector_version: str | None
    source: str | None
    bbox_correct: bool | None
    confidence: str | None
    rejection_reason: str | None
    text: str | None
    text_raw: str | None
    text_source: str | None
    text_engine_version: str | None
    text_confidence: float | None
    text_vlm: str | None
    text_ocr: str | None
    text_choice: str | None
    text_vlm_invalid: str | None
    text_disagreement: bool | None
    cluster_id: int | None
    cluster_subid: str | None
    cluster_distance: float | None
    detected_at: str | None
    locked: bool
    #: The item-crop frame (what the leg returned); ``bbox_norm`` is the
    #: source-image frame.
    bbox_in_parent: list[float] | None
    thumbnail_url: str | None
    #: Position in the leg's raw list: the stable render key.
    candidate_index: int
    selected: bool
    drop_reason: DropReason | None
    mask_iou: float | None
    #: The segmenter's mask outline, source-image frame / item-crop frame.
    mask_polygon: list[list[float]] | None
    mask_polygon_in_parent: list[list[float]] | None


class RegionTestLeg(BaseModel):
    leg: Literal['detector', 'segmenter']
    #: ``skipped``: not run (see ``reason``); ``error``: this leg failed (the
    #: run still answers 200 with the other legs).
    status: Literal['ok', 'skipped', 'error']
    reason: str | None = None
    elapsed_ms: float | None = None
    candidates: list[RegionTestCandidate] = Field(default_factory=list)


class RegionTestProfileRef(BaseModel):
    name: str | None
    revision: int | None
    draft: bool


class RegionTestVerify(BaseModel):
    pack: PackTestPackRef
    vlm: PackTestVlmRef
    prompt: PackTestPrompt
    raw_reply: str | None
    reasoning: str | None
    latency_ms: float
    parse_ok: bool
    parse_error: str | None


class RegionTestResponse(BaseModel):
    crop_id: str
    profile: RegionTestProfileRef
    #: Whether the item's class is one the profile applies to (its
    #: ``parent_classes``); the worker would skip it otherwise.
    item_eligible: bool
    #: Same verdict as ``item_eligible`` under the name the UI gates its
    #: run button on; ``reason`` says why when false, null otherwise.
    testable: bool
    reason: str | None = None
    legs: list[RegionTestLeg]
    verify: RegionTestVerify | None = None
    #: ``selection_accepted``: no VLM ran, so every selected box is shown as
    #: the verifier accepting it; ``vlm_verdicts``: the combined call's
    #: verdicts decided the boxes.
    preview_basis: Literal['selection_accepted', 'vlm_verdicts']
    #: The item as the worker would leave it (``ItemDoc`` wire).
    preview_item: ItemDoc
    validation: ValidationReport


__all__ = [
    'DropReason',
    'PackTestCropResult',
    'PackTestPackRef',
    'PackTestPrompt',
    'PackTestRequest',
    'PackTestResponse',
    'PackTestVlmRef',
    'RegionTestCandidate',
    'RegionTestLeg',
    'RegionTestProfileRef',
    'RegionTestRequest',
    'RegionTestResponse',
    'RegionTestVerify',
    'UseRegionBox',
    'VlmSourceFields',
]
