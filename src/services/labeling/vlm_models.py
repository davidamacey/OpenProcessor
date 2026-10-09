"""Input/output models and error types of the VLM labeler.

Split out of ``vlm_labeler.py``; the models carry no behaviour beyond validation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.services.labeling.region_overlay import (  # noqa: TC001 - pydantic resolves it at runtime
    VlmBoxVerdict,
)


ConfidenceLevel = Literal['high', 'medium', 'low']
ClassReplyFailure = Literal['request_failed', 'unparseable']


# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------


class ItemCrop(BaseModel):
    """A single item crop sent to the VLM for class prediction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    img_id: str = Field(..., description='Caller-controlled crop identifier (e.g. crop UUID).')
    jpeg_bytes: bytes = Field(
        ...,
        description='JPEG-encoded crop bytes. Caller is responsible for resizing to a sane size '
        '(e.g. ≤ 768 px on the long edge) before calling.',
    )
    detector_class: str = Field(
        '',
        description="The item's stored detector class name, set only when the pack opts in "
        '(``detector_hint_min_confidence_pct``); empty = no hint.',
    )
    detector_confidence: float | None = Field(
        None, description='Detector confidence for ``detector_class``.'
    )


class RegionCrop(BaseModel):
    """A single sub-region crop sent to the VLM for is-it-real verification."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crop_id: str = Field(..., description='Caller-controlled region-crop identifier.')
    jpeg_bytes: bytes = Field(..., description='JPEG-encoded region crop bytes.')


class CombinedCrop(BaseModel):
    """Input for the batched ``label_combined_batch`` call.

    Carries the item crop JPEG, this item's candidate region boxes (W8:
    always a list -- N=1 is a list of one, empty when no candidate
    exists from an upstream detector), and a per-crop ``classify`` flag
    so a single batch can mix low-confidence crops (need class label)
    and high-confidence crops (caller already trusts the class).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crop_id: str = Field(..., description='Caller-controlled crop identifier.')
    jpeg_bytes: bytes = Field(..., description='Item crop JPEG bytes.')
    region_bboxes_norm: list[tuple[float, float, float, float]] = Field(
        default_factory=list,
        description=(
            'This item candidate region bboxes in normalized crop coords '
            '[x1, y1, x2, y2] (W8: always a list, possibly empty). When '
            'non-empty, every box is drawn as a numbered overlay on the JPEG '
            'before encoding so the VLM can confirm each one visually.'
        ),
    )
    classify: bool = Field(
        default=True,
        description=(
            'If True, ask the VLM for the item class_id. If False (caller '
            'already has a trusted class), the VLM returns class_id=null and '
            'only fills region fields.'
        ),
    )


class VlmClassPrediction(BaseModel):
    """Prediction for a single item crop."""

    img_id: str
    class_name: str = Field(
        ..., description='Predicted class. Empty string on unrecoverable parse error.'
    )
    confidence: ConfidenceLevel = Field(
        ..., description="One of 'high' | 'medium' | 'low'. Defaults to 'low' on parse fallback."
    )
    proposed_class: str = Field(
        default='',
        description=(
            'Non-empty when the VLM rejected every existing class and proposed a new slug. '
            'Used by the curator queue to surface candidates for new-class review.'
        ),
    )
    # Captures the VLM's raw answer even when the parser fails to map it
    # to a registry class, so a curator can grow the registry from it.
    raw_response: str = Field(
        default='',
        description='Raw text the VLM returned for this crop (best-effort), even on parse failure.',
    )
    make: str = Field(default='', description='Free-text attribute 1 when visible, else "".')
    model: str = Field(default='', description='Free-text attribute 2 when visible, else "".')
    region_visible: bool | None = Field(
        default=None,
        description='Whether the VLM sees a sub-region-of-interest on this crop; None when '
        'not reported.',
    )
    failure: ClassReplyFailure | None = Field(
        default=None,
        description=(
            "Why there is no answer for this crop: 'request_failed' (the call never "
            "completed) or 'unparseable' (no usable entry for this crop in the reply). "
            'None when the reply was parsed -- even if its class is empty.'
        ),
    )


class VlmRegionVerdict(BaseModel):
    """Verdict for a single sub-region crop.

    The ``text`` / ``text_confidence`` fields are populated when the VLM
    reads the region during verify. The same call serves double duty:
    (1) is-this-a-real-region, and (2) what-does-it-say. Combining them
    saves a round-trip per crop.
    """

    crop_id: str
    is_region: bool
    confidence: ConfidenceLevel
    reason: str = Field(default='', description='≤15-word free-text reason from the VLM.')
    text: str | None = Field(
        default=None,
        description='Region text as read by the VLM. None if no region or unreadable.',
    )
    text_confidence: ConfidenceLevel | None = Field(
        default=None,
        description="The VLM's confidence in the text read. None when text is None.",
    )


class VlmCombinedReply(BaseModel):
    """One VLM call returns class + region-verify + region-text.

    Cuts a 3-call worst-case (class fill + region verify + region read)
    to a single round-trip. Crops with a caller-trusted class skip
    class_id and answer just the region fields.

    **D-B (owner decision, 2026-09-26):** the pre-W8 flat region-verify
    shape (``region_bbox_correct`` / ``region_text_reply`` /
    ``region_confidence`` as top-level fields) is dropped -- replaced by
    ``region_boxes``, one :class:`~src.services.labeling.region_overlay
    .VlmBoxVerdict` per candidate box offered (aligned by position with
    the request's ``region_bboxes_norm``, W8.6). ``region_visible``
    stays a separate top-level answer ("is anything region-like visible
    in this photo at all", independent of any specific candidate box).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    img_id: str
    class_id: int | None = Field(
        default=None,
        description='Predicted class id, or -1 if no class matches, or None when skipped.',
    )
    class_confidence: ConfidenceLevel | None = None
    region_visible: bool = False
    region_boxes: list[VlmBoxVerdict] = Field(
        default_factory=list,
        description='One verdict per candidate box offered, aligned by position (box 1..N).',
    )
    make: str = Field(default='', description='Free-text attribute 1 when visible, else "".')
    model: str = Field(default='', description='Free-text attribute 2 when visible, else "".')
    class_raw: str = Field(
        default='',
        description=(
            'The class the VLM named when it answered with a label instead of a '
            "catalog index and the label isn't in the catalog; '' otherwise."
        ),
    )


class CombinedParseFailure(Exception):  # noqa: N818 - documented public symbol
    """Raised when ``label_combined`` cannot parse the VLM's response.

    Callers should fall back to the existing separate-call paths
    (``label_item_batch`` + ``verify_region_batch``) for the affected
    crop.
    """


class VlmTransportError(Exception):
    """An upstream VLM call failed (HTTP / transport): no reply at all.

    Distinct from a reply without a verdict (empty, unparseable, null): a
    caller that bounds retries of no-verdict replies must keep retrying
    these -- the VLM may just be down.
    """


class CombinedTransportError(CombinedParseFailure, VlmTransportError):
    """The combined call itself failed (HTTP / transport): no reply at all.

    Subclasses :class:`CombinedParseFailure` so callers that only fall
    back on any failure keep working unchanged.
    """


class VlmHealth(BaseModel):
    """Health-probe response."""

    reachable: bool
    model: str
    last_error: str | None = None
