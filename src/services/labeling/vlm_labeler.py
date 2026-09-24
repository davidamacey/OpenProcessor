"""
Generic vision-language-model (VLM) labeler service.

Talks to any OpenAI-compatible vision ``/chat/completions`` endpoint
(the reference deployment is ``vllm-gemma4-e4b`` behind OpenWebUI, but
nothing here names that model) to:

- batch-classify item crops into one of a caller-supplied set of class
  names (closed- or open-vocabulary)
- verify whether a crop's sub-region-of-interest (e.g. a printed label
  on a product photo; a license plate on a vehicle crop) is real, and
  read any text on it

Split out of the reference ``gemma_labeler.py`` (a 3-way split: this
module is the orchestration half — transport lives in
``vlm_client.py``, prompt/vocabulary data lives in ``vlm_prompts.py``).
This module lands over the 700-LOC pre-commit ratchet cap on arrival;
that is expected (see ``docs/design/curation_design_rationale.md`` §5)
— the follow-up split of ``VlmLabeler``'s class body is out of scope
for this port.

Design notes
------------
- The upstream VLM is typically launched with a hard cap on images per
  prompt (e.g. ``--limit-mm-per-prompt '{"image":8}'``), so calls chunk
  crops into batches of at most ``max_images_per_call``.
- Uses tenacity (via ``vlm_client.post_chat_with_retry``) for
  retry-with-backoff on transient 5xx + connection errors.
- Robust JSON parsing — VLMs will sometimes wrap output in ```json
  fences despite the system prompt. We strip them and on parse failure
  return all-low-confidence fallbacks so the rest of the pipeline can
  still funnel the crop into the human-review queue rather than
  crashing the batch.
- The wire keys a prompt asks the VLM to return for the
  region-of-interest sub-annotation (``region_visible``,
  ``region_bbox_correct``, ``region_text``, ``region_confidence``) are
  read via a :class:`~src.config.RegionFields` instance rather than
  hardcoded literals (§3.2), so a deployment with existing data under
  different field names (e.g. a proprietary-dataset overlay using
  ``plate_*``) is a config flip, not a code change. A :class:`~src.services.labeling
  .vlm_prompts.PromptPack`'s own templates ask the VLM for the matching
  key names — see that module's docstring.

Public surface
--------------
- :class:`ItemCrop`, :class:`RegionCrop`, :class:`CombinedCrop` — input
  dataclasses
- :class:`VlmClassPrediction`, :class:`VlmRegionVerdict`,
  :class:`VlmCombinedReply`, :class:`VlmHealth` — output dataclasses
- :class:`VlmLabeler` — async client; one instance is intended to be
  shared across the FastAPI process.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import RegionFields, get_region_fields
from src.core.logging import get_logger
from src.services.labeling.vlm_client import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_IMAGES_PER_CALL,
    DEFAULT_MODEL,
    DEFAULT_OPEN_IMAGES_PER_CALL,
    DEFAULT_REQUESTS_PER_SECOND,
    _TokenBucket,
    build_auth_headers,
    build_http_client,
    extract_message_content,
    extract_reasoning_content,
    post_chat_with_retry,
)
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, PromptPack


if TYPE_CHECKING:
    import httpx


logger = get_logger(__name__)


ConfidenceLevel = Literal['high', 'medium', 'low']
ClassReplyFailure = Literal['request_failed', 'unparseable']

# vLLM's ``json_object`` grammar only admits an object, so every batched
# class prompt asks for the per-image array wrapped in ``{"results": ...}``.
_RESULTS_ENVELOPE = 'Wrap the array in one JSON object: {"results": [ ...one entry per image... ]}.'


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


class RegionCrop(BaseModel):
    """A single sub-region crop sent to the VLM for is-it-real verification."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crop_id: str = Field(..., description='Caller-controlled region-crop identifier.')
    jpeg_bytes: bytes = Field(..., description='JPEG-encoded region crop bytes.')


class CombinedCrop(BaseModel):
    """Input for the batched ``label_combined_batch`` call.

    Carries the item crop JPEG, the candidate region bbox (when one
    exists from an upstream detector), and a per-crop ``classify`` flag
    so a single batch can mix low-confidence crops (need class label)
    and high-confidence crops (caller already trusts the class).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crop_id: str = Field(..., description='Caller-controlled crop identifier.')
    jpeg_bytes: bytes = Field(..., description='Item crop JPEG bytes.')
    plate_bbox_norm: tuple[float, float, float, float] | None = Field(
        default=None,
        description=(
            'Optional candidate sub-region bbox in normalized crop coords '
            '[x1, y1, x2, y2]. When provided, the bbox is drawn as a colored '
            'overlay on the JPEG before encoding so the VLM can confirm '
            '"is the box correct?" visually.'
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
    plate_visible: bool | None = Field(
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
    """

    img_id: str
    class_id: int | None = Field(
        default=None,
        description='Predicted class id, or -1 if no class matches, or None when skipped.',
    )
    class_confidence: ConfidenceLevel | None = None
    plate_visible: bool = False
    plate_bbox_correct: bool | None = Field(
        default=None,
        description='True if the proposed region bbox correctly outlines the sub-region; '
        'False if the sub-region is visible elsewhere; None when the reply gave no '
        'verdict on the box (no candidate supplied, or the answer was null / absent).',
    )
    plate_text: str | None = None
    plate_confidence: ConfidenceLevel | None = None
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
    (``label_vehicle_batch`` + ``verify_plate_batch``) for the affected
    crop.
    """


class VlmHealth(BaseModel):
    """Health-probe response."""

    reachable: bool
    model: str
    last_error: str | None = None


# ---------------------------------------------------------------------------
# Helpers — vocabulary / catalog formatting (mechanism; data is PromptPack)
# ---------------------------------------------------------------------------


def format_class_catalog(classes: list[dict[str, Any]], pack: PromptPack) -> str:
    """Render the registry as a grouped, described catalog for the prompt.

    Output looks like::

        cruisers: cruiserbike (low-slung Harley...), vintagebike (pre-1980...)
        sportbikes: sportbike (aggressive forward...)
        cars: acura, buick, ...

    Tokens roughly 2x the flat CSV but the structure plus descriptions sharply
    improves the VLM's ability to pick the right slug for ambiguous/oblique shots.

    Args:
        classes: List of registry class dicts with ``class_name`` and
            ``group`` keys (and ``deprecated`` to skip).
        pack: The :class:`PromptPack` whose ``class_descriptions`` supply
            the optional per-class disambiguator text.
    """
    by_group: dict[str, list[str]] = {}
    for c in classes:
        if c.get('deprecated'):
            continue
        name = c.get('class_name') or c.get('name')
        group = c.get('group') or 'other'
        if not name:
            continue
        desc = pack.class_descriptions.get(name)
        rendered = f'{name} ({desc})' if desc else name
        by_group.setdefault(group, []).append(rendered)

    lines = [f'{group}: {", ".join(sorted(by_group[group]))}' for group in sorted(by_group)]
    return '\n'.join(lines)


def resolve_class_name(
    raw: str | None,
    name_to_id: dict[str, int],
    *,
    confidence: ConfidenceLevel | None = None,
    pack: PromptPack = GENERIC_ITEM_PACK,
) -> str | None:
    """Map a raw VLM reply onto a registry class name (or ``None``).

    Resolution order:
      1. Exact match against an active registry class (``raw in name_to_id``).
      2. Normalised match (lowercase, ``-``/``_`` -> space) against the registry.
      3. Synonym lookup against ``pack.synonyms``, both space- and underscore-joined.

    When ``confidence='low'`` we **skip steps 2 and 3** (the fuzzy /
    synonym attempts) entirely. Empirically low-confidence replies are
    not worth force-fitting into a registry slot — doing so just spends
    CPU normalising a string we have no business trusting, and silently
    overwrites the raw-label field callers persist for later semantic
    clustering. The exact-match path is preserved because the registry
    IS the source of truth — if the VLM typed the slug verbatim we still
    take it.

    Returns ``None`` when nothing matches — callers surface those for
    human review.

    Contract: regardless of whether resolution succeeds, callers MUST
    persist the original ``raw`` answer (or ``proposed_class`` for
    ``__new__`` replies) onto the crop document under a raw-label field.
    That captured value drives registry-growth aggregation and
    hierarchical clustering of unmatched terms.
    """
    if not raw:
        return None
    if raw in name_to_id:
        return raw
    if confidence == 'low':
        logger.debug(
            'vlm_labeler.resolve_skip_force_fit',
            raw=raw,
            reason='confidence_low',
        )
        return None
    norm = raw.strip().lower().replace('-', ' ').replace('_', ' ')
    if norm in name_to_id:
        return norm
    mapped = pack.synonyms.get(norm) or pack.synonyms.get(norm.replace(' ', '_'))
    if mapped and mapped in name_to_id:
        return mapped
    return None


# ---------------------------------------------------------------------------
# Helpers — encoding / parsing
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r'^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$', re.DOTALL)


def _strip_markdown_fences(text: str) -> str:
    """Strip a ```json ... ``` (or plain ``` ... ```) wrapper if present.

    VLMs are told ``no markdown`` but sometimes do it anyway. Be lenient.
    """

    if not text:
        return text
    match = _FENCE_RE.match(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def _b64_jpeg(data: bytes) -> str:
    """Base64-encode JPEG bytes (no data: prefix)."""

    return base64.b64encode(data).decode('ascii')


def _draw_bbox_overlay(
    jpeg_bytes: bytes,
    bbox_norm: tuple[float, float, float, float],
) -> bytes | None:
    """Render the candidate sub-region bbox as a colored rectangle on the crop
    (visual conveyance for the combined VLM call).

    Returns the re-encoded JPEG bytes on success, or None if the source is
    not a decodable image (caller falls back to coords-in-prompt).
    """
    try:
        from io import BytesIO

        from PIL import Image, ImageDraw

        with Image.open(BytesIO(jpeg_bytes)) as src:
            im = src.convert('RGB')
            w, h = im.size
            x1, y1, x2, y2 = bbox_norm
            box = (
                max(0, int(x1 * w)),
                max(0, int(y1 * h)),
                min(w - 1, int(x2 * w)),
                min(h - 1, int(y2 * h)),
            )
            draw = ImageDraw.Draw(im)
            # 3-pixel-wide red rectangle is unambiguous against most
            # backgrounds without obscuring detail underneath.
            draw.rectangle(box, outline=(255, 0, 0), width=3)
            out = BytesIO()
            im.save(out, format='JPEG', quality=90)
            return out.getvalue()
    except Exception:
        return None


def _normalize_confidence(value: Any) -> ConfidenceLevel:
    """Coerce a raw model field into one of high|medium|low; default low."""

    if isinstance(value, str):
        v = value.strip().lower()
        if v in ('high', 'medium', 'low'):
            return v  # type: ignore[return-value]
    return 'low'


_TRUE_STRINGS = frozenset({'true', 'yes', 'y', '1'})
_FALSE_STRINGS = frozenset({'false', 'no', 'n', '0'})


def _coerce_bool(value: Any) -> bool | None:
    """Strict boolean read of a VLM reply field; ``None`` when unrecognized.

    ``bool("false")`` is ``True``, so a model that quotes its booleans
    would otherwise have every "false" read as an accept. A quoted null
    (``"null"``, ``"none"``, ``""``) is no answer, not a ``False``: read
    as ``False`` it turned a verifier that gave no box verdict into a
    reject.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value) if value in (0, 1) else None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE_STRINGS:
            return True
        if v in _FALSE_STRINGS:
            return False
    return None


def _clean_combined_region_text(raw: Any, *, echoes: tuple[str, ...]) -> str | None:
    """The region's transcribed text from a combined reply, or ``None``.

    Drops sentinels ("unknown", "n/a", ...) and any value that is really
    one of the reply's own item answers echoed into the text slot
    (``echoes``: the class name it picked, its ``make`` / ``model``, both
    joined). Those describe the item, not text read off the region.
    """
    if raw is None or isinstance(raw, bool | dict | list):
        return None
    text = str(raw).strip()
    if not text or text.lower() in _TEXT_SENTINELS:
        return None
    if _echo_key(text) in {_echo_key(e) for e in echoes if e}:
        return None
    return text[:32]


def _echo_key(value: str) -> str:
    """Case- and separator-insensitive form ("Adventure Bike" == "adventurebike")."""
    return ''.join(ch for ch in value.casefold() if ch.isalnum())


def _combined_reply_from_entry(
    entry: dict[str, Any],
    *,
    img_id: str,
    fields: RegionFields,
    class_names: list[str] | None,
) -> VlmCombinedReply:
    """Build a :class:`VlmCombinedReply` from one parsed reply object.

    Fail-closed: raises ``ValueError`` when the region-visible answer is
    missing or not a recognizable boolean (the caller leaves the item
    pending rather than stamping ``no_region_visible`` or an accept off a
    reply that never answered). ``region_bbox_correct`` reads ``None``
    unless it is a recognizable boolean, so only an explicit ``true``
    can accept a box and only an explicit ``false`` can reject one;
    ``None`` is no verdict.
    """
    visible = _coerce_bool(entry.get(fields.visible))
    if visible is None:
        msg = f'{fields.visible} missing or not a boolean: {entry.get(fields.visible)!r}'
        raise ValueError(msg)
    class_id, class_raw = _coerce_class_answer(entry, class_names)
    class_conf_raw = entry.get('class_confidence')
    region_conf_raw = entry.get(fields.confidence)
    make = str(entry.get('make') or '').strip()[:48]
    model_name = str(entry.get('model') or '').strip()[:48]
    picked_class = (
        class_names[class_id]
        if class_names and class_id is not None and 0 <= class_id < len(class_names)
        else ''
    )
    echoes = (picked_class, make, model_name, f'{make} {model_name}'.strip())
    return VlmCombinedReply(
        img_id=img_id,
        class_id=class_id,
        class_confidence=(
            _normalize_confidence(class_conf_raw) if class_conf_raw is not None else None
        ),
        plate_visible=visible,
        plate_bbox_correct=_coerce_bool(entry.get(fields.bbox_correct)),
        plate_text=_clean_combined_region_text(entry.get(fields.text), echoes=echoes),
        plate_confidence=(
            _normalize_confidence(region_conf_raw) if region_conf_raw is not None else None
        ),
        make=make,
        model=model_name,
        class_raw=class_raw,
    )


_INDEX_ANSWER_RE = re.compile(r'^(-?\d+)(?:\s*[=:].*)?$', re.DOTALL)


def _coerce_class_answer(
    entry: dict[str, Any], class_names: list[str] | None
) -> tuple[int | None, str]:
    """Read a combined reply's class answer as ``(index, raw_label)``.

    The prompt asks for ``class_id`` as an index into the catalog, but
    models also answer with the catalog entry itself (``"3=sedan"``), the
    class name (``"sedan"``), or a ``class`` / ``class_name`` key. An
    index form returns ``(index, '')``; a name found in ``class_names``
    (case-insensitive) returns its index; any other non-empty label
    returns ``(None, label)`` so the caller can record what the VLM
    actually said. No answer returns ``(None, '')``.
    """
    value = entry.get('class_id')
    if value is None:
        value = entry.get('class', entry.get('class_name'))
    if value is None or isinstance(value, bool | dict | list):
        return None, ''
    if isinstance(value, int | float):
        return (int(value), '') if float(value).is_integer() else (None, '')
    text = str(value).strip()
    if not text or text.lower() in ('null', 'none'):
        return None, ''
    match = _INDEX_ANSWER_RE.match(text)
    if match:
        return int(match.group(1)), ''
    folded = text.casefold()
    for i, name in enumerate(class_names or []):
        if name.casefold() == folded:
            return i, ''
    return None, text[:64]


def _decoded_json_values(text: str) -> list[Any]:
    """Every JSON array/object embedded in ``text``, outermost first.

    Reasoning-channel text is prose with the JSON answer somewhere in it;
    this finds each top-level ``[...]`` / ``{...}`` that decodes.
    """
    decoder = json.JSONDecoder()
    out: list[Any] = []
    pos = 0
    while pos < len(text):
        starts = [i for i in (text.find('[', pos), text.find('{', pos)) if i >= 0]
        if not starts:
            break
        start = min(starts)
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        out.append(value)
        pos = end
    return out


def _class_reply_entries(raw: str) -> list[Any] | None:
    """The per-image entry list of a batched class reply, or ``None``.

    Accepts a bare array, a ``{"results"|"predictions"|"data": [...]}``
    envelope, a single per-image object, or any of those embedded in
    prose (the last one found wins -- a reasoning trace ends with its
    answer).
    """
    try:
        values = [json.loads(raw)]
    except json.JSONDecodeError:
        values = _decoded_json_values(raw)
    for value in reversed(values):
        if isinstance(value, dict):
            for key in ('results', 'predictions', 'data'):
                if isinstance(value.get(key), list):
                    return value[key]
            if 'class' in value or 'img' in value:
                return [value]
        if isinstance(value, list):
            return value
    return None


def _request_failed(chunk: list[ItemCrop]) -> list[VlmClassPrediction]:
    return [
        VlmClassPrediction(
            img_id=c.img_id, class_name='', confidence='low', failure='request_failed'
        )
        for c in chunk
    ]


def _align_batch_entries(parsed: list[Any], n: int) -> list[dict[str, Any] | None] | None:
    """Map a batch reply's entries onto input positions ``0..n-1``.

    Returns ``None`` when the mapping can't be trusted, so the caller
    treats the whole chunk as a parse failure instead of handing one
    image's verdict to another:

    * a duplicate or out-of-range ``img`` index,
    * a mix of indexed and un-indexed entries,
    * un-indexed entries whose count differs from ``n``.

    ``img`` is 1-based per the prompt; a reply that is consistently
    0-based (contains ``0``) is shifted. Missing indices map to ``None``.
    """
    entries = [e for e in parsed if isinstance(e, dict)]
    raw_idx = [e.get('img') for e in entries]
    if all(i is None for i in raw_idx):
        return list(entries) if len(entries) == n else None
    idx: list[int] = []
    for i in raw_idx:
        if i is None or isinstance(i, bool):
            return None
        try:
            idx.append(int(i))
        except (TypeError, ValueError):
            return None
    offset = 1 if 0 in idx else 0
    out: list[dict[str, Any] | None] = [None] * n
    for entry, i in zip(entries, idx, strict=True):
        pos = i + offset - 1
        if not 0 <= pos < n or out[pos] is not None:
            return None
        out[pos] = entry
    return out


# ---------------------------------------------------------------------------
# VlmLabeler
# ---------------------------------------------------------------------------


class VlmLabeler:
    """Async client for an OpenAI-compatible vision-chat VLM endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        api_key: str = DEFAULT_API_KEY,
        max_images_per_call: int = DEFAULT_MAX_IMAGES_PER_CALL,
        requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
        # 60s was tuned for an under-loaded upstream. Under heavy
        # concurrency, per-call latency can climb to 30-90s when KV
        # cache is saturated. 240s gives the worst-case batched
        # verify-and-read call room to finish without dropping the
        # request.
        timeout_s: float = 240.0,
        client: httpx.AsyncClient | None = None,
        pack: PromptPack = GENERIC_ITEM_PACK,
        fields: RegionFields | None = None,
    ) -> None:
        if base_url and not model:
            raise ValueError(
                f'OP_VLM_MODEL is required when a VLM URL is set (base_url={base_url!r}); '
                'there is no default model id.'
            )
        if max_images_per_call < 1:
            raise ValueError('max_images_per_call must be >= 1')
        # Hard cap = the deployment's configured upstream limit
        # (OP_VLM_MAX_IMAGES_PER_CALL), so even an explicit caller value
        # can't exceed what the serving engine accepts per prompt.
        _max_hard = DEFAULT_MAX_IMAGES_PER_CALL
        if max_images_per_call > _max_hard:
            logger.warning(
                'vlm_labeler.max_images_clamped',
                requested=max_images_per_call,
                clamped_to=_max_hard,
                reason=(
                    f'vlm_labeler hard cap is {_max_hard} (OP_VLM_MAX_IMAGES_PER_CALL); '
                    "align it with your VLM deployment's per-prompt image limit."
                ),
            )
            max_images_per_call = _max_hard

        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.max_images_per_call = max_images_per_call
        self.requests_per_second = requests_per_second
        self.timeout_s = timeout_s
        self._pack = pack
        self._fields = fields or get_region_fields()
        # Optional class-name list used by ``label_combined`` callers so
        # they don't have to thread the registry through every call
        # site. Set externally after init. ``name_to_id`` maps the
        # resolved name back to the registry's authoritative class_id —
        # needed because reply.class_id is the *index* into
        # ``class_names`` (which is filtered for non-deprecated
        # entries), not a registry id.
        self.class_names: list[str] = []
        self.name_to_id: dict[str, int] = {}

        self._bucket = _TokenBucket(rate=requests_per_second)
        self._client = client or build_http_client(timeout_s)
        self._owns_client = client is None

    # ----- lifecycle -----

    async def aclose(self) -> None:
        """Close the underlying httpx client (if owned)."""

        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> VlmLabeler:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # ----- HTTP plumbing -----

    @property
    def _headers(self) -> dict[str, str]:
        return build_auth_headers(self.api_key)

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /chat/completions with retry on 5xx + connection errors."""

        url = f'{self.base_url}/chat/completions'
        return await post_chat_with_retry(self._client, url, self._headers, payload, self._bucket)

    # ----- public API -----

    async def health(self) -> VlmHealth:
        """Return a lightweight reachability probe.

        Issues a 1-token "reply OK" prompt; if any error happens the probe
        marks the service unreachable and surfaces ``last_error``.
        """

        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': 'reply with the single word OK'}],
            'max_tokens': 4,
            'temperature': 0.0,
        }
        try:
            resp = await self._post_chat(payload)
            _ = extract_message_content(resp)
            return VlmHealth(reachable=True, model=self.model, last_error=None)
        except Exception as exc:
            logger.warning(
                'vlm_labeler.health_failed',
                model=self.model,
                base_url=self.base_url,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return VlmHealth(
                reachable=False, model=self.model, last_error=f'{type(exc).__name__}: {exc}'
            )

    async def label_vehicle_batch(
        self,
        crops: list[ItemCrop],
        class_names: list[str],
    ) -> list[VlmClassPrediction]:
        """Label up to ``len(crops)`` item crops, chunked at ``max_images_per_call``.

        Returns one :class:`VlmClassPrediction` per input crop, in the
        same order. On unrecoverable parse errors a low-confidence empty
        prediction is returned for the affected crop so downstream code
        can still funnel it into the human-review queue.
        """

        if not crops:
            return []
        if not class_names:
            raise ValueError('class_names must be a non-empty list')

        # Fire all chunks in parallel (asyncio.gather) so upstream
        # concurrency isn't artificially capped by a sequential
        # `for await`. The TokenBucket (requests_per_second) is the
        # actual global rate limit.
        chunks = [
            crops[i : i + self.max_images_per_call]
            for i in range(0, len(crops), self.max_images_per_call)
        ]
        chunk_results_list = await asyncio.gather(
            *[self._label_chunk(c, class_names) for c in chunks],
            return_exceptions=False,
        )
        results: list[VlmClassPrediction] = []
        for cr in chunk_results_list:
            results.extend(cr)
        return results

    async def _label_chunk(
        self,
        chunk: list[ItemCrop],
        class_names: list[str],
    ) -> list[VlmClassPrediction]:
        user_text = self._pack.class_user_template.format(class_names_csv=', '.join(class_names))
        user_text = f'{user_text}\n{_RESULTS_ENVELOPE}'
        user_content: list[dict[str, Any]] = [{'type': 'text', 'text': user_text}]
        for crop in chunk:
            b64 = _b64_jpeg(crop.jpeg_bytes)
            user_content.append(
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
                }
            )

        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self._pack.class_system},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': 0.0,
            'max_tokens': 512,
            # Same grammar constraint as the combined call: without it a
            # server-side reasoning parser can route the whole answer to
            # the reasoning channel and leave ``content`` empty.
            'response_format': {'type': 'json_object'},
        }

        try:
            response = await self._post_chat(payload)
        except Exception as exc:
            logger.error(
                'vlm_labeler.label_chunk_failed',
                chunk_size=len(chunk),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return _request_failed(chunk)

        return self._parse_class_reply(response, chunk)

    def _parse_class_reply(
        self, response: dict[str, Any], chunk: list[ItemCrop]
    ) -> list[VlmClassPrediction]:
        """Parse a class call's reply, falling back to the reasoning channel.

        Only when ``content`` yields nothing usable for any crop is the
        reasoning text tried -- a parsed ``content`` always wins.
        """
        content = _strip_markdown_fences(extract_message_content(response))
        preds = self._parse_vehicle_response(content, chunk, self._fields)
        if any(p.failure is None for p in preds):
            return preds
        reasoning = extract_reasoning_content(response)
        if not reasoning:
            return preds
        from_reasoning = self._parse_vehicle_response(
            reasoning, chunk, self._fields, log_failures=False
        )
        if any(p.failure is None for p in from_reasoning):
            logger.info(
                'vlm_labeler.class_reply_from_reasoning',
                chunk_size=len(chunk),
                content_preview=content[:80],
            )
            return from_reasoning
        return preds

    @staticmethod
    def _parse_vehicle_response(
        raw: str,
        chunk: list[ItemCrop],
        fields: RegionFields,
        *,
        log_failures: bool = True,
    ) -> list[VlmClassPrediction]:
        """Parse the VLM's JSON-array response into one prediction per chunk crop.

        A crop with no usable entry comes back with ``class_name=''`` and
        ``failure='unparseable'`` -- distinct from a parsed entry whose
        class is empty (``failure=None``).
        """

        # Even on parse failure we preserve the raw response so the
        # curator can review what the VLM actually said.
        raw_excerpt = raw[:200]
        fallback = [
            VlmClassPrediction(
                img_id=c.img_id,
                class_name='',
                confidence='low',
                raw_response=raw_excerpt,
                failure='unparseable',
            )
            for c in chunk
        ]

        if not raw:
            if log_failures:
                logger.warning('vlm_labeler.parse_empty_response', chunk_size=len(chunk))
            return fallback

        parsed = _class_reply_entries(raw)
        if parsed is None:
            if log_failures:
                logger.warning(
                    'vlm_labeler.parse_failed',
                    raw_preview=raw[:200],
                    chunk_size=len(chunk),
                )
            return fallback

        # Map img-index → record. The VLM is told to use 1-based ``img`` ids;
        # entries with no index at all are taken positionally when their
        # count matches the chunk.
        entries = [e for e in parsed if isinstance(e, dict)]
        by_index: dict[int, dict[str, Any]] = {}
        if entries and all(e.get('img') is None for e in entries) and len(entries) == len(chunk):
            by_index = dict(enumerate(entries, start=1))
        for indexed in entries:
            raw_idx = indexed.get('img')
            if raw_idx is None:
                continue
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            by_index[idx] = indexed

        out: list[VlmClassPrediction] = []
        for i, crop in enumerate(chunk, start=1):
            entry = by_index.get(i)
            if entry is None:
                out.append(
                    VlmClassPrediction(
                        img_id=crop.img_id,
                        class_name='',
                        confidence='low',
                        raw_response=raw_excerpt,
                        failure='unparseable',
                    )
                )
                continue
            class_name = str(entry.get('class', '') or '').strip()
            confidence = _normalize_confidence(entry.get('confidence'))
            proposed = str(entry.get('proposed_class', '') or '').strip().lower()
            # Sanitize the proposed slug — keep [a-z0-9_], cap length 32.
            proposed = ''.join(c for c in proposed if c.isalnum() or c == '_')[:32]
            # Per-crop raw_response: prefer the entry's `class` (the
            # VLM's actual answer for this crop) so unmatched
            # classifications land in the raw-label field instead of
            # "". Falls back to the full response excerpt when class is
            # empty.
            per_crop_raw = class_name or proposed or raw_excerpt
            make = str(entry.get('make', '') or '').strip()[:48]
            model = str(entry.get('model', '') or '').strip()[:48]
            visible_raw = entry.get(fields.visible)
            visible = bool(visible_raw) if isinstance(visible_raw, bool) else None
            out.append(
                VlmClassPrediction(
                    img_id=crop.img_id,
                    class_name=class_name,
                    confidence=confidence,
                    proposed_class=proposed,
                    raw_response=per_crop_raw,
                    make=make,
                    model=model,
                    plate_visible=visible,
                )
            )
        return out

    async def label_or_propose_batch(
        self,
        crops: list[ItemCrop],
        class_names: list[str],
        *,
        images_per_call: int | None = None,
        class_catalog: str | None = None,
        cluster_hint: str | None = None,
    ) -> list[VlmClassPrediction]:
        """Label crops, but allow the VLM to propose new classes when nothing fits.

        Identical contract to :py:meth:`label_vehicle_batch` except the
        open-vocabulary prompt is used (the VLM may answer ``__new__``
        with a ``proposed_class`` slug). Predictions for unrecognized
        items come back with ``class_name='__new__'`` and a populated
        ``proposed_class`` — callers (e.g. an auto-label pipeline) route
        those into the curator queue rather than committing them as
        labels.

        Optional ``class_catalog`` (formatted via :func:`format_class_catalog`)
        replaces the bare CSV in the prompt with grouped + described classes,
        sharply improving accuracy on ambiguous slugs. ``cluster_hint`` adds a
        per-batch bias line — pass it when labeling a single cluster's members
        to nudge the VLM toward the dominant class hypothesis.
        """

        if not crops:
            return []
        if not class_names and not class_catalog:
            raise ValueError('class_names or class_catalog must be supplied')

        # The open-vocab prompt is verbose (per-image schema includes
        # ``proposed_class``). Smaller VLMs can produce empty responses
        # when the chunk is too dense, so default to a tighter chunk
        # size than the closed-vocab path.
        if images_per_call is not None:
            per_call = images_per_call
        else:
            per_call = DEFAULT_OPEN_IMAGES_PER_CALL
        per_call = max(1, min(per_call, self.max_images_per_call))
        chunks = [crops[i : i + per_call] for i in range(0, len(crops), per_call)]
        chunk_results_list = await asyncio.gather(
            *[
                self._label_chunk_open(
                    c,
                    class_names,
                    class_catalog=class_catalog,
                    cluster_hint=cluster_hint,
                )
                for c in chunks
            ],
            return_exceptions=False,
        )
        results: list[VlmClassPrediction] = []
        for cr in chunk_results_list:
            results.extend(cr)
        return results

    async def _label_chunk_open(
        self,
        chunk: list[ItemCrop],
        class_names: list[str],
        *,
        class_catalog: str | None = None,
        cluster_hint: str | None = None,
    ) -> list[VlmClassPrediction]:
        if class_catalog:
            user_text = (
                'Class catalog (grouped, with brief descriptions where slugs are not '
                'self-evident):\n'
                f'{class_catalog}\n\n'
                'Label each numbered crop. Respond as one JSON object:\n'
                '{"results": [{"img": 1, "class": "<name|__new__>", '
                '"confidence": "high|medium|low", "proposed_class": "<slug or empty>"}, ...]}'
            )
        else:
            user_text = self._pack.open_class_user_template.format(
                class_names_csv=', '.join(class_names)
            )
            user_text = f'{user_text}\n{_RESULTS_ENVELOPE}'
        if cluster_hint:
            user_text = (
                f'Hint: these crops were grouped together by visual similarity; the '
                f'cluster\'s current dominant class hypothesis is "{cluster_hint}". '
                'Use this as a prior, not a constraint — overrule it if the crop '
                'clearly belongs to a different class.\n\n'
                f'{user_text}'
            )
        user_content: list[dict[str, Any]] = [{'type': 'text', 'text': user_text}]
        for crop in chunk:
            b64 = _b64_jpeg(crop.jpeg_bytes)
            user_content.append(
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
                }
            )
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self._pack.open_class_system},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': 0.0,
            'max_tokens': 768,
            'response_format': {'type': 'json_object'},
        }
        try:
            response = await self._post_chat(payload)
        except Exception as exc:
            logger.error(
                'vlm_labeler.label_chunk_open_failed',
                chunk_size=len(chunk),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return _request_failed(chunk)
        return self._parse_class_reply(response, chunk)

    async def verify_plate(self, crop: RegionCrop) -> VlmRegionVerdict:
        """Verify whether a single sub-region crop is real."""

        b64 = _b64_jpeg(crop.jpeg_bytes)
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self._pack.region_system},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': self._pack.region_user},
                        {
                            'type': 'image_url',
                            'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
                        },
                    ],
                },
            ],
            'temperature': 0.0,
            # Reasoning VLMs think out loud before answering, and the
            # verify-and-read prompt is longer than a verify-only ask.
            # 1024 leaves headroom for a long reasoning preamble plus
            # the ~80-token JSON reply.
            'max_tokens': 1024,
        }

        try:
            response = await self._post_chat(payload)
        except Exception as exc:
            logger.error(
                'vlm_labeler.verify_plate_failed',
                crop_id=crop.crop_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return VlmRegionVerdict(
                crop_id=crop.crop_id,
                is_region=False,
                confidence='low',
                reason='upstream error',
            )

        raw = _strip_markdown_fences(extract_message_content(response))
        return self._parse_plate_response(raw, crop)

    async def verify_plate_batch(
        self,
        crops: list[RegionCrop],
        *,
        images_per_call: int | None = None,
    ) -> list[VlmRegionVerdict]:
        """Verify a list of sub-region crops in chunks of ``max_images_per_call``.

        Packing multiple JPEGs per upstream call amortises the prompt
        prefix, the attention-warmup cost, and per-request scheduler
        overhead across several verdicts — this materially lifts
        verify throughput versus one crop per call, while staying
        inside the upstream images-per-prompt cap.

        Returns one :class:`VlmRegionVerdict` per input crop, in the
        same order. Failed chunks fall back to ``is_region=False,
        confidence='low'`` so the caller can route the crop to human
        review (same behaviour as the single-crop fallback).
        """

        if not crops:
            return []

        per_call = images_per_call if images_per_call is not None else self.max_images_per_call
        per_call = max(1, min(per_call, self.max_images_per_call))

        chunks = [crops[i : i + per_call] for i in range(0, len(crops), per_call)]
        chunk_results_list = await asyncio.gather(
            *[self._verify_plate_chunk(c) for c in chunks],
            return_exceptions=False,
        )
        results: list[VlmRegionVerdict] = []
        for cr in chunk_results_list:
            results.extend(cr)
        return results

    async def _verify_plate_chunk(self, chunk: list[RegionCrop]) -> list[VlmRegionVerdict]:
        """Run one upstream verify call over up to ``max_images_per_call`` crops."""

        if not chunk:
            return []
        # Single-crop chunks reuse the canonical (and battle-tested)
        # single-image prompt + parser to avoid regressing the existing
        # single-call path's accuracy when callers happen to pass a
        # length-1 list.
        if len(chunk) == 1:
            verdict = await self.verify_plate(chunk[0])
            return [verdict]

        user_content: list[dict[str, Any]] = [
            {'type': 'text', 'text': self._pack.region_batch_user}
        ]
        for crop in chunk:
            b64 = _b64_jpeg(crop.jpeg_bytes)
            user_content.append(
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
                }
            )

        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self._pack.region_batch_system},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': 0.0,
            # Batched verify-and-read. Per-image output schema is
            # roughly 80 tokens and the reasoning preamble scales with
            # image count. 2048 covers a worst-case 6-image batch;
            # shorter responses still stop at the real EOS.
            'max_tokens': 2048,
        }

        try:
            response = await self._post_chat(payload)
        except Exception as exc:
            logger.error(
                'vlm_labeler.verify_plate_chunk_failed',
                chunk_size=len(chunk),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return [
                VlmRegionVerdict(
                    crop_id=c.crop_id,
                    is_region=False,
                    confidence='low',
                    reason='upstream error',
                )
                for c in chunk
            ]

        raw = _strip_markdown_fences(extract_message_content(response))
        return self._parse_plate_batch_response(raw, chunk)

    async def label_combined(
        self,
        img_id: str,
        jpeg_bytes: bytes,
        *,
        class_names: list[str] | None = None,
        plate_bbox_norm: tuple[float, float, float, float] | None = None,
        draw_overlay: bool = True,
    ) -> VlmCombinedReply:
        """One VLM call returns class + region-verify + region-text.

        Args:
            img_id: Crop identifier (echoed back on the reply).
            jpeg_bytes: Crop JPEG bytes.
            class_names: Class-name slice to classify against. Pass
                None / [] when the caller only wants the region-side
                answers; ``class_id`` returns None.
            plate_bbox_norm: Candidate sub-region bbox in normalized
                crop coords ``[x1, y1, x2, y2]`` from an upstream
                detector. None when no candidate exists — the VLM
                still answers ``plate_visible``.
            draw_overlay: If True (default), draw the region bbox as a
                colored rectangle on the crop bytes before encoding so
                the VLM reasons about it visually. Falls back to
                coords-in-prompt if drawing fails or is disabled.

        Returns:
            :class:`VlmCombinedReply` with class + region fields filled
            per the cohort.

        Raises:
            CombinedParseFailure: response unparseable. Caller falls
                back to the separate-call paths.
        """
        bytes_to_send = jpeg_bytes
        overlay_drawn = False
        if draw_overlay and plate_bbox_norm is not None:
            drew = _draw_bbox_overlay(jpeg_bytes, plate_bbox_norm)
            if drew is not None:
                bytes_to_send = drew
                overlay_drawn = True

        if class_names:
            class_block = (
                'Identify the item class. Choose ONE class_id from: '
                + ', '.join(f'{i}={name}' for i, name in enumerate(class_names))
                + '. '
            )
        else:
            class_block = "Don't classify (the caller already has a class). Set class_id=null. "

        if plate_bbox_norm is not None and not overlay_drawn:
            region_block = f'The proposed region bbox (normalized) is {list(plate_bbox_norm)}. '
        elif plate_bbox_norm is not None:
            region_block = 'A candidate region bbox is drawn on the crop. '
        else:
            region_block = 'No region-bbox candidate was provided. '

        user_text = self._pack.combined_user_template.format(
            class_block=class_block, region_block=region_block
        )
        b64 = _b64_jpeg(bytes_to_send)
        user_content: list[dict[str, Any]] = [
            {'type': 'text', 'text': user_text},
            {
                'type': 'image_url',
                'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
            },
        ]

        payload: dict[str, Any] = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self._pack.combined_system},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': 0.0,
            'max_tokens': 256,
            'response_format': {'type': 'json_object'},
        }

        try:
            response = await self._post_chat(payload)
        except Exception as exc:
            logger.warning(
                'vlm_labeler.label_combined_http_failed',
                img_id=img_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise CombinedParseFailure(f'http error: {exc}') from exc

        raw = _strip_markdown_fences(extract_message_content(response))
        if not raw:
            logger.info('vlm_labeler.combined_parse_failure', img_id=img_id, reason='empty')
            raise CombinedParseFailure('empty response')
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.info(
                'vlm_labeler.combined_parse_failure',
                img_id=img_id,
                raw_excerpt=raw[:200],
            )
            raise CombinedParseFailure(f'invalid json: {exc}') from exc

        if not isinstance(parsed, dict):
            logger.info('vlm_labeler.combined_parse_failure', img_id=img_id, reason='not-an-object')
            raise CombinedParseFailure('response is not a json object')

        try:
            return _combined_reply_from_entry(
                parsed, img_id=img_id, fields=self._fields, class_names=class_names
            )
        except (TypeError, ValueError) as exc:
            logger.info(
                'vlm_labeler.combined_parse_failure',
                img_id=img_id,
                reason='type-coercion',
                error=str(exc),
            )
            raise CombinedParseFailure(f'type coercion: {exc}') from exc

    async def label_combined_batch(
        self,
        crops: list[CombinedCrop],
        *,
        class_names: list[str] | None = None,
        images_per_call: int | None = None,
        draw_overlay: bool = True,
    ) -> dict[str, VlmCombinedReply | None]:
        """Batched combined call: class + region-verify + region-text for many crops.

        Packs ``images_per_call`` (default ``max_images_per_call``) crops
        per upstream call. Each crop carries an optional candidate
        region bbox which is drawn as a colored overlay on the JPEG
        before encoding so the VLM can reason about it visually.

        Returns ``{crop_id: VlmCombinedReply | None}``. A ``None`` value
        means the per-crop entry could not be parsed (missing in
        response, bad JSON, or whole-chunk HTTP failure) — the caller
        should leave such crops in pending rather than write a terminal
        status.
        """

        if not crops:
            return {}

        per_call = images_per_call if images_per_call is not None else self.max_images_per_call
        per_call = max(1, min(per_call, self.max_images_per_call))

        chunks = [crops[i : i + per_call] for i in range(0, len(crops), per_call)]
        chunk_results = await asyncio.gather(
            *[
                self._label_combined_chunk(
                    c,
                    class_names=class_names,
                    draw_overlay=draw_overlay,
                )
                for c in chunks
            ],
            return_exceptions=False,
        )
        merged: dict[str, VlmCombinedReply | None] = {}
        for d in chunk_results:
            merged.update(d)
        return merged

    async def _label_combined_chunk(
        self,
        chunk: list[CombinedCrop],
        *,
        class_names: list[str] | None,
        draw_overlay: bool,
    ) -> dict[str, VlmCombinedReply | None]:
        """Run one upstream combined call over a chunk of crops."""

        if not chunk:
            return {}
        # Length-1 chunks reuse the single-image path so we don't pay the
        # numbered-image prompt overhead for what is functionally just
        # ``label_combined``.
        if len(chunk) == 1:
            crop = chunk[0]
            try:
                reply = await self.label_combined(
                    img_id=crop.crop_id,
                    jpeg_bytes=crop.jpeg_bytes,
                    class_names=class_names if crop.classify else None,
                    plate_bbox_norm=crop.plate_bbox_norm,
                    draw_overlay=draw_overlay,
                )
                return {crop.crop_id: reply}
            except CombinedParseFailure:
                return {crop.crop_id: None}

        any_classify = any(c.classify for c in chunk)
        header = self._pack.combined_batch_rules
        if any_classify and class_names:
            catalog = (
                'Class catalog (use ``class_id`` to refer to entries by index): '
                + ', '.join(f'{i}={name}' for i, name in enumerate(class_names))
                + '.\n'
            )
            header = catalog + header

        user_content: list[dict[str, Any]] = [{'type': 'text', 'text': header}]
        for i, crop in enumerate(chunk, start=1):
            bytes_to_send = crop.jpeg_bytes
            overlay_drawn = False
            if draw_overlay and crop.plate_bbox_norm is not None:
                drew = _draw_bbox_overlay(crop.jpeg_bytes, crop.plate_bbox_norm)
                if drew is not None:
                    bytes_to_send = drew
                    overlay_drawn = True

            if crop.classify:
                directive_class = 'classify the item (choose one ``class_id`` from the catalog)'
            else:
                directive_class = 'skip classification (set ``class_id``=null)'

            if crop.plate_bbox_norm is not None and overlay_drawn:
                directive_region = 'a candidate region bbox is drawn on the crop'
            elif crop.plate_bbox_norm is not None:
                directive_region = (
                    f'the proposed region bbox (normalized) is {list(crop.plate_bbox_norm)}'
                )
            else:
                directive_region = 'no region-bbox candidate was provided'

            directive = f'Image {i}: {directive_class}. {directive_region}.'

            b64 = _b64_jpeg(bytes_to_send)
            user_content.append({'type': 'text', 'text': directive})
            user_content.append(
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
                }
            )

        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self._pack.combined_batch_system},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': 0.0,
            # Combined prompt is bigger than batched verify (per-image
            # directive + class catalog + 8-field JSON schema); some
            # VLMs also emit 1000-1500 tokens of chain-of-thought
            # preamble on multi-image prompts before the JSON. 6144
            # leaves comfortable headroom for a 6-image batch without
            # truncating the closing bracket.
            'max_tokens': 6144,
            # Grammar-constrain the output to a valid JSON object when
            # supported by the upstream server. Without it, a reasoning
            # model can consume the entire turn in the reasoning
            # channel and emit zero visible content. vLLM's json_object
            # grammar only covers objects (not bare arrays), which is
            # why ``combined_batch_system`` asks for
            # ``{"results": [...]}`` rather than a top-level array.
            'response_format': {'type': 'json_object'},
        }

        try:
            response = await self._post_chat(payload)
        except Exception as exc:
            logger.error(
                'vlm_labeler.label_combined_chunk_failed',
                chunk_size=len(chunk),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            # Whole-chunk HTTP failure → per-crop None so the caller can
            # leave each crop in pending for a future retry.
            return {c.crop_id: None for c in chunk}

        raw = _strip_markdown_fences(extract_message_content(response))
        finish_reason = ''
        with contextlib.suppress(KeyError, TypeError, IndexError):
            finish_reason = str(response['choices'][0].get('finish_reason') or '')
        if not raw or finish_reason == 'length':
            # Diagnostic breadcrumb when the VLM either returned empty
            # or was truncated by max_tokens.
            logger.warning(
                'vlm_labeler.combined_batch_response_truncated_or_empty',
                chunk_size=len(chunk),
                finish_reason=finish_reason,
                raw_len=len(raw),
                raw_preview=raw[:300],
            )
        return self._parse_combined_batch_response(
            raw, chunk, self._fields, class_names=class_names
        )

    @staticmethod
    def _parse_combined_batch_response(
        raw: str,
        chunk: list[CombinedCrop],
        fields: RegionFields,
        *,
        class_names: list[str] | None = None,
    ) -> dict[str, VlmCombinedReply | None]:
        """Parse a batched combined response into ``{crop_id: reply | None}``.

        Mirrors the tolerant pattern of :py:meth:`_parse_plate_batch_response`:
        accepts a bare JSON array OR an array embedded in reasoning prose,
        unwraps ``{"results": [...]}`` envelopes, and tolerates positional
        entries that omit the ``img`` index. Per-crop parse failures (or
        an entirely empty response) return ``None`` for the affected crops
        so the caller can leave them in pending instead of stamping a
        terminal status — same policy as ``label_combined`` raising
        :class:`CombinedParseFailure`.
        """

        if not raw:
            logger.warning('vlm_labeler.combined_batch_parse_empty', chunk_size=len(chunk))
            return {c.crop_id: None for c in chunk}

        candidates: list[str] = [raw]
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == '[':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ']' and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(raw[start : i + 1])
                    start = -1

        parsed: Any = None
        for c in candidates:
            try:
                parsed = json.loads(c)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                for key in ('results', 'predictions', 'data'):
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
            if isinstance(parsed, list):
                break
            parsed = None

        if not isinstance(parsed, list):
            logger.warning(
                'vlm_labeler.combined_batch_parse_failed',
                chunk_size=len(chunk),
                raw_len=len(raw),
                raw_preview=raw[:400],
            )
            return {c.crop_id: None for c in chunk}

        aligned = _align_batch_entries(parsed, len(chunk))
        if aligned is None:
            logger.warning(
                'vlm_labeler.combined_batch_misaligned',
                chunk_size=len(chunk),
                n_entries=len(parsed),
                raw_preview=raw[:400],
            )
            return {c.crop_id: None for c in chunk}

        out: dict[str, VlmCombinedReply | None] = {}
        for crop, entry in zip(chunk, aligned, strict=True):
            if entry is None:
                out[crop.crop_id] = None
                continue
            try:
                out[crop.crop_id] = _combined_reply_from_entry(
                    entry,
                    img_id=crop.crop_id,
                    fields=fields,
                    class_names=class_names if crop.classify else None,
                )
            except (TypeError, ValueError) as exc:
                logger.warning(
                    'vlm_labeler.combined_batch_entry_invalid',
                    crop_id=crop.crop_id,
                    error=str(exc),
                    entry_preview=json.dumps(entry, default=str)[:300],
                )
                out[crop.crop_id] = None
        return out

    @staticmethod
    def _parse_plate_batch_response(raw: str, chunk: list[RegionCrop]) -> list[VlmRegionVerdict]:
        """Parse a batched verify response into one verdict per chunk crop.

        Tolerates the same VLM quirks as :py:meth:`_parse_plate_response`:
        leading reasoning prose, ``{"results":[...]}`` envelopes, and
        1-based ``img`` indices.
        """

        fallback = [
            VlmRegionVerdict(
                crop_id=c.crop_id,
                is_region=False,
                confidence='low',
                reason='parse_failure',
            )
            for c in chunk
        ]

        if not raw:
            logger.warning('vlm_labeler.plate_batch_parse_empty', chunk_size=len(chunk))
            return fallback

        # Try the bare reply first; fall back to scanning for the first
        # balanced JSON array embedded in any preamble.
        candidates: list[str] = [raw]
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == '[':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ']' and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(raw[start : i + 1])
                    start = -1

        parsed: Any = None
        for c in candidates:
            try:
                parsed = json.loads(c)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                for key in ('results', 'predictions', 'data'):
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
            if isinstance(parsed, list):
                break
            parsed = None

        if not isinstance(parsed, list):
            logger.warning(
                'vlm_labeler.plate_batch_parse_failed',
                chunk_size=len(chunk),
                raw_preview=raw[:200],
            )
            return fallback

        aligned = _align_batch_entries(parsed, len(chunk))
        if aligned is None:
            logger.warning(
                'vlm_labeler.region_batch_misaligned',
                chunk_size=len(chunk),
                n_entries=len(parsed),
                raw_preview=raw[:200],
            )
            return fallback

        out: list[VlmRegionVerdict] = []
        for crop, entry in zip(chunk, aligned, strict=True):
            if entry is None:
                out.append(
                    VlmRegionVerdict(
                        crop_id=crop.crop_id,
                        is_region=False,
                        confidence='low',
                        reason='missing_in_response',
                    )
                )
                continue
            is_region = _coerce_bool(entry.get('is_region')) is True
            confidence = _normalize_confidence(entry.get('confidence'))
            reason = str(entry.get('reason', '') or '')[:120]
            text, text_confidence = _extract_region_text(entry, is_region=is_region)
            out.append(
                VlmRegionVerdict(
                    crop_id=crop.crop_id,
                    is_region=is_region,
                    confidence=confidence,
                    reason=reason,
                    text=text,
                    text_confidence=text_confidence,
                )
            )
        return out

    async def plate_visible_batch(
        self,
        crops: list[RegionCrop],
        *,
        images_per_call: int | None = None,
    ) -> dict[str, bool]:
        """Pre-filter crops by asking the VLM whether a sub-region is visible.

        Why this exists
        ---------------
        A full region-of-interest detector (e.g. an interactive
        segmentation model) is often the throughput bottleneck. A
        non-trivial fraction of item crops have no visible sub-region
        at all. Asking a one-bit yes/no question up front — packed
        several crops per call — is far cheaper than letting those
        crops walk the full detect → verify pipeline only to be
        discarded downstream.

        Returns ``{crop_id: bool}`` — ``True`` means the sub-region
        appears visible (continue to the detector), ``False`` means
        skip the detector and write a terminal "not visible" status
        directly. On any parse / RPC failure for a crop the verdict
        defaults to ``True`` so we never silently drop a crop that
        might have a real sub-region — the existing detector path
        remains the safety net.
        """

        if not crops:
            return {}

        per_call = images_per_call if images_per_call is not None else self.max_images_per_call
        per_call = max(1, min(per_call, self.max_images_per_call))

        chunks = [crops[i : i + per_call] for i in range(0, len(crops), per_call)]
        chunk_results = await asyncio.gather(
            *[self._plate_visible_chunk(c) for c in chunks],
            return_exceptions=False,
        )
        merged: dict[str, bool] = {}
        for d in chunk_results:
            merged.update(d)
        return merged

    async def _plate_visible_chunk(self, chunk: list[RegionCrop]) -> dict[str, bool]:
        """Run one yes/no upstream call over up to ``max_images_per_call`` crops."""

        if not chunk:
            return {}

        user_content: list[dict[str, Any]] = [
            {'type': 'text', 'text': self._pack.region_visible_user}
        ]
        for crop in chunk:
            b64 = _b64_jpeg(crop.jpeg_bytes)
            user_content.append(
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
                }
            )

        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self._pack.region_visible_system},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': 0.0,
            # Reply is a tight array of ~12 tokens per image, but a
            # reasoning model can emit a 200-400 token chain-of-thought
            # preamble on multi-image prompts. 1024 leaves comfortable
            # headroom; the actual final JSON is still ~12 tokens per
            # image so the wire cost is bounded by what comes after the
            # reasoning section.
            'max_tokens': 1024,
            # Grammar-constrain to a JSON object for the same reason as
            # ``label_combined_batch``; the prompt asks for
            # ``{"results": [...]}`` since the json_object grammar only
            # covers top-level objects.
            'response_format': {'type': 'json_object'},
        }

        try:
            response = await self._post_chat(payload)
        except Exception as exc:
            logger.error(
                'vlm_labeler.plate_visible_chunk_failed',
                chunk_size=len(chunk),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            # Fail-open: assume the sub-region is visible so the
            # detector still gets a crack at it. Costs a wasted
            # detector call we could have skipped but never silently
            # drops a real region.
            return {c.crop_id: True for c in chunk}

        raw = _strip_markdown_fences(extract_message_content(response))
        return self._parse_plate_visible_response(raw, chunk, self._fields)

    @staticmethod
    def _parse_plate_visible_response(
        raw: str,
        chunk: list[RegionCrop],
        fields: RegionFields,
    ) -> dict[str, bool]:
        """Parse the visibility batch reply into ``{crop_id: bool}``.

        Fail-closed on empty response: an empty raw from the upstream
        VLM almost always means the slot timed out or the request
        aborted under heavy concurrent load. Marking those crops as
        ``visible=False`` short-circuits the detect + verify cascade
        instead of forcing them all through the detector, which
        collapses the effective skip rate to near-zero under load —
        the single biggest throughput regression observed when this
        was fail-open at the response level. False negatives are
        recoverable: an operator can re-queue a crop from a review
        queue.

        Per-entry parse failures (missing img index, unparseable
        verdict) remain fail-open because at that point we have
        evidence the VLM did respond — the response was just
        malformed for this image. Fail-open keeps recall intact for
        genuine model-confused cases.
        """

        if not raw:
            logger.warning('vlm_labeler.plate_visible_parse_empty', chunk_size=len(chunk))
            # Fail-closed on empty VLM response.
            return dict.fromkeys((c.crop_id for c in chunk), False)

        # Default fail-open per-crop verdict (VLM responded but maybe
        # garbled a few entries — keep recall on those).
        out: dict[str, bool] = {c.crop_id: True for c in chunk}

        candidates: list[str] = [raw]
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == '[':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ']' and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(raw[start : i + 1])
                    start = -1

        parsed: Any = None
        for c in candidates:
            try:
                parsed = json.loads(c)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                for key in ('results', 'predictions', 'data'):
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
            if isinstance(parsed, list):
                break
            parsed = None

        if not isinstance(parsed, list):
            logger.warning(
                'vlm_labeler.plate_visible_parse_failed',
                chunk_size=len(chunk),
                raw_preview=raw[:200],
            )
            return out

        aligned = _align_batch_entries(parsed, len(chunk))
        if aligned is None:
            # Can't tell which verdict is whose: keep the fail-open default.
            logger.warning(
                'vlm_labeler.region_visible_misaligned',
                chunk_size=len(chunk),
                n_entries=len(parsed),
            )
            return out

        for crop, entry in zip(chunk, aligned, strict=True):
            if entry is None:
                # Fail-open: leave the default True verdict in place.
                continue
            visible_raw = entry.get('visible')
            if visible_raw is None:
                # Tolerate alternate keys callers might emit.
                visible_raw = entry.get('is_region') or entry.get(fields.visible)
            if isinstance(visible_raw, bool):
                out[crop.crop_id] = visible_raw
            elif isinstance(visible_raw, str):
                v = visible_raw.strip().lower()
                if v in ('true', 'yes', 'y', '1', 'visible'):
                    out[crop.crop_id] = True
                elif v in ('false', 'no', 'n', '0', 'not_visible', 'hidden'):
                    out[crop.crop_id] = False
                # else leave as default True (fail-open).
            # Non-bool, non-str → leave as default True.
        return out

    @staticmethod
    def _parse_plate_response(raw: str, crop: RegionCrop) -> VlmRegionVerdict:
        """Parse a single-region verdict, falling back to low-confidence false.

        A VLM sometimes ignores the ``no prose`` instruction and emits a
        chain-of-thought before the JSON. We try the fence-stripped raw
        first, then fall back to extracting the first balanced ``{...}``
        anywhere in the response so reasoning prefixes don't trash the
        verification.
        """

        fallback = VlmRegionVerdict(
            crop_id=crop.crop_id, is_region=False, confidence='low', reason='parse_failure'
        )
        if not raw:
            return fallback
        candidates: list[str] = [_strip_markdown_fences(raw)]
        # Scan for the first balanced JSON object in the raw text. A
        # reasoning prefix often quotes the answer template back
        # before producing the real answer; the first balanced object
        # found this way is the actual reply.
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}' and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(raw[start : i + 1])
                    start = -1
        parsed: Any = None
        for c in candidates:
            try:
                parsed = json.loads(c)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and 'is_region' in parsed:
                break
            parsed = None
        if parsed is None:
            logger.warning(
                'vlm_labeler.plate_parse_failed',
                crop_id=crop.crop_id,
                raw_preview=raw[:200],
            )
            return fallback
        if not isinstance(parsed, dict):
            return fallback

        is_region = _coerce_bool(parsed.get('is_region')) is True

        confidence = _normalize_confidence(parsed.get('confidence'))
        reason = str(parsed.get('reason', '') or '')[:120]
        text, text_confidence = _extract_region_text(parsed, is_region=is_region)
        return VlmRegionVerdict(
            crop_id=crop.crop_id,
            is_region=is_region,
            confidence=confidence,
            reason=reason,
            text=text,
            text_confidence=text_confidence,
        )


# The VLM sometimes emits an empty string or "unknown" instead of true
# null when it can't read the text. Treat those as text=None so
# downstream storage stays clean. Strip whitespace and reject obvious
# sentinels.
_TEXT_SENTINELS = frozenset({'', 'null', 'none', 'unknown', 'unreadable', 'n/a', '-'})


def _extract_region_text(
    parsed: dict[str, Any], *, is_region: bool
) -> tuple[str | None, ConfidenceLevel | None]:
    """Pull region text + confidence from a parsed verdict dict.

    Only honored when ``is_region=True``. Cleans empty/sentinel strings
    to None so callers can treat them uniformly.
    """
    if not is_region:
        return None, None
    raw = parsed.get('text')
    if raw is None:
        return None, None
    text = str(raw).strip()
    if not text or text.lower() in _TEXT_SENTINELS:
        return None, None
    # Cap to 32 chars — longest real-world label text is well under.
    text = text[:32]
    text_conf_raw = parsed.get('text_confidence')
    text_conf = _normalize_confidence(text_conf_raw) if text_conf_raw is not None else 'medium'
    return text, text_conf


__all__ = [
    'DEFAULT_API_KEY',
    'DEFAULT_BASE_URL',
    'DEFAULT_MAX_IMAGES_PER_CALL',
    'DEFAULT_MODEL',
    'DEFAULT_REQUESTS_PER_SECOND',
    'CombinedCrop',
    'CombinedParseFailure',
    'ItemCrop',
    'RegionCrop',
    'VlmClassPrediction',
    'VlmCombinedReply',
    'VlmHealth',
    'VlmLabeler',
    'VlmRegionVerdict',
    'format_class_catalog',
    'resolve_class_name',
]
