"""Pure reply-parsing helpers of the VLM labeler (no I/O, no labeler state).

Split out of ``vlm_labeler.py``.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from src.services.labeling.region_overlay import box_verdicts
from src.services.labeling.vlm_models import (
    ConfidenceLevel,
    ItemCrop,
    VlmClassPrediction,
    VlmCombinedReply,
)


if TYPE_CHECKING:
    from src.config import RegionFields


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


def _unwrap_nested_combined_entry(entry: dict[str, Any], fields: RegionFields) -> dict[str, Any]:
    """Unwrap a combined entry the VLM nested one level down under an invented key.

    Live evidence: a reasoning model sometimes wraps the whole per-image
    answer object under a made-up key instead of the flat shape the
    prompt asks for, e.g.::

        {"img": 2, "layout_analysis": {"region_visible": true, ...}}

    Reading ``fields.visible`` straight off ``entry`` then finds nothing
    and the caller's existing no-verdict handling fires even though the
    VLM did answer -- it just filed the answer under the wrong key. Only
    unwrap when the fix is unambiguous: ``entry`` lacks the top-level
    answer field AND has exactly one dict-valued key (other than
    ``img``) that itself carries that field. Zero or multiple such
    candidates leaves ``entry`` untouched so the existing no-verdict
    path applies rather than guessing.
    """
    if fields.visible in entry:
        return entry
    candidates = [
        value
        for key, value in entry.items()
        if key != 'img' and isinstance(value, dict) and fields.visible in value
    ]
    if len(candidates) != 1:
        return entry
    unwrapped = dict(candidates[0])
    if 'img' in entry and 'img' not in unwrapped:
        unwrapped['img'] = entry['img']
    return unwrapped


def _combined_reply_from_entry(
    entry: dict[str, Any],
    *,
    img_id: str,
    fields: RegionFields,
    class_names: list[str] | None,
    n_boxes: int,
) -> VlmCombinedReply:
    """Build a :class:`VlmCombinedReply` from one parsed reply object.

    Fail-closed: raises ``ValueError`` when the region-visible answer is
    missing or not a recognizable boolean (the caller leaves the item
    pending rather than stamping ``no_region_visible`` or an accept off a
    reply that never answered). ``n_boxes`` candidate boxes were offered
    for this crop (W8.6 numbered overlay); when ``n_boxes > 0`` the reply
    must carry the list-shaped ``fields.boxes`` key or
    :class:`~src.services.labeling.region_overlay.MultiRegionKeysMissingError`
    (a :class:`ValueError` subclass) is raised -- D-B, no flat-shape
    fallback. ``n_boxes == 0`` (no candidate offered) skips box-verdict
    parsing entirely: ``region_boxes`` is ``[]``.

    Unwraps an unambiguous single-key nesting first (see
    :func:`_unwrap_nested_combined_entry`) before reading any field.
    """
    entry = _unwrap_nested_combined_entry(entry, fields)
    visible = _coerce_bool(entry.get(fields.visible))
    if visible is None:
        msg = f'{fields.visible} missing or not a boolean: {entry.get(fields.visible)!r}'
        raise ValueError(msg)
    class_id, class_raw = _coerce_class_answer(entry, class_names)
    class_conf_raw = entry.get('class_confidence')
    make = str(entry.get('make') or '').strip()[:48]
    model_name = str(entry.get('model') or '').strip()[:48]
    # M5 fix (W8 pipeline-wiring review, 2026-09-27): suppress a per-box
    # text reply that's really the item's own class/make/model echoed
    # back into the text slot -- ported from the pre-W8 flat parser's
    # ``_clean_combined_region_text`` echo check, dropped (not ported)
    # when the per-box list shape replaced it.
    picked_class = (
        class_names[class_id]
        if class_names and class_id is not None and 0 <= class_id < len(class_names)
        else ''
    )
    echoes = (picked_class, make, model_name, f'{make} {model_name}'.strip())
    region_boxes = box_verdicts(entry, n_boxes, fields, echoes=echoes) if n_boxes > 0 else []
    return VlmCombinedReply(
        img_id=img_id,
        class_id=class_id,
        class_confidence=(
            _normalize_confidence(class_conf_raw) if class_conf_raw is not None else None
        ),
        region_visible=visible,
        region_boxes=region_boxes,
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
