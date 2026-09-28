"""W8.6: the numbered-overlay VLM contract for multi-box region verify.

Kept separate from :mod:`src.services.labeling.vlm_labeler` (a ratchet-
exempt oversize file already) so this module's growth doesn't add to it.
Self-contained: no import of ``vlm_labeler`` (or vice versa this pass --
see the W8 handback report for what remains to wire the two together).

Regions are always a list (W8.0) -- N=1 draws a single numbered tag "1",
not a separate code path. Every box on the crop sent to the VLM is
numbered so the reply can address "box 3" without ambiguity; the reply's
per-box verdicts are threaded back onto the same numbers by
:func:`box_verdicts`.

**D-B (owner decision, 2026-09-26):** the pre-W8 flat reply shape
(``region_bbox_correct`` etc. as top-level keys) is DROPPED, not kept as
a fallback. A combined reply that lacks the list-shaped verdict key
raises :class:`MultiRegionKeysMissingError` -- list shape only, N=1 is a
list of one.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal

from src.config.region_fields import RegionFields, get_region_fields
from src.core.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = get_logger('region_overlay')

BBoxNorm = tuple[float, float, float, float]
ConfidenceLevel = Literal['high', 'medium', 'low']

# Same visual language as the pre-W8 single-box overlay
# (vlm_labeler._draw_bbox_overlay): a 3-px red outline drawn just outside
# the box so it never paints over the region itself.
_OVERLAY_WIDTH = 3
_TAG_COLOR = (255, 0, 0)
_TAG_TEXT_COLOR = (255, 255, 255)
# Font height as a fraction of the image's short side, floored at 12 px.
_TAG_FONT_FRACTION = 0.04
_TAG_FONT_MIN_PX = 12


@dataclass(frozen=True)
class VlmBoxVerdict:
    """One numbered box's verdict from a combined VLM reply (W8.6).

    ``box`` is the 1-based number the overlay drew. ``bbox_correct`` is
    ``None`` for "no verdict" (never answered, or an unparseable answer)
    -- the caller treats that as a no-verdict box, not a rejection.
    """

    box: int
    bbox_correct: bool | None
    confidence: ConfidenceLevel | None
    text_reply: str | None = None


class MultiRegionKeysMissingError(ValueError):
    """A combined VLM reply lacks the list-shaped box-verdict key (D-B).

    ``code`` is the stable machine code the caller's ``api_error`` /
    logging surfaces -- ``pack_multi_region_keys_missing``.
    """

    code = 'pack_multi_region_keys_missing'


def draw_region_overlay(jpeg_bytes: bytes, boxes: Sequence[BBoxNorm]) -> bytes | None:
    """Draw every box in ``boxes`` as a numbered red rectangle (1-based).

    Same rendering for N=1 (draws "1") as for N>1 -- one code path, no
    single-box special case. Returns the re-encoded JPEG, or ``None`` if
    the source isn't a decodable image (caller falls back to coordinates
    in the prompt text, :func:`render_region_block`).
    """
    if not boxes:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont

        with Image.open(BytesIO(jpeg_bytes)) as src:
            im = src.convert('RGB')
            w, h = im.size
            font_px = max(_TAG_FONT_MIN_PX, round(min(w, h) * _TAG_FONT_FRACTION))
            font = ImageFont.load_default(size=font_px)
            draw = ImageDraw.Draw(im)
            for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
                box = (
                    max(0, int(x1 * w) - _OVERLAY_WIDTH),
                    max(0, int(y1 * h) - _OVERLAY_WIDTH),
                    min(w - 1, int(x2 * w) + _OVERLAY_WIDTH),
                    min(h - 1, int(y2 * h) + _OVERLAY_WIDTH),
                )
                draw.rectangle(box, outline=_TAG_COLOR, width=_OVERLAY_WIDTH)
                _draw_number_tag(draw, str(i), anchor=(box[0], box[1]), font=font, bounds=(w, h))
            out = BytesIO()
            im.save(out, format='JPEG', quality=90)
            return out.getvalue()
    except Exception:
        return None


def _draw_number_tag(
    draw: Any,
    text: str,
    *,
    anchor: tuple[int, int],
    font: Any,
    bounds: tuple[int, int],
) -> None:
    """Draw ``text`` in a filled tag whose top-left sits at ``anchor``,
    clamped inside ``bounds`` (the image size)."""
    w, h = bounds
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(2, round(text_h * 0.25))
    tag_w, tag_h = text_w + 2 * pad, text_h + 2 * pad
    x1 = min(max(0, anchor[0]), max(0, w - tag_w))
    y1 = min(max(0, anchor[1] - tag_h), max(0, h - tag_h))
    x2, y2 = x1 + tag_w, y1 + tag_h
    draw.rectangle((x1, y1, x2, y2), fill=_TAG_COLOR)
    draw.text((x1 + pad - bbox[0], y1 + pad - bbox[1]), text, fill=_TAG_TEXT_COLOR, font=font)


def overlay_description(n: int) -> str:
    """The prompt sentence naming the numbered overlay for ``n`` boxes.

    One template for every N (including 1): "the candidate regions are
    marked by numbered red rectangles (1 to n); the rectangles and
    numbers are not part of the photo".
    """
    return (
        f'the candidate regions are marked by numbered red rectangles (1 to {n}); '
        'the rectangles and numbers are not part of the photo'
    )


def render_region_block(boxes: Sequence[BBoxNorm], *, overlay_drawn: bool) -> str:
    """The ``{region_block}`` prompt placeholder for ``boxes``.

    - no boxes -> a "no candidate" sentence.
    - overlay drawn -> refer to the numbered overlay.
    - overlay failed (:func:`draw_region_overlay` returned ``None``) ->
      fall back to coordinates in the prompt text, still per-number.
    """
    if not boxes:
        return 'No region-bbox candidate was provided. '
    if overlay_drawn:
        return f'In this image {overlay_description(len(boxes))}. Answer for each numbered box. '
    coords = ', '.join(
        f'{i}=[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]'
        for i, (x1, y1, x2, y2) in enumerate(boxes, start=1)
    )
    return f'The proposed region boxes (normalized x1,y1,x2,y2) are: {coords}. Answer for each numbered box. '


_TRUE_STRINGS = frozenset({'true', 'yes', 'y', '1'})
_FALSE_STRINGS = frozenset({'false', 'no', 'n', '0'})


def _coerce_bool(value: Any) -> bool | None:
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


def _normalize_confidence(value: Any) -> ConfidenceLevel | None:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ('high', 'medium', 'low'):
            return v  # type: ignore[return-value]
    return None


# M5 fix (W8 pipeline-wiring review, 2026-09-27): the pre-W8 flat parser's
# full sentinel set (vlm_labeler._TEXT_SENTINELS) -- the W8 rewrite's
# ``_clean_text_reply`` only dropped 4 of these 7, silently storing
# "unreadable"/"-" as if they were real region text.
_TEXT_SENTINELS = frozenset({'', 'null', 'none', 'unknown', 'unreadable', 'n/a', '-'})


def _echo_key(value: str) -> str:
    """Case- and separator-insensitive form ("Adventure Bike" == "adventurebike")."""
    return ''.join(ch for ch in value.casefold() if ch.isalnum())


def _clean_text_reply(value: Any, *, echoes: tuple[str, ...] = ()) -> str | None:
    """The region's transcribed text from a per-box reply, or ``None``.

    Drops sentinels and any value that is really one of the reply's own
    item-level answers echoed into the text slot (``echoes``: the class
    name it picked, its ``make`` / ``model``, both joined) -- M5: this
    suppression lived in the pre-W8 flat parser
    (``vlm_labeler._clean_combined_region_text``) and was dropped, not
    ported, when the per-box list shape replaced it.
    """
    if value is None or isinstance(value, bool | dict | list):
        return None
    text = str(value).strip()
    if not text or text.lower() in _TEXT_SENTINELS:
        return None
    if _echo_key(text) in {_echo_key(e) for e in echoes if e}:
        return None
    return text[:32]


_BOX_NUMBER_PREFIX = 'box'


def _coerce_box_number(value: Any) -> int | None:
    """Accept ``1``, ``"1"``, or ``"box 1"`` -- reject everything else."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v.startswith(_BOX_NUMBER_PREFIX):
            v = v[len(_BOX_NUMBER_PREFIX) :].strip()
        try:
            return int(v)
        except ValueError:
            return None
    return None


def box_verdicts(
    entry: dict[str, Any],
    n: int,
    fields: RegionFields | None = None,
    *,
    echoes: tuple[str, ...] = (),
) -> list[VlmBoxVerdict]:
    """The reply's per-box verdicts, always length ``n`` (W8.6).

    ``entry[fields.boxes]`` must be a list (D-B: no flat-shape fallback)
    -- each element's ``box`` (1-based; accepts ``1``, ``"1"``, ``"box
    1"``), ``fields.bbox_correct``, ``fields.confidence`` and
    ``fields.text`` are read. An out-of-range or duplicate ``box`` is
    ignored (first wins), logged ``vlm_box_index_invalid``. A number with
    no matching element gets ``bbox_correct=None`` (no verdict).

    ``echoes`` (M5): the reply's own item-level answers (the class name
    it picked, ``make``, ``model``, both joined) -- a per-box ``text``
    reply that's really one of those echoed back is dropped, never
    stored as if it were text read off the region.

    Raises :class:`MultiRegionKeysMissingError` when ``entry`` doesn't
    carry the list key at all.
    """
    fields = fields or get_region_fields()
    raw = entry.get(fields.boxes)
    if not isinstance(raw, list):
        msg = f'{fields.boxes!r} missing or not a list in the combined reply'
        raise MultiRegionKeysMissingError(msg)

    by_number: dict[int, dict[str, Any]] = {}
    for element in raw:
        if not isinstance(element, dict):
            continue
        num = _coerce_box_number(element.get('box'))
        if num is None or not (1 <= num <= n) or num in by_number:
            logger.warning('vlm_box_index_invalid', box=element.get('box'), n=n)
            continue
        by_number[num] = element

    verdicts: list[VlmBoxVerdict] = []
    for i in range(1, n + 1):
        element = by_number.get(i)
        if element is None:
            verdicts.append(
                VlmBoxVerdict(box=i, bbox_correct=None, confidence=None, text_reply=None)
            )
            continue
        verdicts.append(
            VlmBoxVerdict(
                box=i,
                bbox_correct=_coerce_bool(element.get(fields.bbox_correct)),
                confidence=_normalize_confidence(element.get(fields.confidence)),
                text_reply=_clean_text_reply(element.get(fields.text), echoes=echoes),
            )
        )
    return verdicts


__all__ = [
    'MultiRegionKeysMissingError',
    'VlmBoxVerdict',
    'box_verdicts',
    'draw_region_overlay',
    'overlay_description',
    'render_region_block',
]
