"""Searchable per-item text: every OCR line read on an item crop.

The region worker already runs the OCR pipeline over the item crop for
its text-hint step; the same lines are stored on the item as
``item_text_lines`` (for display) plus ``item_text_tokens`` (normalized
keyword tokens) so ``GET {prefix}/crops?item_text=<q>`` can find items
by any text printed on them. Matching is backend-owned: the query is
normalized here with the same rule the tokens were written with.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.services.detection.region_text import OcrLine


ITEM_TEXT_LINES_FIELD = 'item_text_lines'
ITEM_TEXT_TOKENS_FIELD = 'item_text_tokens'

# Explicit mapping. ``item_text_lines`` is a plain object array (not
# ``nested``): nothing queries inside a line -- search goes through the
# flat token field -- so there is no reason to pay for one hidden nested
# document per line.
ITEM_TEXT_MAPPING: dict[str, Any] = {
    ITEM_TEXT_LINES_FIELD: {
        'type': 'object',
        'properties': {
            'text': {'type': 'keyword'},
            'box_norm': {'type': 'float'},
            'confidence': {'type': 'float'},
            'rel_height': {'type': 'float'},
        },
    },
    ITEM_TEXT_TOKENS_FIELD: {'type': 'keyword'},
}

# Words: runs of letters/digits in any script (``_`` excluded).
_WORD = re.compile(r'[^\W_]+')


def normalize_tokens(text: str) -> list[str]:
    """Uppercased letter/digit runs of ``text``, in order."""
    return [w.upper() for w in _WORD.findall(text)]


def item_text_lines(lines: Iterable[OcrLine], *, min_confidence: float) -> list[dict[str, Any]]:
    """Wire/storage shape of the item-crop OCR lines kept for search.

    Lines below ``min_confidence`` (recognition score) or with no letter
    or digit are dropped. Order is the OCR pipeline's reading order.
    """
    out: list[dict[str, Any]] = []
    for ln in lines:
        text = ln.text.strip()
        if not text or ln.score < min_confidence or not normalize_tokens(text):
            continue
        out.append(
            {
                'text': text,
                'box_norm': [round(float(v), 5) for v in ln.box],
                'confidence': round(float(ln.score), 4),
                'rel_height': round(float(ln.height), 5),
            }
        )
    return out


def item_text_tokens(lines: Iterable[dict[str, Any]]) -> list[str]:
    """Sorted, de-duplicated search tokens for stored ``item_text_lines``.

    Each word of each line, plus the line with separators removed when it
    has more than one word (so ``ABC-123`` is findable as ``ABC123`` too).
    """
    tokens: set[str] = set()
    for ln in lines:
        words = normalize_tokens(str(ln.get('text') or ''))
        tokens.update(words)
        if len(words) > 1:
            tokens.add(''.join(words))
    return sorted(tokens)


def item_text_update(lines: Iterable[OcrLine], *, min_confidence: float) -> dict[str, Any]:
    """The item-text fields to write for one item crop's OCR lines."""
    stored = item_text_lines(lines, min_confidence=min_confidence)
    return {ITEM_TEXT_LINES_FIELD: stored, ITEM_TEXT_TOKENS_FIELD: item_text_tokens(stored)}


def item_text_query(q: str) -> dict[str, Any] | None:
    """Filter clause for ``?item_text=<q>``: every word of ``q`` must be a
    case-insensitive prefix of some stored token. ``None`` when ``q`` has
    no letter or digit."""
    words = normalize_tokens(q)
    if not words:
        return None
    return {
        'bool': {
            'filter': [{'prefix': {ITEM_TEXT_TOKENS_FIELD: {'value': w}}} for w in words],
        }
    }


def item_text_lines_to_wire(value: Any) -> list[dict[str, Any]]:
    """Stored ``item_text_lines`` -> wire list (``[]`` when absent/malformed)."""
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for ln in value:
        if not isinstance(ln, dict):
            continue
        out.append(
            {
                'text': ln.get('text'),
                'box_norm': ln.get('box_norm'),
                'confidence': ln.get('confidence'),
                'rel_height': ln.get('rel_height'),
            }
        )
    return out


__all__ = [
    'ITEM_TEXT_LINES_FIELD',
    'ITEM_TEXT_MAPPING',
    'ITEM_TEXT_TOKENS_FIELD',
    'item_text_lines',
    'item_text_lines_to_wire',
    'item_text_query',
    'item_text_tokens',
    'item_text_update',
    'normalize_tokens',
]
