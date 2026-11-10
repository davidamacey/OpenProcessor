"""Match keys for a box's text: the compact form and the confusable-folded form.

``text_norm`` is the reading with every character that is not a letter or a digit removed, accents
folded and upper-cased -- the same chain as the ``op_text_compact`` normalizer of the items index
(``src/clients/curation_opensearch/text_analysis.py``): ``pattern_replace`` char filter, then the
``asciifolding`` and ``uppercase`` token filters. ``text_fold`` is ``text_norm`` after the
profile's confusable map (O to 0 and the like). The map is per profile and the index analysis is
fixed per index, so it is applied here, never in the index. Both are storage-only match keys.

OpenSearch normalizers: https://docs.opensearch.org/latest/analyzers/normalizers/
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

# The char filter's pattern: everything that is not a letter or a decimal digit.
NON_ALNUM_PATTERN = '[^\\p{L}\\p{Nd}]'


def _ascii_fold(ch: str) -> str:
    """Drop the accents of one precomposed letter (``asciifolding`` for Latin letters)."""
    decomposed = unicodedata.normalize('NFKD', ch)
    return ''.join(c for c in decomposed if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """The ``text_norm`` of a reading."""
    kept = (ch for ch in text if ch.isalpha() or ch.isdecimal())
    return ''.join(_ascii_fold(ch) for ch in kept).upper()


def fold(text: str, confusables: Mapping[str, str]) -> str:
    """The ``text_fold`` of a reading: ``normalize`` then the profile's confusable map.

    The map keys and values are upper-case characters; it is validated idempotent
    (``confusables_issue``), so folding twice changes nothing.
    """
    return ''.join(confusables.get(ch, ch) for ch in normalize(text))


def confusables_issue(confusables: Mapping[str, str], charset: str) -> str | None:
    """Why a confusable map is not valid, or ``None``.

    Every key and value is one character, no value is also a key (so the fold is idempotent), and
    when the profile has a ``charset`` regex every value matches it (closed over the charset).
    """
    if not confusables:
        return None
    try:
        allowed = re.compile(charset) if charset else None
    except re.error:
        # An invalid charset is reported by its own check.
        allowed = None
    for key, value in confusables.items():
        if len(key) != 1 or len(value) != 1:
            return f'{key!r}: {value!r} must map one character to one character'
        if value in confusables:
            return (
                f'{key!r} -> {value!r}: {value!r} is also a key, the fold would not be idempotent'
            )
        if allowed is not None and not allowed.fullmatch(value):
            return f'{key!r} -> {value!r}: the target is outside text_charset'
    return None
