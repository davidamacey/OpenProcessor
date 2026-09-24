"""Is a region-text reading real text? Validity rules applied before one
reading is chosen.

A reader can answer with something that is not a reading at all: the VLM
echoes the example value its prompt showed ("ABC1234", or a truncation of
it), answers a "can't read it" word ("NOT_READABLE", "N/A"), or a stock
run ("999", "123456"). Stored as the region's text, those poison search,
text-training data and dedup. :class:`RegionTextRules` rejects them so the
chooser (:func:`src.services.detection.region_text.resolve_region_text`)
treats them as no reading and can fall back to the other reader.

Every rule is data: the region profile's normalization and length bounds
(the same ones the OCR reader applies), an optional full-match format
regex, configured placeholder readings, the active prompt pack's quoted
example values, and whether stock runs are rejected. The only built-in
vocabulary is :data:`NO_READING_WORDS` -- generic "no answer" words, not
any domain's text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from src.services.detection.region_text import TextNormalizer


if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.config import DetectionProfile


# Why a reading is not text (``RegionFields.text_vlm_invalid``).
INVALID_NO_READING = 'no_reading'
INVALID_PLACEHOLDER = 'placeholder'
INVALID_SEQUENCE = 'sequence'
INVALID_CHARSET = 'charset'
INVALID_TOO_SHORT = 'too_short'
INVALID_TOO_LONG = 'too_long'
INVALID_FORMAT = 'format'
INVALID_REASONS: tuple[str, ...] = (
    INVALID_NO_READING,
    INVALID_PLACEHOLDER,
    INVALID_SEQUENCE,
    INVALID_CHARSET,
    INVALID_TOO_SHORT,
    INVALID_TOO_LONG,
    INVALID_FORMAT,
)

# Generic "there is no reading" answers, compared as uppercase letters and
# digits only ("not_readable" == "NOT READABLE" == "NOTREADABLE").
NO_READING_WORDS: frozenset[str] = frozenset(
    {
        'NULL',
        'NONE',
        'NIL',
        'NA',
        'NAN',
        'UNKNOWN',
        'UNREADABLE',
        'NOTREADABLE',
        'ILLEGIBLE',
        'NOTLEGIBLE',
        'NOTVISIBLE',
        'NOTAPPLICABLE',
        'NOTAVAILABLE',
        'NOTEXT',
        'BLANK',
        'EMPTY',
        'OBSCURED',
        'REDACTED',
    }
)

# A truncated prompt example ("ABC" / "ABC123" from "ABC1234") is still the
# example; shorter prefixes are too likely to be real text.
_PLACEHOLDER_PREFIX_MIN = 3
_SEQUENCE_MIN = 3


def text_key(text: str) -> str:
    """Uppercase letters and digits only -- the form every rule compares."""
    return ''.join(ch for ch in text.upper() if ch.isalnum())


def _is_run(key: str) -> bool:
    """One repeated character, or one ascending / descending run of
    consecutive digits or letters, at least :data:`_SEQUENCE_MIN` long."""
    if len(key) < _SEQUENCE_MIN or not (key.isdigit() or key.isalpha()):
        return False
    steps = {ord(b) - ord(a) for a, b in pairwise(key)}
    return steps in ({0}, {1}, {-1})


@dataclass(frozen=True)
class RegionTextRules:
    """Validity rules for one region type's readings."""

    normalizer: TextNormalizer = field(default_factory=TextNormalizer)
    len_min: int = 1
    len_max: int = 0  # 0 = unbounded
    format: str = ''
    reject_sequences: bool = False
    placeholders: frozenset[str] = frozenset()  # text_key() forms

    @classmethod
    def from_profile(
        cls, profile: DetectionProfile, *, prompt_examples: Iterable[str] = ()
    ) -> RegionTextRules:
        """Rules from ``profile``'s ``text_*`` fields plus the prompt pack's
        quoted example values (see
        :func:`src.services.labeling.vlm_prompts.prompt_text_examples`)."""
        keys = {text_key(p) for p in (*profile.text_placeholders, *prompt_examples)}
        return cls(
            normalizer=TextNormalizer(
                uppercase=profile.text_uppercase, charset=profile.text_charset
            ),
            len_min=profile.text_len_min,
            len_max=profile.text_len_max,
            format=profile.text_format,
            reject_sequences=profile.text_reject_sequences,
            placeholders=frozenset(k for k in keys if len(k) >= 2),
        )

    def _is_placeholder(self, key: str) -> bool:
        return any(
            key == p or (len(key) >= _PLACEHOLDER_PREFIX_MIN and p.startswith(key))
            for p in self.placeholders
        )

    def invalid_reason(self, text: str | None) -> str | None:
        """Why ``text`` is not a reading (one of :data:`INVALID_REASONS`),
        or ``None`` when it is one. Empty / missing text is no reading."""
        key = text_key(text or '')
        if not key or key in NO_READING_WORDS:
            return INVALID_NO_READING
        if self._is_placeholder(key):
            return INVALID_PLACEHOLDER
        if self.reject_sequences and _is_run(key):
            return INVALID_SEQUENCE
        normalized = self.normalizer.normalize(text or '')
        n = len(normalized.replace(' ', ''))
        if n == 0:
            return INVALID_CHARSET
        if n < max(self.len_min, 1):
            return INVALID_TOO_SHORT
        if self.len_max > 0 and n > self.len_max:
            return INVALID_TOO_LONG
        if self.format and re.fullmatch(self.format, normalized) is None:
            return INVALID_FORMAT
        return None

    def catalog(self) -> dict[str, Any]:
        """The rules as served by ``GET {prefix}/regions/vocabulary``."""
        return {
            'uppercase': self.normalizer.uppercase,
            'charset': self.normalizer.charset,
            'len_min': self.len_min,
            'len_max': self.len_max,
            'format': self.format,
            'reject_sequences': self.reject_sequences,
            'placeholders': sorted(self.placeholders),
            'no_reading_words': sorted(NO_READING_WORDS),
            'invalid_reasons': list(INVALID_REASONS),
        }


def region_text_rules(profile: DetectionProfile, pack: Any | None = None) -> RegionTextRules:
    """The rules for ``profile`` with ``pack``'s quoted examples as
    placeholders (default: the process's resolved prompt pack)."""
    from src.services.labeling.vlm_prompts import prompt_text_examples, resolve_prompt_pack

    return RegionTextRules.from_profile(
        profile, prompt_examples=prompt_text_examples(pack or resolve_prompt_pack())
    )


__all__ = [
    'INVALID_CHARSET',
    'INVALID_FORMAT',
    'INVALID_NO_READING',
    'INVALID_PLACEHOLDER',
    'INVALID_REASONS',
    'INVALID_SEQUENCE',
    'INVALID_TOO_LONG',
    'INVALID_TOO_SHORT',
    'NO_READING_WORDS',
    'RegionTextRules',
    'region_text_rules',
    'text_key',
]
