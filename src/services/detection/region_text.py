"""Read the dominant text of a region crop from per-line OCR output.

A region crop (the region box cut from the item crop with a small
margin) usually carries more text than the one reading an operator
wants: a small header line above the main text, a small slogan below
it, and text on whatever frames the region, near the crop border. The
main text is the *tallest* text on the region, so the reader keeps
lines that are nearly as tall as the tallest one and drops lines whose
center sits in an outer border band of the crop. This is the
height-filter approach commonly used to post-process PaddleOCR output
on cropped regions (keep the tallest line plus any line within a ratio
of its height, order left-to-right, then constrain the character set);
PaddleOCR itself does no such selection -- its pipeline returns every
detected line with a recognition score.

Everything here is pure (no Triton, no OpenSearch) so the selection
rules are unit-testable against synthetic OCR lines. Every threshold
comes from :class:`~src.config.DetectionProfile` (``text_*`` fields,
env ``OP_REGION_DETECTION_TEXT_*``); :meth:`DominantTextConfig.from_profile`
is the only place that reads them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from src.config import DetectionProfile
    from src.services.detection.region_text_rules import RegionTextRules


# Where a region's stored text came from (``RegionFields.text_source``).
TEXT_SOURCE_OCR = 'ocr'
TEXT_SOURCE_VLM = 'vlm'

TEXT_READER_VLM = 'vlm'
TEXT_READER_OCR = 'ocr'
TEXT_READER_VLM_THEN_OCR = 'vlm_then_ocr'
TEXT_READER_BOTH = 'both'
TEXT_READER_MODES = frozenset(
    {TEXT_READER_VLM, TEXT_READER_OCR, TEXT_READER_VLM_THEN_OCR, TEXT_READER_BOTH}
)

# Why the chosen reading won (``RegionFields.text_choice``).
TEXT_CHOICE_AGREE = 'readers_agree'
TEXT_CHOICE_VLM_PREFERRED = 'vlm_preferred'
TEXT_CHOICE_VLM_ONLY = 'vlm_only'
TEXT_CHOICE_OCR_ONLY = 'ocr_only'
TEXT_CHOICE_OCR_MODE = 'ocr_mode'
TEXT_CHOICE_VLM_INVALID = 'vlm_invalid'
TEXT_CHOICE_NONE = 'no_valid_reading'
TEXT_CHOICE_HUMAN = 'human'
TEXT_CHOICES: tuple[str, ...] = (
    TEXT_CHOICE_AGREE,
    TEXT_CHOICE_VLM_PREFERRED,
    TEXT_CHOICE_VLM_ONLY,
    TEXT_CHOICE_OCR_ONLY,
    TEXT_CHOICE_OCR_MODE,
    TEXT_CHOICE_VLM_INVALID,
    TEXT_CHOICE_NONE,
    TEXT_CHOICE_HUMAN,
)

# VLM text confidence arrives as a category; stored as a number so both
# readers share one float field.
VLM_TEXT_CONFIDENCE = {'high': 0.92, 'medium': 0.70, 'low': 0.40}


def validate_text_reader(mode: str) -> str:
    """Return ``mode`` if it is a known text-reader mode, else raise."""
    if mode not in TEXT_READER_MODES:
        msg = f'unknown text_reader {mode!r}; choose from {sorted(TEXT_READER_MODES)}'
        raise ValueError(msg)
    return mode


def ocr_needed(mode: str, *, vlm_text: str | None, vlm_available: bool) -> bool:
    """Whether the region-text OCR reader must run for this region.

    With no VLM configured every mode falls back to OCR -- otherwise a
    deployment without an image LLM would never store region text.
    """
    if not vlm_available:
        return True
    if mode in (TEXT_READER_OCR, TEXT_READER_BOTH):
        return True
    if mode == TEXT_READER_VLM_THEN_OCR:
        return not vlm_text
    return False


@dataclass(frozen=True)
class OcrLine:
    """One recognized text line, box normalized to the crop it was read from."""

    text: str
    box: tuple[float, float, float, float]
    score: float
    det_score: float = 1.0

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) * 0.5, (self.box[1] + self.box[3]) * 0.5)


@dataclass(frozen=True)
class TextNormalizer:
    """Uppercase (optional) and keep only characters matching ``charset``.

    ``charset`` is a regex matching ONE allowed character (e.g.
    ``[A-Z0-9]``); empty keeps every non-whitespace character. Applied
    after uppercasing, so an uppercase-only class still admits lowercase
    OCR output when ``uppercase`` is on.
    """

    uppercase: bool = False
    charset: str = ''

    def normalize(self, text: str) -> str:
        s = text.upper() if self.uppercase else text
        if not self.charset:
            return ''.join(s.split())
        allowed = re.compile(self.charset)
        return ''.join(ch for ch in s if allowed.fullmatch(ch))


@dataclass(frozen=True)
class DominantTextConfig:
    """Selection + normalization rules for one region type."""

    min_height_ratio: float = 0.6
    border_margin: float = 0.08
    normalizer: TextNormalizer = TextNormalizer()
    join: str = ' '
    len_min: int = 1
    len_max: int = 0  # 0 = unbounded
    stopwords: frozenset[str] = frozenset()
    min_confidence: float = 0.0

    @classmethod
    def from_profile(cls, profile: DetectionProfile) -> DominantTextConfig:
        normalizer = TextNormalizer(uppercase=profile.text_uppercase, charset=profile.text_charset)
        return cls(
            min_height_ratio=profile.text_min_height_ratio,
            border_margin=profile.text_border_margin,
            normalizer=normalizer,
            join=profile.text_join,
            len_min=profile.text_len_min,
            len_max=profile.text_len_max,
            # Compared after normalization, so a profile can list them in
            # any case / punctuation.
            stopwords=frozenset(
                n for n in (normalizer.normalize(w) for w in profile.text_stopwords) if n
            ),
            min_confidence=profile.text_min_confidence,
        )


@dataclass(frozen=True)
class DominantTextReading:
    """The reader's verdict for one region crop.

    ``text`` is the filtered, normalized reading (``None`` when nothing
    survived the rules -- ``reason`` says why); ``raw`` is every line the
    OCR produced, in reading order, joined by a space (``''`` when the
    OCR found nothing). ``confidence`` is the minimum recognition score of
    the kept lines (the weakest piece bounds the whole reading),
    ``confidence_mean`` their mean.
    """

    text: str | None
    raw: str
    confidence: float | None
    confidence_mean: float | None
    kept: tuple[OcrLine, ...]
    reason: str


def _valid(lines: Iterable[OcrLine]) -> list[OcrLine]:
    return [ln for ln in lines if ln.text.strip() and ln.width > 0.0 and ln.height > 0.0]


def _same_row(a: OcrLine, b: OcrLine) -> bool:
    """Lines share a row when their vertical overlap covers at least half
    the shorter line."""
    overlap = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
    return overlap >= 0.5 * min(a.height, b.height)


def reading_order(lines: Sequence[OcrLine]) -> list[OcrLine]:
    """Rows top-to-bottom, lines within a row left-to-right.

    Pieces of one text row split by the detector (a gap, a logo between
    them) land in the same row and join left-to-right; genuinely stacked
    rows (two-row text) stay top-to-bottom.
    """
    rows: list[list[OcrLine]] = []
    for ln in sorted(lines, key=lambda x: x.center[1]):
        for row in rows:
            if any(_same_row(ln, other) for other in row):
                row.append(ln)
                break
        else:
            rows.append([ln])
    rows.sort(key=lambda r: sum(x.center[1] for x in r) / len(r))
    return [ln for row in rows for ln in sorted(row, key=lambda x: x.box[0])]


def _in_border(line: OcrLine, margin: float) -> bool:
    if margin <= 0.0:
        return False
    cx, cy = line.center
    return cx < margin or cx > 1.0 - margin or cy < margin or cy > 1.0 - margin


def _strip_stopwords(text: str, cfg: DominantTextConfig) -> str:
    """Normalize ``text`` minus stopwords.

    A whole line matching a stopword is dropped (this is how a multi-word
    stopword like ``NEW YORK`` matches); otherwise single words matching a
    stopword are removed from the line.
    """
    whole = cfg.normalizer.normalize(text)
    if not cfg.stopwords:
        return whole
    if whole in cfg.stopwords:
        return ''
    kept = [w for w in text.split() if cfg.normalizer.normalize(w) not in cfg.stopwords]
    return cfg.normalizer.normalize(' '.join(kept))


def read_dominant_text(lines: Iterable[OcrLine], cfg: DominantTextConfig) -> DominantTextReading:
    """Pick the dominant text from ``lines`` (see module docstring)."""
    valid = _valid(lines)
    raw = ' '.join(ln.text.strip() for ln in reading_order(valid))
    if not valid:
        return DominantTextReading(None, raw, None, None, (), 'no_lines')

    inner = [ln for ln in valid if not _in_border(ln, cfg.border_margin)]
    if not inner:
        return DominantTextReading(None, raw, None, None, (), 'all_border')

    # Normalize before measuring height so a line that is all punctuation
    # or a stopword (a header word printed as large as the main text)
    # never becomes the reference height.
    pieces = [(ln, _strip_stopwords(ln.text, cfg)) for ln in inner]
    pieces = [(ln, norm) for ln, norm in pieces if norm]
    if not pieces:
        return DominantTextReading(None, raw, None, None, (), 'no_text_after_normalize')

    tallest = max(ln.height for ln, _ in pieces)
    floor = cfg.min_height_ratio * tallest
    kept_pairs = [(ln, norm) for ln, norm in pieces if ln.height >= floor]
    ordered = reading_order([ln for ln, _ in kept_pairs])
    norm_of = {id(ln): norm for ln, norm in kept_pairs}
    text = cfg.join.join(norm_of[id(ln)] for ln in ordered)

    scores = [ln.score for ln in ordered]
    conf = min(scores)
    mean = sum(scores) / len(scores)
    kept = tuple(ordered)
    n = len(text.replace(' ', ''))
    if n < max(cfg.len_min, 1):
        return DominantTextReading(None, raw, conf, mean, kept, 'too_short')
    if cfg.len_max > 0 and n > cfg.len_max:
        return DominantTextReading(None, raw, conf, mean, kept, 'too_long')
    if conf < cfg.min_confidence:
        return DominantTextReading(None, raw, conf, mean, kept, 'low_confidence')
    return DominantTextReading(text, raw, conf, mean, kept, 'ok')


def texts_disagree(a: str | None, b: str | None, normalizer: TextNormalizer) -> bool | None:
    """Compare two readings after normalization; ``None`` if either is missing."""
    if not a or not b:
        return None
    na, nb = normalizer.normalize(a), normalizer.normalize(b)
    if not na or not nb:
        return None
    return na != nb


def resolve_region_text(
    mode: str,
    *,
    vlm_text: str | None,
    vlm_confidence: str | None,
    vlm_engine: str,
    ocr: DominantTextReading | None,
    ocr_engine: str,
    normalizer: TextNormalizer,
    rules: RegionTextRules | None = None,
) -> dict[str, Any]:
    """Region text fields for one detected region, keyed by ``RegionFields``
    attribute name (``text``, ``text_raw``, ``text_source``, …).

    Each reading is first checked against ``rules`` (placeholder, "no
    reading" word, stock run, charset / length / format; see
    :mod:`src.services.detection.region_text_rules`): a reading failing
    them is no reading. Of the valid readings the chosen one is the VLM's
    in every mode but ``ocr``, else the OCR reader's -- so a VLM
    placeholder falls back to a valid OCR reading. ``text_choice`` records
    why the chosen reading won (``TEXT_CHOICES``); ``text_vlm_invalid``
    why the VLM's reading was rejected. ``text_vlm`` / ``text_ocr`` record
    each reader's own reading whenever it produced one (the VLM's even
    when rejected, for audit), and ``text_disagreement`` compares the two
    valid readings (normalized). ``text_raw`` is the full unfiltered OCR
    reading whenever OCR ran and found text, else the VLM's verbatim
    reading. Keys with no value are omitted so a write never clears a
    field it has nothing to say about.
    """
    validate_text_reader(mode)
    vlm_text = (vlm_text or '').strip() or None
    ocr_text = ocr.text if ocr is not None else None
    vlm_invalid = rules.invalid_reason(vlm_text) if rules is not None and vlm_text else None
    ocr_rejected = bool(
        rules is not None and ocr_text and rules.invalid_reason(ocr_text) is not None
    )
    if ocr_rejected:
        ocr_text = None
    vlm_valid = None if vlm_invalid else vlm_text
    out: dict[str, Any] = {}
    if vlm_text:
        out['text_vlm'] = vlm_text
    if vlm_invalid:
        out['text_vlm_invalid'] = vlm_invalid
    if ocr_text:
        out['text_ocr'] = ocr_text
    disagree = texts_disagree(vlm_valid, ocr_text, normalizer)
    if disagree is not None:
        out['text_disagreement'] = disagree

    use_ocr = bool(ocr_text) and (mode == TEXT_READER_OCR or not vlm_valid)
    if use_ocr and ocr is not None:
        out['text'] = ocr_text
        out['text_source'] = TEXT_SOURCE_OCR
        out['text_engine_version'] = ocr_engine
        out['text_confidence'] = ocr.confidence
    elif vlm_valid:
        out['text'] = vlm_valid
        out['text_source'] = TEXT_SOURCE_VLM
        out['text_engine_version'] = vlm_engine
        if vlm_confidence:
            out['text_confidence'] = VLM_TEXT_CONFIDENCE.get(vlm_confidence, 0.70)
    choice = _text_choice(
        mode,
        use_ocr=use_ocr,
        vlm_valid=vlm_valid,
        vlm_invalid=vlm_invalid,
        rejected=bool(vlm_invalid) or ocr_rejected,
        disagree=disagree,
    )
    if choice is not None:
        out['text_choice'] = choice
    if ocr is not None and ocr.raw:
        out['text_raw'] = ocr.raw
    elif vlm_text:
        out['text_raw'] = vlm_text
    return out


def _text_choice(
    mode: str,
    *,
    use_ocr: bool,
    vlm_valid: str | None,
    vlm_invalid: str | None,
    rejected: bool,
    disagree: bool | None,
) -> str | None:
    """Why the chosen reading won; ``None`` when no reader said anything."""
    if use_ocr:
        if mode == TEXT_READER_OCR and vlm_valid:
            return TEXT_CHOICE_OCR_MODE
        return TEXT_CHOICE_VLM_INVALID if vlm_invalid else TEXT_CHOICE_OCR_ONLY
    if vlm_valid:
        if disagree is None:
            return TEXT_CHOICE_VLM_ONLY
        return TEXT_CHOICE_VLM_PREFERRED if disagree else TEXT_CHOICE_AGREE
    # No valid reading: say so only when a reader did answer (and every
    # answer was rejected).
    return TEXT_CHOICE_NONE if rejected else None


def ocr_engine_id(profile: DetectionProfile) -> str:
    """``region_text_engine_version`` for an OCR reading: the det + rec model
    ids (and versions) the pipeline ran."""
    return (
        f'{profile.ocr_det_model}:{profile.ocr_det_version}'
        f'+{profile.ocr_rec_model}:{profile.ocr_rec_version}'
    )


__all__ = [
    'TEXT_CHOICES',
    'TEXT_CHOICE_AGREE',
    'TEXT_CHOICE_HUMAN',
    'TEXT_CHOICE_NONE',
    'TEXT_CHOICE_OCR_MODE',
    'TEXT_CHOICE_OCR_ONLY',
    'TEXT_CHOICE_VLM_INVALID',
    'TEXT_CHOICE_VLM_ONLY',
    'TEXT_CHOICE_VLM_PREFERRED',
    'TEXT_READER_BOTH',
    'TEXT_READER_MODES',
    'TEXT_READER_OCR',
    'TEXT_READER_VLM',
    'TEXT_READER_VLM_THEN_OCR',
    'TEXT_SOURCE_OCR',
    'TEXT_SOURCE_VLM',
    'VLM_TEXT_CONFIDENCE',
    'DominantTextConfig',
    'DominantTextReading',
    'OcrLine',
    'TextNormalizer',
    'ocr_engine_id',
    'ocr_needed',
    'read_dominant_text',
    'reading_order',
    'resolve_region_text',
    'texts_disagree',
    'validate_text_reader',
]
