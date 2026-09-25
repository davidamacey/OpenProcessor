"""``PromptPack`` — the domain half of the VLM labeler split (§3.4).

The reference VLM labeler hardcodes ~180 lines of domain-specific
prompt prose (system + user templates, a class description table, a
synonym table) as module constants. That prose is domain content for a
proprietary dataset family and is **not** shipped here — only the
generic *shape* (this dataclass) plus one small, neutral example
instance so the OSS product works out of the box and has test
coverage.

A future deployment-specific pack (in a proprietary-dataset config
overlay) would carry the same field set with the real domain prose.

Field-naming note: the *wire* keys a pack's prompts ask the VLM to
return for the region-of-interest sub-annotation (``region_visible``,
``region_bbox_correct``, ``region_text``, ``region_confidence`` below)
intentionally match ``RegionFields``' defaults — ``vlm_labeler.py``'s
reply parser reads those same keys via ``RegionFields`` (§3.2) rather
than hardcoding them, so a pack and the parser agree on vocabulary by
construction.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from src.core.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class PromptPack:
    """Prompt templates + domain vocabulary for one VLM labeling deployment.

    Mirrors the field set the reference VLM labeler carried as
    inline module constants (§3.4): closed- and open-vocabulary item
    classification, a combined single-call classify+region-verify+text
    prompt (single crop and numbered-batch variants), a region-only
    verify prompt (single + batch), and a region-visibility pre-filter
    prompt (batch). ``class_descriptions`` and ``synonyms`` are the two
    small vocabulary tables ``format_class_catalog`` /
    ``resolve_class_name`` (``vlm_labeler.py``) consult.
    """

    name: str

    # Closed-vocabulary item classification (``label_batch`` equivalent).
    class_system: str
    class_user_template: str  # .format(class_names_csv=...)

    # Open-vocabulary item classification (``label_or_propose_batch``) —
    # allows the VLM to propose a new class slug when nothing fits.
    open_class_system: str
    open_class_user_template: str  # .format(class_names_csv=...)

    # Combined single-call: class + region-verify + region-text, one crop.
    combined_system: str
    combined_user_template: str  # .format(class_block=..., region_block=...)

    # Combined batch: same schema, numbered images in one call.
    combined_batch_system: str
    combined_batch_rules: str

    # Region-only verify — single crop.
    region_system: str
    region_user: str

    # Region-only verify — numbered batch.
    region_batch_system: str
    region_batch_user: str

    # Region-visibility pre-filter — numbered batch, yes/no only.
    region_visible_system: str
    region_visible_user: str

    # Brief visual disambiguators for class slugs the VLM can't decode
    # from the slug alone. Keyed by class name; missing entries just
    # fall back to the bare slug (see ``format_class_catalog``).
    class_descriptions: dict[str, str] = field(default_factory=dict)

    # Maps common free-text answers the VLM might volunteer onto a
    # registry slug (see ``resolve_class_name``). Only used to *rescue*
    # predictions that don't already match a class name verbatim.
    synonyms: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict serialization -- every field is a ``str`` or a
        ``dict[str, str]``, so this round-trips through JSON cleanly."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptPack:
        """Build a :class:`PromptPack` from a plain dict (the inverse of
        :meth:`to_dict`). Unknown keys are ignored so a pack file can carry
        a ``_comment`` field (the convention this repo's other example
        config files use) without tripping ``TypeError``; missing keys
        raise ``TypeError`` the same way the dataclass constructor would,
        since every field here is required domain content, not something
        a deployment-supplied pack should be allowed to silently omit.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, path: str | Path) -> None:
        """Write this pack to ``path`` as pretty-printed JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n')

    @classmethod
    def from_json(cls, path: str | Path) -> PromptPack:
        """Load a :class:`PromptPack` from a JSON file at ``path``.

        Raises the same way ``Path.read_text`` / ``json.loads`` /
        :meth:`from_dict` would on a missing file, malformed JSON, or a
        pack missing a required field -- callers that want a fallback
        (e.g. :func:`resolve_prompt_pack`) are expected to catch and log,
        not this classmethod itself.
        """
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Neutral example pack — generic "product photo" domain.
#
# Mirrors the reference domain-specific structure (item to classify + a
# text-bearing sub-region-of-interest to verify/read) without any
# proprietary vocabulary: classify a package photo into a shipping-type
# class, then verify/read its shipping-label sub-region.
# ---------------------------------------------------------------------------

GENERIC_ITEM_PACK = PromptPack(
    name='generic_item_v1',
    class_system=(
        'You are a photo-classification assistant. Classify each numbered item crop into one '
        'of the provided class names. If unsure, choose the most likely AND set '
        "confidence='low'. Return strict JSON only — no prose, no markdown."
    ),
    class_user_template=(
        'Class names: {class_names_csv}\n'
        'Label each numbered crop. Respond as a JSON array:\n'
        '[{{"img": 1, "class": "<name>", "confidence": "high|medium|low"}}, ...]'
    ),
    open_class_system=(
        'You are a photo-classification assistant. For each numbered item crop: '
        '(1) classify into one of the provided class names; if NO existing class is a '
        'reasonable fit set ``class`` to ``__new__`` and propose a short lowercase '
        "snake_case slug in ``proposed_class``. (2) Set confidence to 'high', 'medium', or "
        "'low'. Return strict JSON only — no prose, no markdown."
    ),
    open_class_user_template=(
        'Class names: {class_names_csv}\n'
        'Label each numbered crop. Respond as a JSON array:\n'
        '[{{"img": 1, "class": "<name|__new__>", "confidence": "high|medium|low", '
        '"proposed_class": "<slug or empty>"}}, ...]'
    ),
    combined_system=(
        'You are labeling an item crop. Return STRICT JSON with these keys: '
        'class_id (int|null), class_confidence (high|medium|low|null), '
        'region_visible (bool), region_bbox_correct (bool|null), '
        'region_text (string|null: the characters printed on the region, copied '
        'verbatim; null when none are legible; never a description of the region or '
        'the item class), region_confidence (high|medium|low|null). '
        'No prose, no markdown.'
    ),
    combined_user_template=(
        '{class_block}'
        '{region_block}'
        'If asked to classify and no class matches, return class_id=-1.\n'
        'If the proposed region bbox correctly outlines the labeled sub-region, set '
        'region_bbox_correct=true and transcribe region_text.\n'
        'If the proposed region bbox is wrong but the sub-region IS visible '
        'elsewhere, set region_bbox_correct=false and region_visible=true.\n'
        'If no such sub-region is visible, set region_visible=false and '
        'region_bbox_correct=null.'
    ),
    combined_batch_system=(
        'You are labeling numbered item crops. Return STRICT JSON: '
        'a single object with key "results" whose value is an array of '
        'per-image objects (one per numbered image, in input order). '
        'Each per-image object has keys: img (1-based index), '
        'class_id (int|null), class_confidence (high|medium|low|null), '
        'region_visible (bool), region_bbox_correct (bool|null), '
        'region_text (string|null: the characters printed on the region, copied '
        'verbatim; null when none are legible; never a description of the region or '
        'the item class), region_confidence (high|medium|low|null). '
        'Output ONLY the JSON object — no prose, no markdown, no reasoning. '
        'Skip the chain-of-thought.'
    ),
    combined_batch_rules=(
        'Return STRICT JSON of the form '
        '{"results": [{"img": 1, ...}, {"img": 2, ...}, ...]}. '
        'Rules common to all images:\n'
        '- If asked to classify and no class matches, return class_id=-1.\n'
        '- If the proposed region bbox correctly outlines the labeled sub-region, set '
        'region_bbox_correct=true and transcribe region_text.\n'
        '- If the proposed region bbox is wrong but the sub-region IS visible '
        'elsewhere, set region_bbox_correct=false and region_visible=true.\n'
        '- If no such sub-region is visible, set region_visible=false and '
        'region_bbox_correct=null.\n'
        '- Respond ONLY with the JSON object above. No prose, no markdown, '
        'no reasoning preamble.\n'
        'Per-image directives follow with each image:'
    ),
    region_system=(
        'You verify whether an image shows a real labeled sub-region (e.g. a printed '
        'shipping/barcode label) and read the text on it. Output ONLY a single JSON object '
        'on the last line — no reasoning, no preamble, no markdown. Reasoning models: skip '
        'the chain-of-thought.'
    ),
    region_user=(
        'Decide: does this crop show a real printed label region, or something else (blank '
        'surface, packaging tape, unrelated marking)? If it does, also read the text on it. '
        'Reply with exactly one JSON object using these keys: is_region (boolean), '
        'confidence ("high" or "medium" or "low"), reason (string up to 15 words), text (the '
        'label text as a string, or null if no label or unreadable), text_confidence '
        '("high" or "medium" or "low" — or null when text is null).'
    ),
    region_batch_system=(
        'You verify whether each numbered image shows a real labeled sub-region. Output ONLY '
        'a JSON array — one object per image, in input order — no reasoning, no preamble, no '
        'markdown. Reasoning models: skip the chain-of-thought.'
    ),
    region_batch_user=(
        'For each numbered crop decide: is this a real printed label region, or something '
        'else? If it is, also read the text on it. Respond as a JSON array:\n'
        '[{"img": 1, "is_region": true, "confidence": "high|medium|low", '
        '"reason": "<=15 words", "text": "<the characters printed on the label>" or null, '
        '"text_confidence": "high|medium|low" or null}, ...]\n'
        'Copy the text exactly as printed; use null when no text is legible. Never '
        'guess or fill in an example value.'
    ),
    region_visible_system=(
        'You decide whether each numbered item crop contains a visible labeled sub-region '
        '(even partial / angled / small). Output ONLY a JSON object of the form '
        '{"results": [...]}, one entry per image in input order — no prose, no markdown. '
        'Reasoning models: do not echo a chain-of-thought.'
    ),
    region_visible_user=(
        'For each numbered crop, answer: is a labeled sub-region visible anywhere in the '
        'image? Count partial, angled, or small regions as visible; count blank or missing '
        'regions as not visible. Respond as a JSON object whose ``results`` field is an '
        'array of per-image verdicts in input order:\n'
        '{"results": [{"img": 1, "visible": true|false}, ...]}'
    ),
    class_descriptions={
        'box': 'rectangular cardboard shipping box',
        'envelope': 'flat paper or poly mailer',
        'tube': 'cylindrical mailing tube',
    },
    synonyms={
        'carton': 'box',
        'package': 'box',
        'mailer': 'envelope',
        'poly bag': 'envelope',
        'poly_bag': 'envelope',
    },
)


# A quoted value in a prompt: "..." or '...'. Double-quoted strings are
# consumed first so an apostrophe inside one never opens a single-quoted
# match.
_QUOTED = re.compile(r'"([^"\n]{0,64})"|\'([^\'\n]{0,64})\'')
# Not an example value: a schema placeholder or an enumeration.
_NOT_AN_EXAMPLE = re.compile(r'[<>|{}]')
_JSON_KEY_FOLLOWS = re.compile(r'\s*:')


def prompt_text_examples(pack: PromptPack) -> frozenset[str]:
    """Every literal example value quoted in ``pack``'s prompts.

    A model shown an example value (``"text": "ABC 1234"``) tends to echo
    it -- or a truncation of it -- when it can't read the real text. The
    region-text rules
    (:class:`~src.services.detection.region_text_rules.RegionTextRules`)
    treat a reading that matches one of these as a placeholder, not text,
    whatever pack is deployed. JSON keys (a quoted string followed by a
    colon), schema placeholders and enumerations are not examples.
    """
    out: set[str] = set()
    for value in asdict(pack).values():
        if not isinstance(value, str):
            continue
        for match in _QUOTED.finditer(value):
            quoted = (match.group(1) or match.group(2) or '').strip()
            if not quoted or _NOT_AN_EXAMPLE.search(quoted):
                continue
            if _JSON_KEY_FOLLOWS.match(value, match.end()):
                continue
            out.add(quoted)
    return frozenset(out)


_PACK_FILE_CACHE: dict[str, tuple[int, PromptPack]] = {}


def _load_pack_file(path: Path) -> PromptPack | None:
    """Load a pack file, cached on ``(path, mtime)``; ``None`` (with a
    logged warning) when the file is missing or malformed."""
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        logger.warning('prompt_pack_path_missing', path=str(path))
        return None
    cached = _PACK_FILE_CACHE.get(str(path))
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        pack = PromptPack.from_json(path)
    except Exception as exc:
        logger.warning('prompt_pack_load_failed', path=str(path), error=str(exc))
        return None
    _PACK_FILE_CACHE[str(path)] = (mtime, pack)
    return pack


def _config(cfg: Any | None) -> Any:
    if cfg is None:
        from src.config.curation import get_curation_config

        return get_curation_config()
    return cfg


def resolve_prompt_pack(cfg: Any | None = None) -> PromptPack:
    """Resolve the *default* :class:`PromptPack` for this process.

    Mirrors the ``CurationConfig``-driven resolution
    ``get_curation_config()`` establishes for index names / paths (see
    ``docs/design/curation_design_rationale.md`` §2.1): a deployment
    points ``OP_PROMPT_PACK_PATH`` at its own JSON pack (pallets, food
    items, ...) instead of forking any code. Never raises -- a missing
    path, a missing file, or a malformed/incomplete pack all fall back to
    :data:`GENERIC_ITEM_PACK` with a logged warning, so a bad deployment
    config degrades the labeling vocabulary rather than crashing the
    process. Additional selectable packs (``OP_PROMPT_PACK_PATHS``) are
    listed by :func:`available_prompt_packs`.

    Args:
        cfg: A :class:`~src.config.curation.CurationConfig` instance, or
            ``None`` to use the process-wide default
            (``get_curation_config()``).
    """
    path = getattr(_config(cfg), 'prompt_pack_path', None)
    if path is None:
        return GENERIC_ITEM_PACK
    pack = _load_pack_file(Path(path))
    return pack if pack is not None else GENERIC_ITEM_PACK


def available_prompt_packs(cfg: Any | None = None) -> dict[str, PromptPack]:
    """Every selectable pack, keyed by ``name``.

    Always includes the built-in :data:`GENERIC_ITEM_PACK`, plus each
    loadable file in ``OP_PROMPT_PACK_PATHS`` and the default
    ``OP_PROMPT_PACK_PATH`` pack. Unloadable files are skipped with a
    logged warning (same degrade-not-crash contract as
    :func:`resolve_prompt_pack`). On a name collision the default pack
    wins, then the earlier ``OP_PROMPT_PACK_PATHS`` entry.
    """
    config = _config(cfg)
    packs: dict[str, PromptPack] = {GENERIC_ITEM_PACK.name: GENERIC_ITEM_PACK}
    default = resolve_prompt_pack(config)
    for path in getattr(config, 'prompt_pack_paths', ()) or ():
        pack = _load_pack_file(Path(path))
        if pack is None:
            continue
        if pack.name in packs and pack.name != GENERIC_ITEM_PACK.name:
            logger.warning('prompt_pack_name_collision', name=pack.name, path=str(path))
            continue
        packs[pack.name] = pack
    packs[default.name] = default
    return packs


def get_prompt_pack(name: str, cfg: Any | None = None) -> PromptPack | None:
    """The selectable pack called ``name``, or ``None`` if not configured."""
    return available_prompt_packs(cfg).get(name)


__all__ = [
    'GENERIC_ITEM_PACK',
    'PromptPack',
    'available_prompt_packs',
    'get_prompt_pack',
    'prompt_text_examples',
    'resolve_prompt_pack',
]
