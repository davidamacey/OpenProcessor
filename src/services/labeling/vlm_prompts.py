"""``PromptPack`` — the domain half of the VLM labeler split.

A deployment's prompt prose (system + user templates, a class
description table, a synonym table) is domain-specific, so it lives in
a ``PromptPack`` instance rather than module constants — only the
generic *shape* (this dataclass) plus one small, neutral example
instance ship here, so the product works out of the box and has test
coverage.

A deployment-specific pack (in a deployment's own config overlay)
carries the same field set with its own domain prose.

Field-naming note: the *wire* keys a pack's prompts ask the VLM to
return for the region-of-interest sub-annotation (``region_visible``,
``region_bbox_correct``, ``region_text``, ``region_confidence`` below)
intentionally match ``RegionFields``' defaults — ``vlm_reply_parse.py``'s
reply parser reads those same keys via ``RegionFields`` rather
than hardcoding them, so a pack and the parser agree on vocabulary by
construction.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.labeling.region_overlay import (
    REPLY_BBOX_CORRECT_KEY,
    REPLY_CONFIDENCE_KEY,
    REPLY_TEXT_KEY,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)


@dataclass(frozen=True)
class PromptPack:
    """Prompt templates + domain vocabulary for one VLM labeling deployment.

    Carries the field set a deployment needs: closed- and open-vocabulary item
    classification, a combined single-call classify+region-verify+text
    prompt (single crop and numbered-batch variants), a region-only
    verify prompt (single + batch), and a region-visibility pre-filter
    prompt (batch). ``class_descriptions`` and ``synonyms`` are the two
    small vocabulary tables ``format_class_catalog`` /
    ``resolve_class_name`` (``vlm_class_names.py``) consult.
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

    # Glob patterns (``fnmatch``, case-insensitive, against the sanitized
    # slug) for proposed class names that are scene / image-quality words,
    # not classes (``blurry_*``, ``*_scene``). A matching proposal is
    # dropped so the crop reads as "nothing fits", not a new-class
    # candidate. Empty = no filtering.
    proposal_denylist: list[str] = field(default_factory=list)

    # Registry prior (#61): when > 0, open-vocabulary labeling tells the VLM
    # the top-k registry classes (by validated count) and pending proposal
    # names as a hint. 0 = off. Bounded by ``MAX_REGISTRY_PRIOR_TOP_K``.
    registry_prior_top_k: int = 0

    # Detector hint (#193): when > 0, open-vocabulary labeling tells the VLM each
    # item's stored detector class name when the detector confidence is at least
    # this many percent. 0 = off. Bounded by ``MAX_DETECTOR_HINT_PCT`` (100).
    detector_hint_min_confidence_pct: int = 0

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


def proposal_denied(slug: str, patterns: Sequence[str]) -> bool:
    """True when ``slug`` matches any ``patterns`` glob (case-insensitive)."""
    low = slug.lower()
    return any(fnmatch.fnmatchcase(low, str(p).lower()) for p in patterns)


#: Shipped proposal-name denylist: scene / image-quality words that are never a
#: class. A live COCO oracle run (#193) put ``abstract_background`` at 60% of pending
#: proposals, so the abstract / background / generic-object families are listed beside
#: the blur and emptiness words. Globs are anchored (``fnmatch``, whole slug) and
#: tested against the COCO class names so none of them eats a real class.
DEFAULT_PROPOSAL_DENYLIST: tuple[str, ...] = (
    'abstract*',
    'background*',
    '*_background',
    'object',
    'objects',
    '*_object',
    'blurry*',
    'blurred*',
    '*_blurry',
    '*_blur',
    'blur_*',
    'out_of_focus*',
    'low_quality*',
    'low_resolution*',
    'unclear*',
    'unidentified*',
    'scene*',
    '*_scene',
    'empty*',
    'unknown*',
    '*_image',
    '*_photo',
    '*_abstract',
    'blank',
    'blank_*',
    'indoor',
    'indoors',
    'outdoor',
    'outdoors',
    'shadow*',
    '*_shadow',
)


# ---------------------------------------------------------------------------
# Neutral example pack — generic "product photo" domain.
#
# Same structure any deployment pack needs (item to classify + a
# text-bearing sub-region-of-interest to verify/read), with no
# domain-specific vocabulary: classify a package photo into a shipping-type
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
        'region_visible (bool: true if the labeled sub-region is visible anywhere in '
        'the image, regardless of whether any proposed box below is correct), '
        'region_boxes (array, one entry per numbered candidate box shown in the '
        'image -- always an array, even for a single box: '
        '[{"box": <the number on the overlay>, '
        '"region_bbox_correct": bool|null, '
        '"region_text": string|null (the characters printed on that box\'s region, '
        'copied verbatim; null when none are legible; never a description of the '
        'region or the item class), '
        '"region_confidence": "high"|"medium"|"low"|null}, ...]). '
        'No prose, no markdown.'
    ),
    combined_user_template=(
        '{class_block}'
        '{region_block}'
        'If asked to classify and no class matches, return class_id=-1.\n'
        'For each numbered box: if it correctly outlines the labeled sub-region, set '
        "that box's region_bbox_correct=true and transcribe its region_text.\n"
        "If a box is wrong but the sub-region IS visible elsewhere, set that box's "
        'region_bbox_correct=false and region_visible=true.\n'
        'If no such sub-region is visible anywhere, set region_visible=false and '
        "every box's region_bbox_correct=null."
    ),
    combined_batch_system=(
        'You are labeling numbered item crops. Return STRICT JSON: '
        'a single object with key "results" whose value is an array of '
        'per-image objects (one per numbered image, in input order). '
        'Each per-image object has keys: img (1-based index), '
        'class_id (int|null), class_confidence (high|medium|low|null), '
        'region_visible (bool: true if the labeled sub-region is visible anywhere in '
        'that image, regardless of whether any proposed box is correct), '
        'region_boxes (array, one entry per numbered candidate box shown in that '
        'image -- always an array, even for a single box: '
        '[{"box": <the number on that image\'s overlay>, '
        '"region_bbox_correct": bool|null, '
        '"region_text": string|null (the characters printed on that box\'s region, '
        'copied verbatim; null when none are legible; never a description of the '
        'region or the item class), '
        '"region_confidence": "high"|"medium"|"low"|null}, ...]). '
        'Output ONLY the JSON object — no prose, no markdown, no reasoning. '
        'Skip the chain-of-thought.'
    ),
    combined_batch_rules=(
        'Return STRICT JSON of the form '
        '{"results": [{"img": 1, ...}, {"img": 2, ...}, ...]}. '
        'Rules common to all images:\n'
        '- If asked to classify and no class matches, return class_id=-1.\n'
        '- For each numbered box: if it correctly outlines the labeled sub-region, '
        "set that box's region_bbox_correct=true and transcribe its region_text.\n"
        '- If a box is wrong but the sub-region IS visible elsewhere, set that '
        "box's region_bbox_correct=false and region_visible=true.\n"
        '- If no such sub-region is visible anywhere, set region_visible=false and '
        "every box's region_bbox_correct=null.\n"
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
    # Deliberately no class_descriptions/synonyms: they name registry classes,
    # and the shipped default must validate with zero warnings on any registry
    # (empty, COCO, ...). Domain vocabulary belongs in a stored pack.
    proposal_denylist=list(DEFAULT_PROPOSAL_DENYLIST),
)


# ---------------------------------------------------------------------------
# Neutral text-free example pack — a region with nothing to read.
#
# Same structure as GENERIC_ITEM_PACK (classify the item, verify a
# sub-region of interest on it), but no prompt asks for text: the pack a
# text-free region profile (``text_reader='none'``) pairs with. The reply
# parser treats every text key as optional, so leaving them out is all a
# text-free pack has to do.
# ---------------------------------------------------------------------------

GENERIC_REGION_PACK = PromptPack(
    name='generic_region_v1',
    class_system=GENERIC_ITEM_PACK.class_system,
    class_user_template=GENERIC_ITEM_PACK.class_user_template,
    open_class_system=GENERIC_ITEM_PACK.open_class_system,
    open_class_user_template=GENERIC_ITEM_PACK.open_class_user_template,
    combined_system=(
        'You are labeling an item crop. Return STRICT JSON with these keys: '
        'class_id (int|null), class_confidence (high|medium|low|null), '
        'region_visible (bool: true if the sub-region of interest is visible anywhere '
        'in the image, regardless of whether any proposed box below is correct), '
        'region_boxes (array, one entry per numbered candidate box shown in the '
        'image -- always an array, even for a single box: '
        '[{"box": <the number on the overlay>, '
        '"region_bbox_correct": bool|null, '
        '"region_confidence": "high"|"medium"|"low"|null}, ...]). '
        'No prose, no markdown.'
    ),
    combined_user_template=(
        '{class_block}'
        '{region_block}'
        'If asked to classify and no class matches, return class_id=-1.\n'
        'For each numbered box: if it correctly outlines the sub-region of interest, '
        "set that box's region_bbox_correct=true.\n"
        "If a box is wrong but the sub-region IS visible elsewhere, set that box's "
        'region_bbox_correct=false and region_visible=true.\n'
        'If no such sub-region is visible anywhere, set region_visible=false and '
        "every box's region_bbox_correct=null."
    ),
    combined_batch_system=(
        'You are labeling numbered item crops. Return STRICT JSON: '
        'a single object with key "results" whose value is an array of '
        'per-image objects (one per numbered image, in input order). '
        'Each per-image object has keys: img (1-based index), '
        'class_id (int|null), class_confidence (high|medium|low|null), '
        'region_visible (bool: true if the sub-region of interest is visible anywhere '
        'in that image, regardless of whether any proposed box is correct), '
        'region_boxes (array, one entry per numbered candidate box shown in that '
        'image -- always an array, even for a single box: '
        '[{"box": <the number on that image\'s overlay>, '
        '"region_bbox_correct": bool|null, '
        '"region_confidence": "high"|"medium"|"low"|null}, ...]). '
        'Output ONLY the JSON object — no prose, no markdown, no reasoning. '
        'Skip the chain-of-thought.'
    ),
    combined_batch_rules=(
        'Return STRICT JSON of the form '
        '{"results": [{"img": 1, ...}, {"img": 2, ...}, ...]}. '
        'Rules common to all images:\n'
        '- If asked to classify and no class matches, return class_id=-1.\n'
        '- For each numbered box: if it correctly outlines the sub-region of '
        "interest, set that box's region_bbox_correct=true.\n"
        '- If a box is wrong but the sub-region IS visible elsewhere, set that '
        "box's region_bbox_correct=false and region_visible=true.\n"
        '- If no such sub-region is visible anywhere, set region_visible=false and '
        "every box's region_bbox_correct=null.\n"
        '- Respond ONLY with the JSON object above. No prose, no markdown, '
        'no reasoning preamble.\n'
        'Per-image directives follow with each image:'
    ),
    region_system=(
        'You verify whether an image shows the sub-region of interest (a distinct part of '
        'the item). Output ONLY a single JSON object on the last line — no reasoning, no '
        'preamble, no markdown. Reasoning models: skip the chain-of-thought.'
    ),
    region_user=(
        'Decide: does this crop show the sub-region of interest, or something else (a '
        'different part of the item, background, an unrelated object)? Reply with exactly '
        'one JSON object using these keys: is_region (boolean), confidence ("high" or '
        '"medium" or "low"), reason (string up to 15 words).'
    ),
    region_batch_system=(
        'You verify whether each numbered image shows the sub-region of interest. Output '
        'ONLY a JSON array — one object per image, in input order — no reasoning, no '
        'preamble, no markdown. Reasoning models: skip the chain-of-thought.'
    ),
    region_batch_user=(
        'For each numbered crop decide: is this the sub-region of interest, or something '
        'else? Respond as a JSON array:\n'
        '[{"img": 1, "is_region": true, "confidence": "high|medium|low", '
        '"reason": "<=15 words"}, ...]'
    ),
    region_visible_system=(
        'You decide whether each numbered item crop contains a visible sub-region of '
        'interest (even partial / angled / small). Output ONLY a JSON object of the form '
        '{"results": [...]}, one entry per image in input order — no prose, no markdown. '
        'Reasoning models: do not echo a chain-of-thought.'
    ),
    region_visible_user=(
        'For each numbered crop, answer: is the sub-region of interest visible anywhere in '
        'the image? Count partial, angled, or small regions as visible; count hidden or '
        'missing regions as not visible. Respond as a JSON object whose ``results`` field '
        'is an array of per-image verdicts in input order:\n'
        '{"results": [{"img": 1, "visible": true|false}, ...]}'
    ),
    proposal_denylist=list(DEFAULT_PROPOSAL_DENYLIST),
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


# ---------------------------------------------------------------------------
# Reply-key contract (W3, any_domain_plan.md §3.3) -- the wire keys each
# pack "call" (a system+user template pair) must ask the VLM to return,
# so ``pack_validation.py`` can check a draft pack names every key its
# matching parser reads. Keys named here are the RegionFields *default*
# names (``region_visible``, ``region_bbox_correct``, ``region_text``,
# ``region_confidence``, ``region_boxes``, ``region_set_complete``) --
# callers that need a deployment's actual (possibly overridden) field
# names resolve them via ``get_region_fields()`` and remap this table,
# since a pack's prose always talks about the *default* vocabulary the
# parser reads through ``RegionFields``.
# ---------------------------------------------------------------------------

REPLY_KEY_CONTRACT: dict[str, dict[str, list[str]]] = {
    'classify': {
        'fields': ['class_system', 'class_user_template'],
        'required': ['img', 'class', 'confidence'],
        'optional': ['make', 'model'],
    },
    'open_classify': {
        'fields': ['open_class_system', 'open_class_user_template'],
        'required': ['img', 'class', 'confidence', 'proposed_class'],
        'optional': [],
    },
    'combined': {
        'fields': ['combined_system', 'combined_user_template'],
        'required': [
            'class_id',
            'class_confidence',
            'region_visible',
            REPLY_BBOX_CORRECT_KEY,
            REPLY_CONFIDENCE_KEY,
        ],
        'optional': [REPLY_TEXT_KEY],
        # W8: the list-shaped per-box verdict keys -- required whenever the
        # active/given profile's max_regions_per_item > 1 (D-B, list shape
        # only; see pack_validation.pack_multi_region_keys_missing).
        'multi_region': ['region_boxes', 'box', REPLY_BBOX_CORRECT_KEY, REPLY_CONFIDENCE_KEY],
    },
    'combined_batch': {
        'fields': ['combined_batch_system', 'combined_batch_rules'],
        'required': [
            'img',
            'results',
            'class_id',
            'class_confidence',
            'region_visible',
            REPLY_BBOX_CORRECT_KEY,
            REPLY_CONFIDENCE_KEY,
        ],
        'optional': [REPLY_TEXT_KEY],
        'multi_region': ['region_boxes', 'box', REPLY_BBOX_CORRECT_KEY, REPLY_CONFIDENCE_KEY],
    },
    'region_verify': {
        'fields': ['region_system', 'region_user'],
        'required': ['is_region', 'confidence'],
        'optional': ['reason', 'text', 'text_confidence'],
    },
    'region_verify_batch': {
        'fields': ['region_batch_system', 'region_batch_user'],
        'required': ['img', 'is_region', 'confidence'],
        'optional': ['reason', 'text', 'text_confidence'],
    },
    'region_visible': {
        'fields': ['region_visible_system', 'region_visible_user'],
        'required': ['results', 'img', 'visible'],
        'optional': [],
    },
}

# The three fields that go through ``str.format`` -- every other field is
# sent verbatim (any_domain_plan.md §3.3).
FORMATTED_PLACEHOLDERS: dict[str, tuple[str, ...]] = {
    'class_user_template': ('class_names_csv',),
    'open_class_user_template': ('class_names_csv',),
    'combined_user_template': ('class_block', 'region_block'),
}


BUILT_IN_PACKS: tuple[PromptPack, ...] = (GENERIC_ITEM_PACK, GENERIC_REGION_PACK)
_BUILT_IN_NAMES = frozenset(p.name for p in BUILT_IN_PACKS)

# Resolution (default/active pack selection, file overrides, the
# config-store activation pin) lives in vlm_prompt_resolution.py -- split
# out to stay under the 700-LOC ratchet; re-exported here so every
# existing `from src.services.labeling.vlm_prompts import ...` caller is
# unaffected. Import placed after BUILT_IN_PACKS/_BUILT_IN_NAMES: that
# module imports them back from this one.
from src.services.labeling.vlm_prompt_resolution import (  # noqa: E402
    active_prompt_pack,
    available_prompt_packs,
    get_prompt_pack,
    prompt_pack_stamp,
    resolve_prompt_pack,
)


__all__ = [
    'BUILT_IN_PACKS',
    'DEFAULT_PROPOSAL_DENYLIST',
    'FORMATTED_PLACEHOLDERS',
    'GENERIC_ITEM_PACK',
    'GENERIC_REGION_PACK',
    'REPLY_KEY_CONTRACT',
    'PromptPack',
    'active_prompt_pack',
    'available_prompt_packs',
    'get_prompt_pack',
    'prompt_pack_stamp',
    'prompt_text_examples',
    'resolve_prompt_pack',
]
