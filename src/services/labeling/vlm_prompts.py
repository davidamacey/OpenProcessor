"""``PromptPack`` — the domain half of the VLM labeler split (§3.4).

The reference ``gemma_labeler.py`` hardcodes ~180 lines of vehicle /
plate prompt prose (system + user templates, a vehicle-class
description table, a synonym table) as module constants. That prose is
domain content for a proprietary dataset family and is **not** shipped
here — only the generic *shape* (this dataclass) plus one small, neutral
example instance so the OSS product works out of the box and has test
coverage.

A future deployment-specific pack (e.g. a ``VEHICLE_PROMPT_PACK`` in a
proprietary-dataset config overlay, see Appendix A of the plan) would
carry the same field set with the real vehicle/plate prose.

Field-naming note: the *wire* keys a pack's prompts ask the VLM to
return for the region-of-interest sub-annotation (``region_visible``,
``region_bbox_correct``, ``region_text``, ``region_confidence`` below)
intentionally match ``RegionFields``' defaults — ``vlm_labeler.py``'s
reply parser reads those same keys via ``RegionFields`` (§3.2) rather
than hardcoding them, so a pack and the parser agree on vocabulary by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptPack:
    """Prompt templates + domain vocabulary for one VLM labeling deployment.

    Mirrors the field set the reference ``gemma_labeler.py`` carried as
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


# ---------------------------------------------------------------------------
# Neutral example pack — generic "product photo" domain.
#
# Mirrors the reference vehicle+plate structure (item to classify + a
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
        'region_text (string|null), region_confidence (high|medium|low|null). '
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
        'region_text (string|null), region_confidence (high|medium|low|null). '
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
        '"reason": "<=15 words", "text": "ABC 1234" or null, '
        '"text_confidence": "high|medium|low" or null}, ...]'
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


__all__ = ['GENERIC_ITEM_PACK', 'PromptPack']
