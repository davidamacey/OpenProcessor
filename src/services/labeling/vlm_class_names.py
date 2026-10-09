"""Class-name helpers of the VLM labeler: catalog rendering and reply-to-registry resolution.

Split out of ``vlm_labeler.py``. Mechanism only; the data is the ``PromptPack``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, PromptPack


if TYPE_CHECKING:
    from src.services.labeling.vlm_models import ConfidenceLevel


logger = get_logger(__name__)


def format_class_catalog(classes: list[dict[str, Any]], pack: PromptPack) -> str:
    """Render the registry as a grouped, described catalog for the prompt.

    Output looks like::

        tools: hammer (claw head), wrench (open end)
        containers: bottle, jar, ...

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
