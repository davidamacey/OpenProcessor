"""Pack-declared attributes: free-form facts a VLM reports about an item or a
box (car make, a rim material) that are suggestions, never labels.

A pack declares them as :class:`AttributeSpec` entries. This module owns the
spec shape and its limits, the prompt block rendered from the specs, and the
reply parser that keeps only declared keys with allowed values. It imports
nothing from the prompt pack, which imports it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from src.services.labeling.vlm_models import AttributeValue, ConfidenceLevel


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

AppliesTo = Literal['item', 'box']
APPLIES_TO: tuple[str, ...] = ('item', 'box')

KEY_RE = re.compile(r'^[a-z][a-z0-9_]{0,31}$')
MAX_ATTRIBUTES = 16
MAX_DESCRIPTION_LEN = 300
MAX_VALUES = 50
MAX_VALUE_LEN = 48

_CONFIDENCES: tuple[ConfidenceLevel, ...] = ('high', 'medium', 'low')


def normalize_value(raw: Any) -> str:
    """A reply or vocabulary value as stored: stripped, lowercase, capped."""
    return str(raw if raw is not None else '').strip().lower()[:MAX_VALUE_LEN]


@dataclass(frozen=True)
class AttributeSpec:
    key: str
    description: str = ''
    values: list[str] = field(default_factory=list)
    """Closed vocabulary (lowercase); empty = free text."""
    applies_to: AppliesTo = 'item'

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttributeSpec:
        """Build a spec from a pack body entry; the vocabulary is normalised
        and unknown keys are ignored (validation reports malformed entries
        before a pack is saved)."""
        raw_values = data.get('values') or []
        values = [v for v in (normalize_value(x) for x in raw_values) if v]
        applies = data.get('applies_to', 'item')
        return cls(
            key=str(data.get('key', '')),
            description=str(data.get('description', '')),
            values=values,
            applies_to='box' if applies == 'box' else 'item',
        )


def specs_from_body(raw: Any) -> list[AttributeSpec]:
    """The specs of a pack body's ``attributes`` value (``[]`` when absent)."""
    if not isinstance(raw, list):
        return []
    return [
        item if isinstance(item, AttributeSpec) else AttributeSpec.from_dict(item)
        for item in raw
        if isinstance(item, AttributeSpec | dict)
    ]


def specs_for(specs: Iterable[AttributeSpec], applies_to: AppliesTo) -> list[AttributeSpec]:
    return [s for s in specs if s.applies_to == applies_to]


def render_attribute_block(specs: Sequence[AttributeSpec]) -> str:
    """The ``{attribute_block}`` text: one line per attribute with its
    description and, for a closed vocabulary, the allowed values."""
    lines = []
    for spec in specs:
        line = f'- {spec.key}'
        if spec.description:
            line += f': {spec.description}'
        if spec.values:
            line += f' (one of: {", ".join(spec.values)})'
        lines.append(line)
    return '\n'.join(lines)


def parse_attributes(
    raw: Any, specs: Sequence[AttributeSpec]
) -> tuple[dict[str, AttributeValue], int]:
    """``({key: answer}, invalid_values)`` from a reply entry's ``attributes``.

    Keys the pack did not declare are dropped silently. A value outside a
    closed vocabulary is dropped and counted. A missing or unknown
    confidence becomes ``low``. An empty value is no answer."""
    if not isinstance(raw, dict):
        return {}, 0
    declared = {s.key: s for s in specs}
    out: dict[str, AttributeValue] = {}
    invalid = 0
    for key, answer in raw.items():
        spec = declared.get(key)
        if spec is None:
            continue
        if isinstance(answer, dict):
            value_raw, conf_raw = answer.get('value'), answer.get('confidence')
        else:
            value_raw, conf_raw = answer, None
        value = normalize_value(value_raw)
        if not value or value in ('null', 'none', 'unknown', 'n/a'):
            continue
        if spec.values and value not in spec.values:
            invalid += 1
            continue
        conf = str(conf_raw).strip().lower() if conf_raw is not None else 'low'
        out[key] = AttributeValue(
            value=value,
            confidence=conf if conf in _CONFIDENCES else 'low',
        )
    return out, invalid


DEFAULT_ASSIST_SYSTEM = (
    'You are a photo-classification assistant. For each numbered item crop, classify it into '
    'one of the provided class names and report the requested attributes. Attributes are '
    'observations: leave one out when the crop does not show it; never guess. Return strict '
    'JSON only, no prose, no markdown.'
)
DEFAULT_ASSIST_USER_TEMPLATE = (
    'Class names: {class_list}\n'
    'Attributes to report (only those visible):\n{attribute_block}\n'
    'Answer for each of the {n_images} numbered crops as a JSON array:\n'
    '[{{"img": 1, "class": "<name|__new__>", "confidence": "high|medium|low", '
    '"proposed_class": "<slug or empty>", "attributes": '
    '{{"<key>": {{"value": "<value>", "confidence": "high|medium|low"}}}}}}, ...]'
)
DEFAULT_ATTRIBUTES_SYSTEM = (
    'You describe numbered crops. Report only the requested attributes, only when the crop '
    'shows them; never guess. Return strict JSON only, no prose, no markdown.'
)
DEFAULT_ATTRIBUTES_USER_TEMPLATE = (
    'Attributes to report (only those visible):\n{attribute_block}\n'
    'Answer for each of the {n_images} numbered crops as a JSON array:\n'
    '[{{"img": 1, "attributes": '
    '{{"<key>": {{"value": "<value>", "confidence": "high|medium|low"}}}}}}, ...]'
)

__all__ = [
    'APPLIES_TO',
    'DEFAULT_ASSIST_SYSTEM',
    'DEFAULT_ASSIST_USER_TEMPLATE',
    'DEFAULT_ATTRIBUTES_SYSTEM',
    'DEFAULT_ATTRIBUTES_USER_TEMPLATE',
    'KEY_RE',
    'MAX_ATTRIBUTES',
    'MAX_DESCRIPTION_LEN',
    'MAX_VALUES',
    'MAX_VALUE_LEN',
    'AppliesTo',
    'AttributeSpec',
    'normalize_value',
    'parse_attributes',
    'render_attribute_block',
    'specs_for',
    'specs_from_body',
]
