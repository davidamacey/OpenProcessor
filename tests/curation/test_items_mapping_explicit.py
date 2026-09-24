"""Every field the curation writers put on items must be explicitly mapped.

A field left to OpenSearch dynamic mapping becomes ``text`` + ``.keyword``,
so terms aggregations and sorts on the bare name fail outright (a live
deployment's ``region_status`` breakdown 400'd this way) and exact-match
filters only work by accident of tokenization.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.clients.curation_opensearch import _items_body
from src.config.region_fields import RegionFields


_PROPS = _items_body()['mappings']['properties']

# Status/provenance fields the API and workers filter, aggregate or sort on.
_MUST_BE_KEYWORD = {
    'status',
    'text_source',
    'text_engine_version',
    'label_source',
    'source',
    'pairing',
    'detector',
    'bbox_frame',
    'confidence',
}


@pytest.mark.parametrize(
    'attr', [f.name for f in dataclasses.fields(RegionFields) if f.name != 'prefix']
)
def test_every_region_field_is_explicitly_mapped(attr: str) -> None:
    name = getattr(RegionFields(), attr)
    assert name in _PROPS, f'RegionFields.{attr} ({name}) has no explicit items mapping'


@pytest.mark.parametrize('attr', sorted(_MUST_BE_KEYWORD))
def test_aggregatable_region_fields_are_keyword(attr: str) -> None:
    assert _PROPS[getattr(RegionFields(), attr)]['type'] == 'keyword'


@pytest.mark.parametrize(
    ('field', 'expected'),
    [
        ('proposal_name', 'keyword'),
        ('vlm_confidence', 'keyword'),
        ('vlm_raw_class', 'keyword'),
        ('vlm_proposed_class', 'keyword'),
        ('needs_new_class', 'boolean'),
        # scripts/curation/worker/verify.py writes this on every combined
        # VLM verify call; src/services/curation/autolabel/selection.py
        # range-queries it. Found unmapped in a live mapping diff audit —
        # see docs/design/new_class_proposal_resolve_plan.md.
        ('vlm_verify_completed_at', 'date'),
    ],
)
def test_item_label_fields_are_mapped(field: str, expected: str) -> None:
    assert _PROPS.get(field, {}).get('type') == expected


def test_region_text_supports_exact_and_partial_search() -> None:
    mapping = _PROPS[RegionFields().text]
    assert mapping['type'] == 'keyword'
    assert mapping['fields']['search']['type'] == 'text'


def test_no_query_targets_a_dynamic_keyword_subfield_of_a_region_field() -> None:
    """Explicitly keyword-mapped fields have no ``.keyword`` subfield."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    pattern = re.compile(r'\{\w+\.\w+\}\.keyword')
    offenders = [
        f'{p.relative_to(root)}:{i}'
        for base in ('src', 'scripts')
        for p in (root / base).rglob('*.py')
        for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, offenders


def test_class_history_is_an_unindexed_object() -> None:
    """F-22: class_id_history is never queried as `nested` (rg -n "'nested'"
    src scripts turns up none), yet a nested mapping cost a hidden Lucene doc
    per entry and a FieldExistsQuery[_primary_term] parent filter on every
    top-level query. Mapped `object enabled:False` instead -- the per-entry
    fields (CLASS_STATE_FIELDS etc.) stay in `_source`, unmapped, which is
    all label_undo.py (a plain `_source` read, never a query/agg) needs."""
    assert _PROPS['class_id_history'] == {'type': 'object', 'enabled': False}


@pytest.mark.parametrize(
    ('field', 'expected'),
    [
        ('class_excluded', 'boolean'),
        ('excluded_by', 'keyword'),
        ('excluded_reason', 'keyword'),
        ('excluded_prior_class_validated', 'boolean'),
        ('excluded_prior_cluster_id', 'integer'),
        ('excluded_prior_cluster_subid', 'keyword'),
    ],
)
def test_exclusion_fields_are_mapped(field: str, expected: str) -> None:
    assert _PROPS.get(field, {}).get('type') == expected
