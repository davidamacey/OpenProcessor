"""labels_confirmed mapping must declare every field a writer sets.

`label_import.py` (mismatch provenance) and the class-merge relabel path
(`routers/curation/classes.py`) both write fields the ``op_labels_confirmed``
mapping never declared, so they were picked up dynamically instead of with
a deliberate type. This locks the mapping (and the migration that adds it
to a long-lived index) to the literal field set those writers use.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.clients.curation_opensearch import (
    INDEX_BODIES,
    LABELS_CONFIRMED_EXTRA_MAPPING,
    ensure_labels_confirmed_fields,
)
from src.config import IndexRole, get_curation_config


config = get_curation_config()

# Literal field sets taken from the writers themselves:
#   - label_import.py:449-458 (mismatch provenance on a labels_confirmed row)
#   - routers/curation/classes.py:506-518 (class-merge relabel script)
LABEL_IMPORT_MISMATCH_FIELDS = {
    'class_mismatch',
    'detector_class_id',
    'detector_class_name',
    'detector_confidence',
}
CLASS_MERGE_RELABEL_FIELDS = {
    'class_id',
    'class_name',
    'class_source',
    'updated_at',
}


def test_labels_confirmed_mapping_declares_every_writer_field() -> None:
    body_props = INDEX_BODIES[IndexRole.LABELS_CONFIRMED]['mappings']['properties']
    for field in LABEL_IMPORT_MISMATCH_FIELDS | CLASS_MERGE_RELABEL_FIELDS:
        assert field in body_props, f'{field!r} is written but not mapped'


def test_labels_confirmed_extra_mapping_matches_body() -> None:
    body_props = INDEX_BODIES[IndexRole.LABELS_CONFIRMED]['mappings']['properties']
    for field, spec in LABELS_CONFIRMED_EXTRA_MAPPING.items():
        assert body_props[field] == spec


@pytest.mark.asyncio
async def test_ensure_labels_confirmed_fields_puts_one_mapping_per_field() -> None:
    client = AsyncMock()
    client.indices.put_mapping = AsyncMock(return_value={'acknowledged': True})
    result = await ensure_labels_confirmed_fields(client)

    assert result['acknowledged'] is True
    assert result['index'] == config.labels_confirmed_index
    assert set(result['fields_added']) == set(LABELS_CONFIRMED_EXTRA_MAPPING)
    assert client.indices.put_mapping.await_count == len(LABELS_CONFIRMED_EXTRA_MAPPING)

    seen_fields = set()
    for call in client.indices.put_mapping.await_args_list:
        assert call.kwargs['index'] == config.labels_confirmed_index
        props = call.kwargs['body']['properties']
        assert len(props) == 1
        seen_fields.update(props)
    assert seen_fields == set(LABELS_CONFIRMED_EXTRA_MAPPING)


@pytest.mark.asyncio
async def test_ensure_labels_confirmed_fields_swallows_field_conflict() -> None:
    client = AsyncMock()
    client.indices.put_mapping = AsyncMock(
        side_effect=RuntimeError('mapper_parsing_exception: field already exists')
    )
    result = await ensure_labels_confirmed_fields(client)
    assert result['acknowledged'] is True
    assert result['fields_added'] == []
