"""The stored per-image counts: mapping, migration and registration (R-13, lane 4A)."""

from __future__ import annotations

import asyncio
from typing import Any

from src.clients.curation_opensearch.lifecycle import INDEX_BODIES
from src.config.curation_roles import IndexRole
from src.services.curation import index_bootstrap


COUNT_KEYS = ('items', 'labeled', 'validated', 'unlabeled', 'in_holdout')


def _images() -> dict[str, Any]:
    return INDEX_BODIES[IndexRole.IMAGES]['mappings']['properties']


def test_the_images_body_maps_integer_counts_and_a_date_stamp() -> None:
    props = _images()
    assert props['counts']['properties'] == {k: {'type': 'integer'} for k in COUNT_KEYS}
    assert props['counts_at'] == {'type': 'date'}


def test_the_count_fields_migration_is_registered() -> None:
    from src.clients.curation_opensearch.count_fields import ensure_images_count_fields

    assert ensure_images_count_fields in [m for _, m in index_bootstrap._migrations()]


def test_the_migration_puts_the_mapping_onto_the_images_index() -> None:
    from src.clients.curation_opensearch.count_fields import ensure_images_count_fields
    from src.config.curation_accessors import images_index

    puts: list[dict[str, Any]] = []

    class _Indices:
        async def put_mapping(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
            puts.append({'index': index, 'body': body})
            return {'acknowledged': True}

    class _Client:
        indices = _Indices()

    result = asyncio.run(ensure_images_count_fields(_Client()))  # type: ignore[arg-type]
    assert result['acknowledged'] is True
    assert puts == [
        {
            'index': images_index(),
            'body': {'properties': {k: _images()[k] for k in ('counts', 'counts_at')}},
        }
    ]
