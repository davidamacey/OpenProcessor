"""The images index maps ``decode_route`` as a keyword, in the create body and the migration."""

from __future__ import annotations

from typing import Any

import pytest
from opensearchpy.exceptions import OpenSearchException

from src.clients.curation_opensearch.ensure_overlay_fields import ensure_images_upload_fields
from src.clients.curation_opensearch.lifecycle import INDEX_BODIES


class _Indices:
    def __init__(self) -> None:
        self.puts: list[dict[str, Any]] = []

    async def put_mapping(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.puts.append({'index': index, 'body': body})
        return {'acknowledged': True}


class _Client:
    def __init__(self) -> None:
        self.indices = _Indices()


def test_the_images_body_maps_decode_route_as_a_keyword() -> None:
    [body] = [b for role, b in INDEX_BODIES.items() if role.value == 'images']
    assert body['mappings']['properties']['decode_route'] == {'type': 'keyword'}


@pytest.mark.asyncio
async def test_the_migration_adds_decode_route_to_an_existing_images_index() -> None:
    client = _Client()
    result = await ensure_images_upload_fields(client)  # type: ignore[arg-type]
    assert result['acknowledged'] is True
    first, *writer_fields = client.indices.puts
    assert first['body']['properties']['decode_route'] == {'type': 'keyword'}
    assert 'decode_route' in result['fields_added']
    # the ingest writer's own fields go one PUT each (a dynamic mapping on one must not block
    # the others)
    assert [next(iter(p['body']['properties'])) for p in writer_fields] == [
        'ingest_write_id',
        'item_ids',
        'ingest_incomplete',
        'repair_started_at',
    ]


@pytest.mark.asyncio
async def test_a_conflict_on_one_writer_field_does_not_block_the_others() -> None:
    class _Conflicting(_Indices):
        async def put_mapping(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
            if 'item_ids' in body['properties']:
                raise OpenSearchException(400, 'illegal_argument_exception', 'mapper conflict')
            return await super().put_mapping(index=index, body=body)

    client = _Client()
    client.indices = _Conflicting()

    result = await ensure_images_upload_fields(client)  # type: ignore[arg-type]

    assert {'ingest_incomplete', 'repair_started_at'} <= set(result['fields_added'])
    assert 'item_ids' not in result['fields_added']
