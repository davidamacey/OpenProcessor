"""The images index maps ``decode_route`` as a keyword, in the create body and the migration."""

from __future__ import annotations

from typing import Any

import pytest

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
    [put] = client.indices.puts
    assert put['body']['properties']['decode_route'] == {'type': 'keyword'}
    assert 'decode_route' in result['fields_added']
