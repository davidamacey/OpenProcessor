"""Found live: ``GET /persons`` was always empty and ``GET /persons/{id}``
always 404. The faces index maps ``person_id`` as a ``keyword`` (no
``.keyword`` sub-field), but the queries addressed ``person_id.keyword``."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.face_identity import FaceIdentityService


@pytest.fixture
def service() -> tuple[FaceIdentityService, AsyncMock]:
    svc = FaceIdentityService.__new__(FaceIdentityService)
    search = AsyncMock(
        return_value={'hits': {'hits': []}, 'aggregations': {'persons': {'buckets': []}}}
    )
    svc._opensearch = MagicMock()
    svc._opensearch.client.search = search
    return svc, search


def _body(search: AsyncMock) -> dict[str, Any]:
    call = search.await_args
    assert call is not None
    return call.kwargs['body']


@pytest.mark.asyncio
async def test_person_faces_are_matched_on_the_keyword_field(
    service: tuple[FaceIdentityService, AsyncMock],
) -> None:
    svc, search = service

    await svc.get_person_faces('person_abc')

    assert _body(search)['query'] == {'term': {'person_id': 'person_abc'}}


@pytest.mark.asyncio
async def test_persons_are_aggregated_on_the_keyword_field(
    service: tuple[FaceIdentityService, AsyncMock],
) -> None:
    svc, search = service

    await svc.get_all_persons(limit=5)

    assert _body(search)['aggs']['persons']['terms']['field'] == 'person_id'
