"""``GET /export/status`` serves ``can_export`` / ``blocking_reasons`` from the
same query + message the export route's 422 uses."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation.export_readiness import (
    NO_VALIDATED_ITEMS_REASON,
    export_blocking_reasons,
    validated_export_query,
)


@pytest.fixture
def client_factory(monkeypatch: pytest.MonkeyPatch):
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    monkeypatch.setattr(
        'src.routers.curation.export._resolve_current_export_dir',
        lambda: (_ for _ in ()).throw(FileNotFoundError()),
    )

    def make(fake_os) -> TestClient:
        from _curation_app import mount_curation_routers

        app = FastAPI()
        mount_curation_routers(app, curation_router)
        app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
        return TestClient(app)

    return make


def _os_with_count(count: int | Exception) -> AsyncMock:
    fake = AsyncMock()
    if isinstance(count, Exception):
        fake.count = AsyncMock(side_effect=count)
    else:
        fake.count = AsyncMock(return_value={'count': count})
    return fake


def test_status_blocks_with_the_422_reason_when_nothing_validated(client_factory) -> None:
    fake = _os_with_count(0)
    body = client_factory(fake).get('/curation/projects/default/export/status').json()
    assert body['status'] == 'idle'
    assert body['can_export'] is False
    assert body['blocking_reasons'] == [NO_VALIDATED_ITEMS_REASON]
    # The count ran the very query the export scrolls.
    assert fake.count.await_args.kwargs['body'] == {'query': validated_export_query()}


def test_status_can_export_when_validated_items_exist(client_factory) -> None:
    body = client_factory(_os_with_count(7)).get('/curation/projects/default/export/status').json()
    assert body['can_export'] is True
    assert body['blocking_reasons'] == []


def test_status_readiness_is_null_not_false_when_count_fails(client_factory) -> None:
    body = (
        client_factory(_os_with_count(ConnectionError('down')))
        .get('/curation/projects/default/export/status')
        .json()
    )
    assert body['can_export'] is None
    assert body['blocking_reasons'] == []


@pytest.mark.asyncio
async def test_export_service_refuses_with_the_same_reason_and_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.curation.export import GenericYoloExportService
    from src.services.curation.export_readiness import NothingToExportError

    scroll = AsyncMock(return_value=[])
    monkeypatch.setattr(GenericYoloExportService, '_scroll_items', scroll)
    monkeypatch.setattr(
        'src.services.curation.export.items_index_generation', AsyncMock(return_value=None)
    )
    service = GenericYoloExportService.__new__(GenericYoloExportService)
    service.opensearch = AsyncMock()
    service.config = type('C', (), {'items_index': 'idx'})()  # type: ignore[assignment]
    with pytest.raises(NothingToExportError) as exc:
        await service.export_dataset(copy_images=False)
    assert str(exc.value) == NO_VALIDATED_ITEMS_REASON
    assert scroll.await_args is not None
    assert scroll.await_args.args[0] == validated_export_query()
    assert await export_blocking_reasons(_os_with_count(0), 'idx') == [NO_VALIDATED_ITEMS_REASON]
