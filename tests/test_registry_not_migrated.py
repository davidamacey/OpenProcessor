"""A project with a legacy ``class_registry.json`` and no registry document is refused (409
``registry_not_migrated``) on every class route until ``migrate-registry`` has run; other projects
and the inference routes are untouched. A project with neither gets its empty document created."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from _registry_doc_fake import FakeDocOpenSearch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.clients.curation_opensearch import registry as registry_module
from src.clients.curation_opensearch.registry import REGISTRY_DOC_ID, ClassRegistry
from src.core.exceptions import RequestRejectedError


if TYPE_CHECKING:
    from pathlib import Path


def _store(tmp_path: Path, name: str = 'a') -> ClassRegistry:
    return ClassRegistry(
        index=f'op_{name}__settings',
        classes_index=f'op_{name}__classes',
        legacy_path=tmp_path / name / 'class_registry.json',
        project=name,
    )


def _legacy_file(store: ClassRegistry) -> None:
    store.legacy_path.parent.mkdir(parents=True, exist_ok=True)
    store.legacy_path.write_text('{"version": 1, "classes": []}', encoding='utf-8')


@pytest.mark.asyncio
async def test_file_without_document_is_refused_with_the_migrate_command(tmp_path: Path) -> None:
    store, client = _store(tmp_path), FakeDocOpenSearch()
    _legacy_file(store)

    with pytest.raises(RequestRejectedError) as exc:
        await store.snapshot(client)

    assert exc.value.status_code == 409
    assert exc.value.detail['error'] == 'registry_not_migrated'
    assert 'openprocessor curation migrate-registry --project a' in exc.value.detail['message']
    assert client.docs == {}


@pytest.mark.asyncio
async def test_every_mutator_is_refused_too_and_creates_no_document(tmp_path: Path) -> None:
    store, client = _store(tmp_path), FakeDocOpenSearch()
    _legacy_file(store)

    for call in (
        store.add_class(client, 'x'),
        store.ensure_class(client, 'x'),
        store.set_hotkeys(client, {0: 'a'}),
    ):
        with pytest.raises(RequestRejectedError):
            await call
    assert client.docs == {}


@pytest.mark.asyncio
async def test_no_file_and_no_document_creates_the_empty_document(tmp_path: Path) -> None:
    store, client = _store(tmp_path), FakeDocOpenSearch()

    snap = await store.snapshot(client)

    assert snap.classes == []
    assert f'{store.index}/{REGISTRY_DOC_ID}' in client.docs


@pytest.mark.asyncio
async def test_a_migrated_project_answers_even_with_a_leftover_file(tmp_path: Path) -> None:
    store, client = _store(tmp_path), FakeDocOpenSearch()
    await store.add_class(client, 'car')
    _legacy_file(store)  # a stale copy left behind: the document is the authority

    snap = await store.snapshot(client)

    assert [c.class_name for c in snap.classes] == ['car']


@pytest.mark.asyncio
async def test_peeking_at_another_project_never_creates_its_document(tmp_path: Path) -> None:
    store, client = _store(tmp_path), FakeDocOpenSearch()

    snap = await store.snapshot(client, create=False)

    assert snap.classes == []
    assert client.docs == {}


def test_class_route_of_the_unmigrated_project_is_409_and_detect_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _curation_app import SCOPED, mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router

    unmigrated, migrated = _store(tmp_path, 'a'), _store(tmp_path, 'b')
    _legacy_file(unmigrated)
    client = FakeDocOpenSearch()
    monkeypatch.setattr(registry_module, '_registries', {})
    monkeypatch.setattr(registry_module, 'get_class_registry', lambda: unmigrated)
    # the router package re-exports the name its dependency calls
    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: unmigrated)

    app = mount_curation_routers(FastAPI(), router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: client
    http = TestClient(app, raise_server_exceptions=False)

    @app.get('/detect_probe')
    async def _inference_probe() -> dict[str, Any]:
        return {'ok': True}

    refused = http.get(f'{SCOPED}/classes')
    assert refused.status_code == 409
    assert refused.json()['detail']['error'] == 'registry_not_migrated'
    assert http.get('/detect_probe').status_code == 200

    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: migrated)
    assert http.get(f'{SCOPED}/classes').status_code != 409
