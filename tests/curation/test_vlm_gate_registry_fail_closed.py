"""The VLM gate pairs a pack/profile with the project's classes. An unreadable or not-yet-migrated
class registry must stop the gate; it must not pass as "no classes"."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from _registry_doc_fake import FakeDocOpenSearch

from src.clients.curation_opensearch import registry as registry_module
from src.clients.curation_opensearch.registry import REGISTRY_DOC_ID, ClassRegistry
from src.core.exceptions import RequestRejectedError
from src.services.curation.class_registry_names import registry_class_names


if TYPE_CHECKING:
    from pathlib import Path


def _real_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: FakeDocOpenSearch
) -> ClassRegistry:
    store = ClassRegistry(
        index='op_gate__settings',
        classes_index='op_gate__classes',
        legacy_path=tmp_path / 'class_registry.json',
        project='gate',
    )
    monkeypatch.setattr(registry_module, '_registries', {})
    monkeypatch.setattr('src.services.curation.registry_access.get_class_registry', lambda: store)

    async def _client() -> Any:
        return client

    monkeypatch.setattr('src.services.curation.registry_access.make_curation_opensearch', _client)
    return store


@pytest.mark.asyncio
async def test_a_corrupt_registry_document_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeDocOpenSearch()
    store = _real_store(monkeypatch, tmp_path, client)
    client.docs[f'{store.index}/{REGISTRY_DOC_ID}'] = {'class_registry': {'classes': 'not a list'}}
    client.seqs[f'{store.index}/{REGISTRY_DOC_ID}'] = 1

    with pytest.raises(ValueError, match='classes'):
        await registry_class_names()


@pytest.mark.asyncio
async def test_a_registry_with_no_document_and_no_file_is_no_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_store(monkeypatch, tmp_path, FakeDocOpenSearch())

    assert await registry_class_names() == frozenset()


@pytest.mark.asyncio
async def test_a_registry_still_in_the_legacy_file_stops_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _real_store(monkeypatch, tmp_path, FakeDocOpenSearch())
    store.legacy_path.write_text('{"version": 1, "classes": []}')

    with pytest.raises(RequestRejectedError) as exc:
        await registry_class_names()
    assert exc.value.detail['error'] == 'registry_not_migrated'
