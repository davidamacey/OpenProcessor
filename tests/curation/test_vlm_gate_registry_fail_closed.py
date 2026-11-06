"""The VLM gate pairs a pack/profile with the project's classes. An unreadable class registry
must stop the gate; it must not pass as "no classes"."""

from __future__ import annotations

import pytest
from _class_registry_fake import InMemoryRegistry, use_registry

from src.db.repos.classes import ClassRegistryError
from src.services.curation.class_registry_names import registry_class_names


class _Unreadable(InMemoryRegistry):
    async def snapshot(self, **_kw: object) -> object:
        raise ClassRegistryError('the class registry cannot be read')


@pytest.mark.asyncio
async def test_an_unreadable_registry_stops_the_gate() -> None:
    use_registry(_Unreadable())

    with pytest.raises(ClassRegistryError, match='cannot be read'):
        await registry_class_names()


@pytest.mark.asyncio
async def test_an_empty_registry_is_no_classes() -> None:
    use_registry(InMemoryRegistry())

    assert await registry_class_names() == frozenset()
