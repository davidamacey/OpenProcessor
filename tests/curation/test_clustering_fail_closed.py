"""The probe's class-registry read fails closed: an unreadable registry stops the probe and an
empty one is 'no classes', never a silent empty answer."""

from __future__ import annotations

from typing import Any

import pytest
from _class_registry_fake import InMemoryRegistry, use_registry

from src.services.curation.probe_predictions import _registry_class_ids


@pytest.mark.asyncio
async def test_an_unreadable_class_registry_stops_the_probe() -> None:
    class _Unreadable(InMemoryRegistry):
        async def snapshot(self, _client: Any = None, **_kw: Any) -> Any:
            raise ConnectionError('registry document unreadable')

    use_registry(_Unreadable())

    with pytest.raises(ConnectionError, match='unreadable'):
        await _registry_class_ids()  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_an_empty_class_registry_is_no_classes() -> None:
    use_registry(InMemoryRegistry())

    assert await _registry_class_ids() == {}  # type: ignore[arg-type]
