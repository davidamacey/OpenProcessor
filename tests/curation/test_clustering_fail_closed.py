"""Clustering steps whose OpenSearch call failed report the failure; they no longer
answer ``0`` (nothing parked, no false positives), which the run treats as a result."""

from __future__ import annotations

from typing import Any

import pytest
from _class_registry_fake import InMemoryRegistry, use_registry

from src.services.curation.clustering.residual_gate import _park_gated_residuals
from src.services.curation.probe_predictions import _registry_class_ids


class _Down:
    async def search(self, **_: Any) -> dict[str, Any]:
        raise RuntimeError('opensearch is down')

    async def update_by_query(self, **_: Any) -> dict[str, Any]:
        raise RuntimeError('opensearch is down')


@pytest.mark.asyncio
async def test_a_failed_park_is_an_error_not_zero_parked() -> None:
    with pytest.raises(RuntimeError, match='down'):
        await _park_gated_residuals(_Down(), max_rank=3, min_blur_ratio=0.5)


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
