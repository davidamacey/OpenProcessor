"""The ingest profile resolver refuses when the config store cannot confirm the
project's activation, instead of silently running the deployment profile."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import pytest
from config_store.memory_config import current as memory_config_now

from src.services.config_store import get_config_store
from src.services.config_store.store import reset_config_stores
from src.services.curation.ingest_detector import (
    IngestProfileUnresolvedError,
    project_ingest_profiles,
    refreshed_ingest_profiles,
)


@pytest.fixture(autouse=True)
def _fresh_stores(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'env_model')
    reset_config_stores()
    yield
    reset_config_stores()


def test_a_store_that_never_loaded_refuses_instead_of_using_the_env_profile() -> None:
    assert get_config_store().current.loaded_at == 0.0
    with pytest.raises(IngestProfileUnresolvedError):
        project_ingest_profiles()


def test_a_stale_store_refuses_even_with_no_activation_in_the_snapshot() -> None:
    store = get_config_store()
    store.current = dataclasses.replace(store.current, loaded_at=time.monotonic(), stale=True)
    with pytest.raises(IngestProfileUnresolvedError):
        project_ingest_profiles()


def test_a_loaded_fresh_store_with_no_activation_uses_the_env_profile() -> None:
    store = get_config_store()
    store.current = dataclasses.replace(store.current, loaded_at=time.monotonic())
    resolved = project_ingest_profiles()
    assert (resolved.source, resolved.primary.detector_model) == ('env', 'env_model')


@pytest.mark.asyncio
async def test_a_failed_first_refresh_refuses() -> None:
    memory_config_now().down = True
    with pytest.raises(IngestProfileUnresolvedError):
        await refreshed_ingest_profiles()


@pytest.mark.asyncio
async def test_a_failed_refresh_after_a_good_load_refuses_then_recovers() -> None:
    memory = memory_config_now()
    assert (await refreshed_ingest_profiles()).source == 'env'
    store = get_config_store()
    memory.down = True
    await store.refresh()
    assert store.current.stale
    with pytest.raises(IngestProfileUnresolvedError):
        project_ingest_profiles()
    # the outage ends with the epoch unchanged: the store must clear `stale`
    memory.down = False
    await store.refresh()
    assert not store.current.stale
    assert project_ingest_profiles().source == 'env'


@pytest.mark.asyncio
async def test_the_empty_answer_repair_scan_refuses_without_a_readable_profile() -> None:
    from src.services.curation.empty_vlm_answer_repair import plan_repairs

    memory_config_now().down = True

    class _NoHits:
        async def search(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {'hits': {'hits': []}}

    with pytest.raises(IngestProfileUnresolvedError):
        await plan_repairs(_NoHits(), index='items')
