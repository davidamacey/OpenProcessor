"""If the control database is unreachable when the API starts, the project bootstrap
must keep retrying in the background, not give up for the life of the
process. Before, one failed attempt meant no ``default`` record on a fresh
install, no registry poll loop and no region-class seed."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.services.projects.registry import ProjectRegistry, set_project_registry


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_registry')]


@pytest.mark.asyncio
async def test_bootstrap_retries_until_the_database_is_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.projects import bootstrap

    attempts = 0

    real = bootstrap.bootstrap_default_project

    async def _flaky_bootstrap() -> Any:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError('database connection failed')
        return await real()

    seeded: list[str] = []
    monkeypatch.setattr(bootstrap, 'bootstrap_default_project', _flaky_bootstrap)
    monkeypatch.setattr(bootstrap, '_BOOTSTRAP_RETRY_INITIAL_S', 0.01)

    async def _seed() -> None:
        seeded.append('seeded')

    monkeypatch.setattr(bootstrap, '_seed_region_classes', _seed)
    registry = ProjectRegistry()
    set_project_registry(registry)
    try:
        task = await bootstrap.startup_bootstrap_project_registry_safe()
        assert task is not None
        for _ in range(200):
            if registry.get('default') is not None and seeded:
                break
            await asyncio.sleep(0.01)
        assert registry.get('default') is not None
        assert seeded == ['seeded']
        assert attempts == 3
        await bootstrap.shutdown_project_registry(task)
    finally:
        set_project_registry(None)
