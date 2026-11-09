"""The project registry snapshot and the default project's bootstrap, over Postgres."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.services.projects.bootstrap import bootstrap_default_project
from src.services.projects.registry import ProjectRegistry

from .conftest import aseed_project


_pg = pytest.mark.usefixtures('pg_registry')


@pytest.mark.db
@_pg
def test_bootstrap_is_idempotent() -> None:
    async def _run() -> None:
        first = await bootstrap_default_project()
        second = await bootstrap_default_project()
        assert first.slug == second.slug == 'default'
        assert first.created_at == second.created_at  # not overwritten
        assert first.id == second.id

    asyncio.run(_run())


@pytest.mark.db
@_pg
def test_registry_snapshot_refreshes_on_revision_change() -> None:
    async def _run() -> None:
        registry = ProjectRegistry()
        await registry.ensure_fresh()
        assert registry.get('default') is None
        await bootstrap_default_project()
        await registry.ensure_fresh()
        assert registry.get('default') is not None
        assert registry.refreshed

    asyncio.run(_run())


@pytest.mark.db
@_pg
def test_refresh_reads_every_project() -> None:
    n = 12

    async def _run() -> ProjectRegistry:
        for i in range(n):
            await aseed_project(f'proj-{i:03d}')
        registry = ProjectRegistry()
        await registry.ensure_fresh()
        return registry

    registry = asyncio.run(_run())
    assert len(registry.snapshot()) == n  # exactly the stored rows, nothing synthesized


# --- Per-project caches and state (P1 codemod) ---


def _new_record(slug: str, tmp_path: Any) -> Any:
    import dataclasses
    from datetime import UTC, datetime

    from src.config.curation_accessors import base_curation_config
    from src.config.projects import ProjectRecord, resources_for_new

    now = datetime.now(UTC).isoformat()
    resources = resources_for_new(slug, base_curation_config())
    resources = dataclasses.replace(
        resources,
        class_registry_path=tmp_path / slug / 'class_registry.json',
        project_state_dir=tmp_path / slug / 'state',
    )
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources,
    )


def test_class_registry_is_per_project(tmp_path: Any) -> None:
    from src.config.project_context import bind_project
    from src.services.classes.registry import get_class_registry

    alpha, beta = _new_record('alpha', tmp_path), _new_record('beta', tmp_path)

    with bind_project(alpha):
        alpha_registry = get_class_registry()
        assert get_class_registry() is alpha_registry
    with bind_project(beta):
        beta_registry = get_class_registry()
    assert alpha_registry is not beta_registry


def test_the_real_registry_is_keyed_by_the_bound_slug(tmp_path: Any, monkeypatch: Any) -> None:
    from src.config.project_context import bind_project
    from src.services.classes import registry as registry_module

    monkeypatch.setattr(registry_module, '_registries', {})
    alpha, beta = _new_record('alpha', tmp_path), _new_record('beta', tmp_path)
    with bind_project(alpha):
        assert registry_module.get_class_registry().project == 'alpha'
    with bind_project(beta):
        assert registry_module.get_class_registry().project == 'beta'


def test_index_bootstrap_runs_once_per_project(tmp_path: Any, monkeypatch: Any) -> None:
    from src.config import get_curation_config
    from src.config.project_context import bind_project
    from src.services.curation import index_bootstrap

    created: list[str] = []

    async def _fake_locked(_opensearch: Any) -> None:
        created.append(get_curation_config().project_slug)
        index_bootstrap._INDEXES_BOOTSTRAPPED.add(get_curation_config().project_slug)

    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', set())
    monkeypatch.setattr(index_bootstrap, '_bootstrap_indexes', _fake_locked)

    async def _run() -> None:
        for slug in ('alpha', 'alpha', 'beta'):
            with bind_project(_new_record(slug, tmp_path)):
                await index_bootstrap.ensure_indexes(object())

    asyncio.run(_run())
    assert created == ['alpha', 'beta']


def test_job_dirs_nest_per_project_default_included(tmp_path: Any) -> None:
    from pathlib import Path

    from src.config.project_context import bind_project, project_jobs_dir

    base = tmp_path / 'jobs'
    # tests/conftest.py binds `default` for this test: it nests like any project.
    assert project_jobs_dir(base) == Path(base / 'projects' / 'default')
    with bind_project(_new_record('beta', tmp_path)):
        assert project_jobs_dir(base) == Path(base / 'projects' / 'beta')


@pytest.mark.db
@_pg
def test_default_is_an_ordinary_bootstrapped_project() -> None:
    """``default`` gets the standard naming, and a registry knows it only
    from its stored record (nothing is synthesized from the env)."""

    async def _run() -> ProjectRegistry:
        registry = ProjectRegistry()
        await registry.ensure_fresh()
        assert registry.get('default') is None  # never bootstrapped: unknown
        await bootstrap_default_project()
        await registry.ensure_fresh()
        return registry

    record = asyncio.run(_run()).get('default')
    assert record is not None
    # `default` is an ordinary project (D-A): the same index names as any other slug.
    from src.config.curation_roles import IndexRole

    assert set(record.resources.indexes.values()) == {
        f'op_prj_default__{role.value}' for role in IndexRole
    }
    assert record.resources.class_registry_path.parts[-2:] == ('default', 'class_registry.json')
    # projects_plan.md sec5.3/sec5.5 (owner D1): model_prefix is the one
    # deliberate exception to "no default special case" -- default stays
    # unprefixed so every model promoted before projects existed, and
    # every core-pipeline model, keeps resolving as default's own.
    assert record.resources.model_prefix == ''


@pytest.mark.db
@_pg
def test_bootstrap_never_resets_an_existing_default() -> None:
    async def _run() -> None:
        await aseed_project('default', status='archived', description='mine')
        again = await bootstrap_default_project()
        assert again.status == 'archived'
        assert again.description == 'mine'

    asyncio.run(_run())


@pytest.mark.db
@_pg
def test_concurrent_default_bootstrap_creates_one_record() -> None:
    async def _run() -> list[Any]:
        return list(await asyncio.gather(bootstrap_default_project(), bootstrap_default_project()))

    first, second = asyncio.run(_run())
    assert first.id == second.id
