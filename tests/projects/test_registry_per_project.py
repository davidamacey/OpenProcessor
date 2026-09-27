"""P1 commit 1: the project registry (op_projects) snapshot + bootstrap."""

from __future__ import annotations

import asyncio
from typing import Any

from src.services.projects.bootstrap import bootstrap_default_project
from src.services.projects.registry import ProjectRegistry


def test_bootstrap_is_idempotent(fake_registry_client) -> None:
    client = fake_registry_client

    async def _run() -> None:
        first = await bootstrap_default_project(client)
        second = await bootstrap_default_project(client)
        assert first.slug == second.slug == 'default'
        assert first.created_at == second.created_at  # not overwritten

    asyncio.run(_run())


def test_bootstrap_touches_no_data_index(fake_registry_client) -> None:
    """The registry doc store is the only thing written -- no reindex /
    update_by_query call shape is exercised (the fake would raise
    AttributeError if bootstrap tried one)."""
    client = fake_registry_client
    asyncio.run(bootstrap_default_project(client))
    assert set(client.docs) == {'project:default', 'meta:projects_revision'}


def test_registry_snapshot_refreshes_on_revision_change(fake_registry_client) -> None:
    client = fake_registry_client

    async def _run() -> None:
        await bootstrap_default_project(client)
        registry = ProjectRegistry(lambda: client)
        await registry.ensure_fresh()
        assert set(registry.snapshot()) == {'default'}

        # A second project appears only after ensure_fresh re-reads.
        from datetime import UTC, datetime

        from src.config.curation import base_curation_config
        from src.config.projects import ProjectRecord, resources_for_new
        from src.services.projects.registry import record_to_doc

        now = datetime.now(UTC).isoformat()
        record = ProjectRecord(
            slug='alpha',
            display_name='Alpha',
            description='',
            status='active',
            revision=1,
            created_at=now,
            updated_at=now,
            origin=None,
            resources=resources_for_new('alpha', base_curation_config()),
        )
        await client.index(index='op_projects', id='project:alpha', body=record_to_doc(record))
        await client.index(index='op_projects', id='meta:projects_revision', body={'revision': 2})

        await registry.ensure_fresh()
        assert set(registry.snapshot()) == {'default', 'alpha'}
        alpha = registry.get('alpha')
        assert alpha is not None
        assert alpha.resources.model_prefix == 'alpha__'

    asyncio.run(_run())


def test_registry_skips_search_when_revision_unchanged(fake_registry_client) -> None:
    """Only a GET happens when the counter has not moved -- no _search."""
    client = fake_registry_client
    search_calls = []
    orig_search = client.search

    async def _spy_search(**kwargs):
        search_calls.append(kwargs)
        return await orig_search(**kwargs)

    client.search = _spy_search  # type: ignore[method-assign]

    async def _run() -> None:
        await bootstrap_default_project(client)
        registry = ProjectRegistry(lambda: client)
        await registry.ensure_fresh()
        assert len(search_calls) == 1
        await registry.ensure_fresh()
        assert len(search_calls) == 1  # unchanged revision -> no second search

    asyncio.run(_run())


# --- Per-project caches and state (P1 codemod) ---


def _new_record(slug: str, tmp_path: Any) -> Any:
    import dataclasses
    from datetime import UTC, datetime

    from src.config.curation import base_curation_config
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


def test_class_registry_is_per_project(tmp_path: Any, monkeypatch: Any) -> None:
    import json

    from src.clients import curation_opensearch
    from src.config.project_context import bind_project

    monkeypatch.setattr(curation_opensearch, '_registries', {})
    alpha, beta = _new_record('alpha', tmp_path), _new_record('beta', tmp_path)
    for record, name in ((alpha, 'alpha_zebra'), (beta, 'beta_heron')):
        path = record.resources.class_registry_path
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'version': 1, 'classes': [{'id': 1, 'name': name}]}))

    with bind_project(alpha):
        alpha_registry = curation_opensearch.get_class_registry()
        assert alpha_registry.path == alpha.resources.class_registry_path
    with bind_project(beta):
        beta_registry = curation_opensearch.get_class_registry()
        assert beta_registry.path == beta.resources.class_registry_path
    assert alpha_registry is not beta_registry


def test_index_bootstrap_runs_once_per_project(tmp_path: Any, monkeypatch: Any) -> None:
    from src.config.project_context import bind_project
    from src.routers.curation import _common

    created: list[str] = []

    async def _fake_locked(_opensearch: Any) -> None:
        created.append(_common.config.project_slug)
        _common._INDEXES_BOOTSTRAPPED.add(_common.config.project_slug)

    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', set())
    monkeypatch.setattr(_common, '_ensure_indexes_locked', _fake_locked)

    async def _run() -> None:
        for slug in ('alpha', 'alpha', 'beta'):
            with bind_project(_new_record(slug, tmp_path)):
                await _common._ensure_indexes(object())

    asyncio.run(_run())
    assert created == ['alpha', 'beta']


def test_job_dirs_nest_per_project_default_included(tmp_path: Any, monkeypatch: Any) -> None:
    from pathlib import Path

    from src.config.project_context import bind_project
    from src.services.curation import embedding_viz, probe_job
    from src.services.curation.item_scores import job as scores_job
    from src.services.curation.selection import job as select_job

    monkeypatch.setenv('OP_SCORES_STATE_DIR', str(tmp_path / 'scores'))
    monkeypatch.setenv('OP_PROBE_JOBS_DIR', str(tmp_path / 'probe'))
    monkeypatch.setenv('OP_SELECT_JOBS_DIR', str(tmp_path / 'select'))
    monkeypatch.setenv('OP_VIZ_JOBS_DIR', str(tmp_path / 'viz'))
    dirs = {
        'scores': scores_job._state_dir,
        'probe': probe_job._jobs_dir,
        'select': select_job._jobs_dir,
        'viz': embedding_viz._jobs_dir,
    }
    # tests/conftest.py binds `default` for this test: it nests like any project.
    for name, fn in dirs.items():
        assert fn() == Path(tmp_path / name / 'projects' / 'default')
    with bind_project(_new_record('beta', tmp_path)):
        for name, fn in dirs.items():
            assert fn() == Path(tmp_path / name / 'projects' / 'beta')
        assert embedding_viz.viz_state_joblib_path().startswith(str(tmp_path / 'beta' / 'state'))


def test_bootstrap_creates_the_registry_index_with_a_strict_mapping(fake_registry_client) -> None:
    """The first registry write never auto-creates ``op_projects`` with a
    dynamic mapping."""
    client = fake_registry_client
    asyncio.run(bootstrap_default_project(client))
    body = client.indices.created['op_projects']
    assert body['mappings']['dynamic'] is False
    assert body['mappings']['properties']['slug'] == {'type': 'keyword'}


def test_revision_bump_is_optimistic(fake_registry_client) -> None:
    """Two writers that read the same revision both land: the loser of the
    race retries on its conflict instead of overwriting the winner."""
    from src.services.projects.bootstrap import bump_revision

    client = fake_registry_client

    async def _run() -> list[int]:
        await bump_revision(client)
        return list(await asyncio.gather(bump_revision(client), bump_revision(client)))

    results = asyncio.run(_run())
    assert sorted(results) == [2, 3]
    assert client.docs['meta:projects_revision'] == {'revision': 3}


def test_refresh_reads_every_project_past_one_page(fake_registry_client) -> None:
    """No size cap: more projects than one search page are all loaded."""
    from datetime import UTC, datetime

    from src.config.curation import base_curation_config
    from src.config.projects import ProjectRecord, resources_for_new
    from src.services.projects import registry as registry_mod
    from src.services.projects.registry import record_to_doc

    client = fake_registry_client
    now = datetime.now(UTC).isoformat()
    n = registry_mod._REFRESH_PAGE_SIZE * 2 + 3
    for i in range(n):
        slug = f'p{i:05d}'
        record = ProjectRecord(
            slug=slug,
            display_name=slug,
            description='',
            status='active',
            revision=1,
            created_at=now,
            updated_at=now,
            origin=None,
            resources=resources_for_new(slug, base_curation_config()),
        )
        client.docs[f'project:{slug}'] = record_to_doc(record)
        client.seq[f'project:{slug}'] = 1
    client.docs['meta:projects_revision'] = {'revision': 7}
    client.seq['meta:projects_revision'] = 1

    registry = ProjectRegistry(lambda: client)
    asyncio.run(registry.ensure_fresh())
    assert len(registry.snapshot()) == n  # exactly the stored records, nothing synthesized


def test_default_is_an_ordinary_bootstrapped_project(fake_registry_client) -> None:
    """``default`` gets the standard naming, and a registry knows it only
    from its stored record (nothing is synthesized from the env)."""
    client = fake_registry_client

    async def _run() -> ProjectRegistry:
        registry = ProjectRegistry(lambda: client)
        await registry.ensure_fresh()
        assert registry.get('default') is None  # never bootstrapped: unknown
        await bootstrap_default_project(client)
        await registry.ensure_fresh()
        return registry

    record = asyncio.run(_run()).get('default')
    assert record is not None
    assert set(record.resources.indexes.values()) == {
        f'op_prj_default__{role.value}' for role in record.resources.indexes
    }
    assert record.resources.class_registry_path.parts[-2:] == ('default', 'class_registry.json')
    assert record.resources.model_prefix == 'default__'


def test_bootstrap_never_resets_an_existing_default_on_a_read_error(fake_registry_client) -> None:
    """Re-review R2: a transient read failure (503 right after the cluster
    starts, a timeout) is not "first boot". The bootstrap raises (startup
    retries later) and the archived record is untouched."""
    import dataclasses

    import pytest

    from src.services.projects.registry import doc_to_record, record_to_doc

    client = fake_registry_client
    archived = asyncio.run(bootstrap_default_project(client))
    archived = dataclasses.replace(archived, status='archived', revision=5)
    client.docs['project:default'] = record_to_doc(archived)
    real_get = client.get

    async def _flaky_get(**kwargs: Any) -> Any:
        if kwargs.get('id') == 'project:default':
            raise ConnectionError('503 no_shard_available_action_exception')
        return await real_get(**kwargs)

    client.get = _flaky_get
    with pytest.raises(ConnectionError):
        asyncio.run(bootstrap_default_project(client))
    client.get = real_get
    stored = doc_to_record(client.docs['project:default'])
    assert (stored.status, stored.revision) == ('archived', 5)
    assert asyncio.run(bootstrap_default_project(client)).status == 'archived'


def test_concurrent_default_bootstrap_creates_one_record(fake_registry_client) -> None:
    """Two workers booting together: exactly one create wins, the other
    returns the winner's record instead of overwriting it."""
    client = fake_registry_client

    async def _run() -> list[Any]:
        return list(
            await asyncio.gather(
                bootstrap_default_project(client), bootstrap_default_project(client)
            )
        )

    first, second = asyncio.run(_run())
    assert first.created_at == second.created_at
    assert client.docs['meta:projects_revision'] == {'revision': 1}
