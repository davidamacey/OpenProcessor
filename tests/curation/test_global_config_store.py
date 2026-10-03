"""M3 (W2 review): ``op_global_configs`` -- the one ``ConfigStore`` that
is NOT scoped to any project (sibling to ``op_projects``), built as the
foundation for W9's VLM endpoint registry / any future
``local_vlm:desired``-style global config axis.

No CRUD routes exist yet (W9's job); this covers the storage primitive
only: isolation from every per-project store, no project binding
required, and the guard's recognition of ``op_global_configs`` as a
legitimate global-scope index.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from curation._fake_config_opensearch import FakeConfigOpenSearch
from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.config_store.global_store import (
    ensure_global_configs_index,
    get_global_config_store,
    global_configs_index,
    reset_global_config_store,
)
from src.services.config_store.index import activate, get_activation, save_config
from src.services.config_store.store import ConfigStore, get_config_store, reset_config_stores


if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = pytest.mark.unbound


def _record(slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
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


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    reset_config_stores()
    reset_global_config_store()
    yield
    reset_config_stores()
    reset_global_config_store()


def test_get_global_config_store_requires_no_bind_project() -> None:
    """Contrast ``get_config_store()``, which raises ``ProjectNotBound``
    unbound -- the global store needs no project context at all."""
    store = get_global_config_store()
    assert isinstance(store, ConfigStore)
    assert store.index == global_configs_index()


def test_get_global_config_store_is_a_process_singleton_regardless_of_binding() -> None:
    """Calling it while a project happens to be bound has no effect on
    which store instance it returns."""
    store_unbound = get_global_config_store()
    with bind_project(_record('alpha')):
        store_bound = get_global_config_store()
    assert store_unbound is store_bound


def test_global_store_is_not_conflated_with_any_project_store() -> None:
    """The global store and a same-process project store are cached
    separately -- binding a project never hands back the global store,
    and vice versa."""
    global_store = get_global_config_store()
    with bind_project(_record('alpha')):
        project_store = get_config_store()
    assert global_store is not project_store
    assert global_store.index != project_store.index


@pytest.mark.asyncio
async def test_global_store_write_is_invisible_to_a_project_store_and_vice_versa() -> None:
    """A doc written to the global store never shows up through any
    project's own ``ConfigStore``, and a project's own doc never shows up
    in the global store -- both share one fake OpenSearch client."""
    client = FakeConfigOpenSearch()
    g_index = global_configs_index()
    project = _record('alpha')

    await save_config(
        client, g_index, kind='prompt_pack', name='global_vlm', body={}, expected_revision=None
    )
    await activate(
        client, g_index, axis='prompt_pack', name='global_vlm', revision=1, expected_active=None
    )

    with bind_project(project):
        from src.config import get_curation_config

        project_index = get_curation_config().configs_index
        await save_config(
            client,
            project_index,
            kind='prompt_pack',
            name='alpha_only',
            body={},
            expected_revision=None,
        )

    global_store = ConfigStore(index=g_index, mode='live', label='__global__')
    project_store = ConfigStore(index=project_index, mode='live', label='alpha')
    await global_store.refresh(client)
    await project_store.refresh(client)

    assert 'global_vlm' in global_store.current.packs
    assert 'alpha_only' not in global_store.current.packs
    assert 'alpha_only' in project_store.current.packs
    assert 'global_vlm' not in project_store.current.packs

    # The project's own store never sees the global axis's activation.
    with bind_project(project):
        project_pack_activation = await get_activation(client, project_index, 'prompt_pack')
    assert project_pack_activation is None


@pytest.mark.asyncio
async def test_ensure_global_configs_index_is_idempotent() -> None:
    client = FakeConfigOpenSearch()
    assert not await client.indices.exists(index=global_configs_index())
    await ensure_global_configs_index(client)
    assert await client.indices.exists(index=global_configs_index())
    # A second call (every process restart) must not raise or re-create.
    await ensure_global_configs_index(client)
    assert await client.indices.exists(index=global_configs_index())


def test_global_index_name_inside_the_project_namespace_is_refused(monkeypatch) -> None:
    from src.config.projects import project_index_prefix

    monkeypatch.setenv('OP_GLOBAL_CONFIGS_INDEX', f'{project_index_prefix()}shared__configs')
    with pytest.raises(ValueError, match='project index prefix'):
        global_configs_index()


def test_global_index_name_equal_to_the_registry_is_refused(monkeypatch) -> None:
    monkeypatch.setenv('OP_PROJECTS_INDEX', 'op_registry')
    monkeypatch.setenv('OP_GLOBAL_CONFIGS_INDEX', 'op_registry')
    with pytest.raises(ValueError, match='project registry'):
        global_configs_index()
