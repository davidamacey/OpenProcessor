"""The deployment-wide ``ConfigStore``: the one store that is NOT scoped to any project, the
foundation of the VLM endpoint registry. Isolation from every per-project store, no project
binding required, and the compare-and-set rule for "nothing active"."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.config_store import documents
from src.services.config_store.global_store import (
    get_global_config_store,
    reset_global_config_store,
)
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
    assert store.is_global


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
    assert global_store.label != project_store.label


@pytest.mark.asyncio
async def test_global_store_write_is_invisible_to_a_project_store_and_vice_versa() -> None:
    """An endpoint written to the deployment never shows up through a project's own store, and
    a project's own pack never shows up in the registry."""
    await documents.save_config(
        kind='vlm_endpoint', name='global_vlm', body={'base_url': 'http://v/v1', 'model': 'm'},
        expected_revision=None,
    )  # fmt: skip
    with bind_project(_record('alpha')):
        await documents.save_config(
            kind='prompt_pack', name='alpha_only', body={}, expected_revision=None
        )
        project_store = get_config_store()
        await project_store.refresh()
        detail = await documents.get_activation_detail('prompt_pack')

    global_store = get_global_config_store()
    await global_store.refresh()

    assert 'global_vlm' in global_store.current.vlm_endpoints
    assert not global_store.current.packs
    assert 'alpha_only' in project_store.current.packs
    assert not project_store.current.vlm_endpoints
    # the project's own store never sees an activation it did not make
    assert detail is None


@pytest.mark.asyncio
async def test_activate_treats_nothing_active_as_one_state_for_compare_and_set() -> None:
    """Never activated and explicitly off both mean "nothing active": a missing
    or null-ref ``expected_active`` is accepted for either; it still conflicts
    when something IS active."""
    off = {'name': None, 'revision': None}
    with bind_project(_record('alpha')):
        await documents.save_config(
            kind='open_vocab_set', name='a', body={}, expected_revision=None
        )

        # never activated: both spellings of "nothing" pass
        await documents.activate(axis='open_vocab', name='a', revision=1, expected_active=off)
        await documents.activate(
            axis='open_vocab',
            name=None,
            revision=None,
            expected_active={'name': 'a', 'revision': 1},
        )
        # explicitly off now: a missing expected_active is still "nothing"
        await documents.activate(axis='open_vocab', name='a', revision=1, expected_active=None)
        # something active: a missing or different expected_active conflicts
        for stale in (None, off, {'name': 'b', 'revision': 1}):
            with pytest.raises(documents.ActiveConflictError):
                await documents.activate(
                    axis='open_vocab', name='a', revision=1, expected_active=stale
                )
