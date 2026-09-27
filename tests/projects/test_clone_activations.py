"""Glue G1 (projects_plan.md sec 11 W2): CLONEABLE_AXES gains
``activations`` -- cloning copies the source project's active
prompt-pack/region-profile config bodies plus the activation itself
into the target project's own (isolated) configs index."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from curation._fake_config_opensearch import FakeConfigOpenSearch

from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.routers.curation._project_models import CLONEABLE_AXES
from src.services.config_store.index import get_activation
from src.services.config_store.store import reset_config_stores
from src.services.projects.clone import _apply_clone


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
    yield
    reset_config_stores()


def test_activations_is_cloneable() -> None:
    assert 'activations' in CLONEABLE_AXES


@pytest.mark.asyncio
async def test_clone_activations_copies_active_pack_and_profile() -> None:
    client = FakeConfigOpenSearch()
    source = _record('alpha')
    target = _record('beta')

    from src.services.config_store.index import activate, save_config

    with bind_project(source):
        from src.config import get_curation_config

        idx = get_curation_config().configs_index
        pack_doc = await save_config(
            client,
            idx,
            kind='prompt_pack',
            name='wheel',
            body={'class_system': 'wheel-pack'},
            expected_revision=None,
        )
        await activate(
            client,
            idx,
            axis='prompt_pack',
            name='wheel',
            revision=pack_doc['revision'],
            expected_active=None,
        )
        profile_doc = await save_config(
            client,
            idx,
            kind='region_profile',
            name='wheel_profile',
            body={'detector_model': 'wheel_v1'},
            expected_revision=None,
        )
        await activate(
            client,
            idx,
            axis='detection_profile',
            name='wheel_profile',
            revision=profile_doc['revision'],
            expected_active=None,
        )

    await _apply_clone(client, target_record=target, source=source, axes=['activations'])

    with bind_project(target):
        from src.config import get_curation_config

        target_idx = get_curation_config().configs_index
        pack_activation = await get_activation(client, target_idx, 'prompt_pack')
        profile_activation = await get_activation(client, target_idx, 'detection_profile')

    assert pack_activation is not None
    assert pack_activation['name'] == 'wheel'
    assert profile_activation is not None
    assert profile_activation['name'] == 'wheel_profile'

    # Source untouched (still its own activation, never overwritten).
    with bind_project(source):
        from src.config import get_curation_config

        source_idx = get_curation_config().configs_index
        source_activation = await get_activation(client, source_idx, 'prompt_pack')
    assert source_activation is not None
    assert source_activation['name'] == 'wheel'


@pytest.mark.asyncio
async def test_clone_activations_is_a_noop_when_source_has_none() -> None:
    client = FakeConfigOpenSearch()
    source = _record('gamma')
    target = _record('delta')

    # No panic, no write, when the source axis was never activated.
    await _apply_clone(client, target_record=target, source=source, axes=['activations'])

    with bind_project(target):
        from src.config import get_curation_config

        idx = get_curation_config().configs_index
        assert await get_activation(client, idx, 'prompt_pack') is None
        assert await get_activation(client, idx, 'detection_profile') is None
