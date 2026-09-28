"""Reviewer probe for B2 (place under tests/projects/). Asserts the correct
behavior; on branch HEAD it raises 409 target_not_empty."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from curation._fake_config_opensearch import FakeConfigOpenSearch

from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.config_store.store import reset_config_stores
from src.services.projects.clone import _apply_clone


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
def _reset():
    reset_config_stores()
    yield
    reset_config_stores()


@pytest.mark.asyncio
async def test_prompt_packs_plus_activations_axes_together() -> None:
    client = FakeConfigOpenSearch()
    source = _record('alpha')
    target = _record('beta')
    from src.services.config_store.index import activate, save_config

    with bind_project(source):
        from src.config import get_curation_config

        idx = get_curation_config().configs_index
        doc = await save_config(
            client,
            idx,
            kind='prompt_pack',
            name='wheel',
            body={'class_system': 'x'},
            expected_revision=None,
        )
        await activate(
            client,
            idx,
            axis='prompt_pack',
            name='wheel',
            revision=doc['revision'],
            expected_active=None,
        )
    await _apply_clone(
        client, target_record=target, source=source, axes=['prompt_packs', 'activations']
    )
