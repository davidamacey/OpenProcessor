"""Shared fakes and helpers for the project-clone tests that run the real clone flow
against an in-memory config index."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from curation._fake_config_opensearch import FakeConfigOpenSearch

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.config_store.store import reset_config_stores


class Fake(FakeConfigOpenSearch):
    async def count(self, index: str, body: Any = None) -> dict[str, Any]:  # noqa: ARG002
        return {'count': 0}

    async def bulk(self, body: list[Any], refresh: Any = False) -> dict[str, Any]:  # noqa: ARG002
        return {'errors': False, 'items': []}


def record(slug: str) -> ProjectRecord:
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
def isolated_state(tmp_path, monkeypatch):
    import src.config.curation_accessors as curation_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    curation_mod._default_curation_config = None
    reset_config_stores()
    yield
    reset_config_stores()
    curation_mod._default_curation_config = None


async def save(client, rec, kind, name, body, expected=None):
    from src.config import get_curation_config
    from src.services.config_store.index import save_config

    with bind_project(rec):
        return await save_config(
            client,
            get_curation_config().configs_index,
            kind=kind,
            name=name,
            body=body,
            expected_revision=expected,
        )


async def activate_config(client, rec, axis, name, revision, expected=None):
    from src.config import get_curation_config
    from src.services.config_store.index import activate

    with bind_project(rec):
        return await activate(
            client,
            get_curation_config().configs_index,
            axis=axis,
            name=name,
            revision=revision,
            expected_active=expected,
        )


def patch_lifecycle(monkeypatch, source, target):
    from src.services.projects import lifecycle as lifecycle_mod, registry as registry_mod

    monkeypatch.setattr(lifecycle_mod, '_resolve_existing', AsyncMock(return_value=source))
    monkeypatch.setattr(
        lifecycle_mod, '_get_mutable_record', AsyncMock(return_value=(target, 1, 1))
    )
    monkeypatch.setattr(lifecycle_mod, 'write_record', AsyncMock(return_value=None))
    reg = MagicMock()
    reg.ensure_fresh = AsyncMock(return_value=None)
    monkeypatch.setattr(registry_mod, 'get_project_registry', lambda: reg)


def target_docs(client, rec):
    from src.config import get_curation_config

    with bind_project(rec):
        idx = get_curation_config().configs_index
    return dict(client._docs.get(idx, {}))


async def clone_into(client, source, target):
    from src.services.projects.clone import clone_settings_into

    return await clone_settings_into(
        client, slug=target.slug, from_slug=source.slug, axes=None, expected_revision=1
    )
