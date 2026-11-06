"""Shared fakes and helpers for the project-clone tests that run the real clone flow
against an in-memory config index."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from config_store import memory_config

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.config_store.store import reset_config_stores


class Fake:
    """The OpenSearch client of the clone flow: only item counts and bulks are asked of it."""

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
    from src.services.config_store.documents import save_config

    del client
    with bind_project(rec):
        saved, _ = await save_config(kind=kind, name=name, body=body, expected_revision=expected)
    return saved


async def activate_config(client, rec, axis, name, revision, expected=None):
    from src.services.config_store.documents import Registered, activate

    del client
    # a name without a revision is a built-in: its body is stored as a system revision
    registered = (
        Registered({'class_system': name}, 'built-in') if name and revision is None else None
    )
    with bind_project(rec):
        return await activate(
            axis=axis,
            name=name,
            revision=revision,
            expected_active=expected,
            registered=registered,
        )


async def active_ref(rec, axis):
    """The project's activation of ``axis`` as ``{name, revision}`` (``None`` when never activated)."""
    from src.services.config_store.documents import read_activation

    read = await read_activation(rec.slug, axis)
    return None if read is None else {'name': read.name, 'revision': read.revision}


def patch_lifecycle(monkeypatch, source, target):
    from src.services.projects import lifecycle as lifecycle_mod, registry as registry_mod

    monkeypatch.setattr(lifecycle_mod, '_resolve_existing', AsyncMock(return_value=source))
    monkeypatch.setattr(lifecycle_mod, '_get_mutable_record', AsyncMock(return_value=target))
    # the revision bump after a clone writes the row; here it hands back the record unchanged
    monkeypatch.setattr('src.services.projects.store.run_repo', AsyncMock(return_value=target))
    monkeypatch.setattr('src.services.projects.records.record_from_row', lambda row: row)
    reg = MagicMock()
    reg.ensure_fresh = AsyncMock(return_value=None)
    monkeypatch.setattr(registry_mod, 'get_project_registry', lambda: reg)


def target_docs(client, rec):
    """Every stored document of the project (all revisions) and its activations, to compare
    before and after a refused clone."""
    del client
    memory = memory_config.current()
    pid = memory.project_id(rec.slug)
    docs = {
        f'{d.kind}:{d.name}': {n: r['body'] for n, r in d.revisions.items()}
        for d in memory.docs.values()
        if d.project_id == pid
    }
    active = {
        f'active:{axis}': (ref.name, ref.revision)
        for (project, axis), ref in memory.active.items()
        if project == pid
    }
    return {**docs, **active}


async def clone_into(client, source, target):
    from src.services.projects.clone import clone_settings_into

    return await clone_settings_into(
        client, slug=target.slug, from_slug=source.slug, axes=None, expected_revision=1
    )
