"""The bound project carries its database id and storage prefix from the row."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from src.config.project_context import (
    ProjectHasNoIdError,
    bind_project,
    current_project_id,
    current_storage_prefix,
)
from src.config.projects import new_project_record
from src.services.projects.registry import ProjectRegistry

from .conftest import aseed_project


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_registry')]


def _loaded(slug: str):
    async def _run():
        await aseed_project(slug)
        registry = ProjectRegistry()
        await registry.refresh_strict()
        return registry.get(slug)

    return asyncio.run(_run())


def test_a_bound_registry_record_exposes_the_database_id_and_storage_prefix() -> None:
    record = _loaded('ctxproj')
    assert record is not None
    assert isinstance(record.id, uuid.UUID)
    with bind_project(record):
        assert current_project_id() == record.id
        assert current_storage_prefix() == 'projects/ctxproj'


def test_a_record_built_without_a_database_has_no_id_and_fails_closed() -> None:
    from src.config.curation_accessors import base_curation_config

    record = new_project_record('nodb', base_curation_config())
    with bind_project(record):
        with pytest.raises(ProjectHasNoIdError):
            current_project_id()
        with pytest.raises(ProjectHasNoIdError):
            current_storage_prefix()
