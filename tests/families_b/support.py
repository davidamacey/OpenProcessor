"""The real curation routes over a migrated database, for the family-B start/cancel tests.

Only the project binding and the OpenSearch dependency are replaced; ``start_job`` runs for real
against the ``pg`` database with the fake producer of ``job_fixtures``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import pytest_asyncio
from fastapi import Depends, FastAPI
from fixtures.job_rows import make_project

from src.config.curation_accessors import base_curation_config
from src.config.project_context import set_bound_project
from src.config.projects import ProjectRecord, resources_for_new
from src.core.db_session import install_database_error_handlers
from src.routers.curation._project_deps import install_project_exception_handlers


if TYPE_CHECKING:
    import uuid


def record_for(project: uuid.UUID, slug: str) -> ProjectRecord:
    now = dt.datetime.now(dt.UTC).isoformat()
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
    return dataclasses.replace(record, id=project)


@pytest.fixture
def app(pg, pg_env, job_env, producer, wakes, monkeypatch):
    from src.services.classes import registry as registry_module

    monkeypatch.setattr(registry_module, '_registries', {})  # the real, PostgreSQL registry
    monkeypatch.setattr(registry_module, '_project_ids', {})
    from src.core.dependencies import get_curation_opensearch
    from src.routers import curation as curation_package
    from src.routers.curation._project_deps import bind_path_project

    records: dict[str, ProjectRecord] = {}
    application = FastAPI()

    async def bind(project: str) -> None:
        set_bound_project(records[project])

    async def opensearch() -> Any:
        return application.state.opensearch

    application.dependency_overrides[bind_path_project] = bind
    application.dependency_overrides[get_curation_opensearch] = opensearch
    application.include_router(
        curation_package.router,
        prefix='/curation/projects/{project}',
        dependencies=[Depends(bind_path_project)],
    )
    install_database_error_handlers(application)
    install_project_exception_handlers(application)
    application.state.records = records
    application.state.opensearch = None
    return application


@pytest_asyncio.fixture
async def client(app):
    from src.db.engine import dispose_engines

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as http:
        yield http
    await dispose_engines()


def project(app, pg, slug: str | None = None) -> tuple[str, uuid.UUID]:
    pid = make_project(pg, slug)
    from fixtures.job_rows import sql

    [(real,)] = sql(pg, 'SELECT slug FROM projects WHERE id = %s', pid)
    app.state.records[real] = record_for(pid, real)
    return real, pid


@pytest.fixture
def queue_env(monkeypatch, tmp_path):
    """A valid queue URL and password file, so a lazily loaded config can resolve."""
    password = tmp_path / 'queue_password'
    password.write_text('test-password\n')
    monkeypatch.setenv('OP_QUEUE_URL', 'redis://valkey:6379/0')
    monkeypatch.setenv('OP_QUEUE_PASSWORD_FILE', str(password))


@pytest.fixture
def worker_app(queue_env):
    from src.tasks.app import app

    app.loader.import_default_modules()
    return app


def bind_worker_to(monkeypatch, app, opensearch) -> None:
    """The worker side of a handler: its project registry and its OpenSearch client."""

    class _Registry:
        async def lookup(self, slug):
            return app.state.records[slug]

    monkeypatch.setattr('src.services.projects.registry.get_project_registry', _Registry)
    monkeypatch.setattr(
        'src.services.projects.guard.make_script_opensearch', lambda *_a, **_k: opensearch
    )
