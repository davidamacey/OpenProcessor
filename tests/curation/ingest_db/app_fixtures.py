"""The real curation routes over PostgreSQL for the ingest-run and ingest-source tests.

Only the project binding (by slug), the OpenSearch dependency and the ingest-service check are
replaced: the routes, the job service, the queue repositories and the wire mapping are
production code. Imported by name into the test modules that use it."""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import pytest_asyncio
from _fake_object_store import FakeObjectStore
from fastapi import Depends, FastAPI
from fixtures.job_rows import make_project

from src.config.curation_accessors import base_curation_config
from src.config.project_context import set_bound_project
from src.config.projects import ProjectRecord, resources_for_new
from src.core.db_session import install_database_error_handlers
from src.routers.curation._project_deps import install_project_exception_handlers


if TYPE_CHECKING:
    import uuid
    from collections.abc import AsyncIterator
    from pathlib import Path

BUCKET_BODY = {'bucket': 'imgs', 'prefix': 'in/'}


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
def served_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from src.services.curation import image_serving

    root = tmp_path / 'served'
    root.mkdir()
    monkeypatch.setattr(
        image_serving,
        '_configured_roots',
        lambda config=None: (root.resolve(),),  # noqa: ARG005
    )
    return root


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeObjectStore:
    from src.services.curation.bulk_ingest import source_scan

    fake = FakeObjectStore()
    for i in range(3):
        fake.put('imgs', f'in/a{i}.jpg', b'x%d' % i)
    monkeypatch.setenv('OP_S3_ALLOWED', 'imgs/in/')
    monkeypatch.setattr(source_scan, 'get_object_store', lambda: fake)
    return fake


@pytest.fixture
def app(pg, pg_env, monkeypatch: pytest.MonkeyPatch):
    from src.routers import curation as curation_package
    from src.routers.curation import ingest_sources
    from src.routers.curation._common import _raw_opensearch_dep, _registry_dep
    from src.routers.curation._project_deps import bind_path_project

    records: dict[str, ProjectRecord] = {}
    application = FastAPI()

    async def bind(project: str) -> None:
        set_bound_project(records[project])

    async def no_service(*_a: Any, **_k: Any) -> None:
        return None

    application.dependency_overrides[bind_path_project] = bind
    application.dependency_overrides[_raw_opensearch_dep] = lambda: None
    application.dependency_overrides[_registry_dep] = lambda: None
    monkeypatch.setattr(ingest_sources, 'get_ingest_service', no_service)
    application.include_router(
        curation_package.router,
        prefix='/curation/projects/{project}',
        dependencies=[Depends(bind_path_project)],
    )
    install_database_error_handlers(application)
    install_project_exception_handlers(application)
    application.state.records = records
    return application


@pytest_asyncio.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:
    from src.db.engine import dispose_engines

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as http:
        yield http
    await dispose_engines()


def project(app, pg, slug: str) -> uuid.UUID:
    pid = make_project(pg, slug)
    app.state.records[slug] = record_for(pid, slug)
    return pid


def url(slug: str, tail: str) -> str:
    return f'/curation/projects/{slug}/ingest/{tail}'
