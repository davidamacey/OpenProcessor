"""``open_vocab_drain`` through the job path: start, claim, drain every pending image, finish."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from curation.open_vocab_fixtures import (
    FakeSegmenter,
    StatefulRegistry,
    cand,
    ingested_world,
    make_set,
)
from curation.reprocess_fixtures import docs, images_index
from fixtures.job_rows import sql
from projects.conftest import seed_default_project

from src.config.project_context import bind_project
from src.services.curation import open_vocab_ingest
from src.services.curation.open_vocab_ingest import start_drain
from src.services.projects.registry import ProjectRegistry, set_project_registry


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_registry')]


@pytest.fixture
def worker_app(queue_env):
    from src.tasks.app import app

    app.loader.import_default_modules()
    return app


def test_the_registered_task_drains_every_pending_image_once(
    pg, job_env, producer, wakes, kinds, worker_app, queue_env, tmp_path, monkeypatch
) -> None:
    import src.config.curation_accessors as curation_mod
    from src.services.projects import capacity as capacity_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(ProjectRegistry())
    registry = StatefulRegistry()
    monkeypatch.setattr('src.services.curation.open_vocab_run.get_class_registry', lambda: registry)

    # ingest resolves the bound project's row (the content key), so seed it before ingesting
    record = asyncio.run(seed_default_project())
    with bind_project(record):
        fake, service, ids = asyncio.run(ingested_world(tmp_path, monkeypatch, n=3))
    for image_id in ids:
        fake.docs(images_index())[image_id]['open_vocab_status'] = 'pending'
    seg = FakeSegmenter()
    seg.default = [cand()]

    async def active() -> Any:
        return make_set(run_on_ingest=True), 2

    async def get_service(*_a: Any) -> Any:
        return service

    monkeypatch.setattr(open_vocab_ingest, 'current_active_set', active)
    monkeypatch.setattr(
        'src.services.projects.guard.make_script_opensearch', lambda *_a, **_k: fake
    )
    monkeypatch.setattr(
        'src.services.curation.ingest_service_factory.get_ingest_service', get_service
    )
    monkeypatch.setattr('src.services.detection.segmenter_http.segment_image_http', seg)

    # the fake world's indexes are the default project's: the job runs as that project
    with bind_project(record):
        assert asyncio.run(start_drain()) is True
    [(job_id,)] = sql(pg, "SELECT id FROM jobs WHERE kind = 'open_vocab_drain'")
    assert producer.sent() == [str(job_id)]

    async def run() -> None:
        outcome = await asyncio.to_thread(
            worker_app.tasks['op.families_c.open_vocab_drain'].apply,
            args=[str(job_id)],
            task_id=str(job_id),
        )
        assert outcome.successful(), outcome.traceback

    asyncio.run(run())

    [(status, result)] = sql(pg, 'SELECT status, result FROM jobs WHERE id = %s', job_id)
    assert (status, result) == ('completed', {'drained': 3, 'segmenter_down': 0})
    assert {fake.docs(images_index())[i]['open_vocab_status'] for i in ids} == {'done'}
    assert len(seg.calls) == 3
    assert docs(fake) != {}
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)
