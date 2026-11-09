"""The reprocess job's run never writes a locked item (4A-1)."""

from __future__ import annotations

import copy
from pathlib import Path  # noqa: TC003 - pytest fixture annotation

import pytest

from curation.jobs_a.ctx import FakeCtx
from curation.reprocess_fixtures import FakeTriton, docs, jpeg_bytes, make_fake, servable_root
from src.db.session import async_session_factory
from src.services.curation import reprocess_job


pytestmark = pytest.mark.db


@pytest.mark.asyncio
async def test_the_job_entry_point_never_writes_a_locked_item_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pg, job_env, producer, wakes
) -> None:
    from types import SimpleNamespace

    from fixtures.job_rows import make_project

    from curation.reprocess_fixtures import make_service as make

    project = make_project(pg, 'default')  # the ingest below publishes events to it
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes())
    fake = make_fake([])
    triton = FakeTriton([(0.05, 0.05, 0.6, 0.6, 0.9, 1)])
    service = make(fake, triton)
    res = await service.ingest_one(path.read_bytes(), str(path))
    crop_id = next(iter(docs(fake)))
    docs(fake)[crop_id].update(
        {'class_source': 'human', 'label_source': 'human', 'class_id': 0, 'class_name': 'gadget',
         'class_validated': True}
    )  # fmt: skip
    before = copy.deepcopy(docs(fake)[crop_id])
    triton.detections = []  # nothing found: a stale-removal pass

    monkeypatch.setattr(
        reprocess_job,
        'get_curation_config',
        lambda: SimpleNamespace(items_index='i', images_index='m'),
    )
    async with async_session_factory()() as session:
        job = await reprocess_job.start(
            session, project, request={}, scopes=['detect'], image_ids=[res.image_id]
        )
    result, with_errors = await reprocess_job.run_reprocess(
        FakeCtx(project, job.id, job.params), fake, service
    )

    assert with_errors is False
    assert result['results'][0]['locked_skipped'] == 1
    after = docs(fake)[crop_id]
    assert {k for k in after if after[k] != before.get(k)} == set()
