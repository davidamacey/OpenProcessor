"""A reprocess job that changes a vector queues the changed items and images for near-duplicate
rescoring: one ``near_dup_pending`` row per level, reason ``vector_changed``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import psycopg
import pytest
from _records_world import make_project_row
from fixtures.job_fixtures import job_env, producer, wakes  # noqa: F401

from curation.jobs_a.ctx import FakeCtx
from curation.reprocess_fixtures import (
    FakePE,
    FakeTriton,
    images_index,
    item,
    jpeg_bytes,
    make_fake,
    make_service,
    servable_root,
)
from src.config.project_context import current_project
from src.db.session import async_session_factory
from src.services.curation import reprocess_job
from src.services.curation.reprocess_embed_spec import EmbedSpec


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.db


@pytest.mark.asyncio
async def test_a_reprocess_that_changes_a_vector_adds_one_pending_row_per_level(
    job_env: Any,  # noqa: F811 - the fixture is requested by name
    producer: Any,  # noqa: F811 - the fixture is requested by name
    wakes: Any,  # noqa: F811 - the fixture is requested by name
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = make_project_row(job_env, current_project().record.slug)
    monkeypatch.setattr(current_project().record, 'id', project, raising=False)
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes())
    fake = make_fake(
        [item('car', image_id='img-1', image_path=str(path), bbox_norm=(0.1, 0.1, 0.5, 0.5))],
        [{'image_id': 'img-1', 'image_path': str(path), 'pe_embedding': [0.5, 0.5, 0.5]}],
    )
    service = make_service(fake, FakeTriton([]), FakePE())
    cfg = reprocess_job.get_curation_config()
    assert images_index() == cfg.images_index

    async with async_session_factory()() as session:
        job = await reprocess_job.start(
            session,
            project,
            request={},
            scopes=['embed'],
            image_ids=['img-1'],
            embed=EmbedSpec(crop_ids=None, only_missing=False, parts=['crop', 'frame']),
        )
    ctx = FakeCtx(project, job.id, job.params)

    await reprocess_job.run_reprocess(ctx, fake, service)

    with psycopg.connect(job_env.conninfo('super')) as conn:
        rows = conn.execute(
            'SELECT level, member_id, reason FROM near_dup_pending WHERE project_id = %s '
            'ORDER BY level',
            (project,),
        ).fetchall()
    assert rows == [('image', 'img-1', 'vector_changed'), ('item', 'car', 'vector_changed')]
