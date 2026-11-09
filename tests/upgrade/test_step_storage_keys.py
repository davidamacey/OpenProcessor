"""Rows written with absolute results paths or ``pytorch_models/<file>`` keys get the relative
keys; no file moves; running it again changes nothing."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from src.db.session import deployment_tx
from src.services.upgrade.from_060.steps import storage_keys


if TYPE_CHECKING:
    from tests.upgrade._v060 import V060Install

pytestmark = [pytest.mark.db, pytest.mark.asyncio]


def _seed(clean_db) -> tuple[str, uuid.UUID]:
    job = uuid.uuid4()
    with clean_db.engine('super').begin() as conn:
        project = conn.execute(text("SELECT id FROM projects WHERE slug = 'default'")).scalar_one()
        conn.execute(
            text(
                'INSERT INTO jobs (id, kind, queue, status, project_id, max_queue_wait, requested_by) '
                "VALUES (:j, 'bakeoff', 'gpu_eval', 'queued', :p, interval '1 hour', 'test')"
            ),
            {'j': job, 'p': project},
        )
        conn.execute(
            text(
                'INSERT INTO bakeoff_runs (job_id, project_id, eval_dataset_key, models, '
                'results_storage_key) VALUES (:j, :p, :k, :m, :r)'
            ),
            {
                'j': job,
                'p': project,
                'k': 'export:e1',
                'm': '[]',
                'r': '/state/projects/default/bakeoff_jobs/out/20260101T000000Z',
            },
        )
        conn.execute(
            text(
                'INSERT INTO models (project_id, name, version, origin, family, task, labels, '
                "artifact_storage_key, created_by) VALUES (:p, 'up', 1, 'uploaded', 'yolo', "
                "'detect', ARRAY['car'], 'pytorch_models/car.pt', 'test')"
            ),
            {'p': project},
        )
    return 'default', job


async def test_old_keys_are_rewritten_once(
    make_runner, clean_db, owner_sessions, v060: V060Install
) -> None:
    await make_runner().run()
    _, job = _seed(clean_db)

    async def rewrite() -> tuple[int, int]:
        async with owner_sessions() as session, deployment_tx(session, 'test storage keys'):
            return (
                await storage_keys.rewrite_bakeoff_results(session),
                await storage_keys.rewrite_uploaded_model_keys(session),
            )

    assert await rewrite() == (1, 1)
    with clean_db.engine('super').connect() as conn:
        assert (
            conn.execute(
                text('SELECT results_storage_key FROM bakeoff_runs WHERE job_id = :j'), {'j': job}
            ).scalar_one()
            == 'bakeoff_results/default/20260101T000000Z'
        )
        assert (
            conn.execute(
                text("SELECT artifact_storage_key FROM models WHERE name = 'up'")
            ).scalar_one()
            == 'models/pytorch_models/car.pt'
        )
    assert await rewrite() == (0, 0)
