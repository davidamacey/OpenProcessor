"""2E-3 (R-25): a 1,000,000-row ledger loads through COPY in one transaction; the time is recorded."""

from __future__ import annotations

import datetime as dt
import time
import uuid

import pytest
from sqlalchemy import text

from src.db.models import DatasetImport, Job, Project, ProjectSlug
from src.db.session import deployment_tx
from src.services.upgrade.from_060 import ledger_copy


pytestmark = pytest.mark.db

ROWS = 1_000_000


def _images(import_id: uuid.UUID, n: int):
    for i in range(n):
        yield (import_id, f'images/{i:07d}.jpg', f'img{i}', [f'c{i}'], 'written', None)


@pytest.mark.asyncio
async def test_a_million_ledger_rows_load_in_one_transaction(
    owner_sessions, clean_db, capsys
) -> None:
    project, job, imp = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    now = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    async with owner_sessions() as session, deployment_tx(session, 'test ledger'):
        session.add(ProjectSlug(slug='big'))
        await session.flush()
        session.add(
            Project(
                id=project, slug='big', display_name='Big', status='active', index_prefix='op_prj_big__',
                storage_prefix='projects/big', model_prefix='big__', mlflow_experiment='openprocessor-big',
                created_by='test',
            )
        )  # fmt: skip
        await session.flush()
        session.add(
            Job(
                id=job, project_id=project, kind='dataset_import', queue='jobs', status='completed',
                max_queue_wait=dt.timedelta(hours=1), requested_by='test', created_at=now, finished_at=now,
            )
        )  # fmt: skip
        await session.flush()
        session.add(
            DatasetImport(
                id=imp,
                project_id=project,
                job_id=job,
                source_key='source/x',
                format='coco',
                mapping={},
                options={},
            )
        )
        await session.flush()
        started = time.perf_counter()
        copied = await ledger_copy.copy_import_images(session, _images(imp, ROWS))
        elapsed = time.perf_counter() - started
    assert copied == ROWS
    with clean_db.engine('super').connect() as conn:
        assert conn.execute(text('SELECT count(*) FROM dataset_import_images')).scalar_one() == ROWS
    with capsys.disabled():
        print(f'\nCOPY of {ROWS:,} dataset_import_images rows in one transaction: {elapsed:.1f} s')
