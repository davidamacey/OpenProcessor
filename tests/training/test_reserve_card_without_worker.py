"""A card no GPU worker ever recorded can still be reserved (review RR-9)."""

from __future__ import annotations

import pytest
from fixtures.job_rows import make_project, new_job, sql
from sqlalchemy.orm import Session

from src.db.repos import gpu_reservations as reservations, jobs as job_repo
from src.db.session import sync_project_tx
from tests.training._cards import card as _card


pytestmark = [pytest.mark.db]


def test_a_reservation_for_a_card_with_no_worker_succeeds_and_records_the_card(pg) -> None:
    project = make_project(pg)
    sql(pg, 'DELETE FROM gpu_reservations')
    sql(pg, 'DELETE FROM gpu_devices')
    with (
        Session(pg.engine('app'), expire_on_commit=False) as session,
        sync_project_tx(session, project),
    ):
        job = job_repo.insert_queued(session, new_job(project)).job.id
    assert sql(pg, "SELECT count(*) FROM gpu_devices WHERE gpu_uuid = 'GPU-orphan'") == [(0,)]

    with Session(pg.engine('app')) as session, sync_project_tx(session, project):
        reservations.reserve(session, job, [_card('GPU-orphan', 3, host='no-worker-host')])

    assert sql(pg, "SELECT host FROM gpu_devices WHERE gpu_uuid = 'GPU-orphan'") == [
        ('no-worker-host',)
    ]
    assert sql(pg, 'SELECT gpu_uuid FROM gpu_reservations') == [('GPU-orphan',)]


def test_the_card_row_and_the_reservation_are_one_transaction(pg) -> None:
    project = make_project(pg)
    sql(pg, 'DELETE FROM gpu_reservations')
    sql(pg, 'DELETE FROM gpu_devices')
    with (
        Session(pg.engine('app'), expire_on_commit=False) as session,
        sync_project_tx(session, project),
    ):
        job = job_repo.insert_queued(session, new_job(project)).job.id
    session = Session(pg.engine('app'))
    try:
        with sync_project_tx(session, project):
            reservations.reserve(session, job, [_card('GPU-rb', 0)])
            raise RuntimeError('the caller fails after reserving')
    except RuntimeError:
        pass
    finally:
        session.close()
    assert sql(pg, 'SELECT count(*) FROM gpu_devices') == [(0,)]
    assert sql(pg, 'SELECT count(*) FROM gpu_reservations') == [(0,)]
