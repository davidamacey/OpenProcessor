"""Training reserves cards by UUID: swapped CUDA numbering still names the same physical card."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fixtures.job_rows import make_project, new_job, sql

from src.db.repos import gpu_devices, gpu_reservations as reservations, jobs as job_repo
from src.db.session import sync_project_tx
from src.domain.gpu_order import host_id_at, parse_order, position_of
from tests.training._cards import HOST, card as _card


if TYPE_CHECKING:
    import uuid


pytestmark = [pytest.mark.db]

CARD_A, CARD_B = 'GPU-aaaa', 'GPU-bbbb'


@pytest.fixture
def world(pg):
    project = make_project(pg)
    for statement in ('DELETE FROM gpu_reservations', 'DELETE FROM gpu_devices'):
        sql(pg, statement)
    return pg, project


def _job(pg, project) -> uuid.UUID:
    from sqlalchemy.orm import Session

    with (
        Session(pg.engine('app'), expire_on_commit=False) as session,
        sync_project_tx(session, project),
    ):
        return job_repo.insert_queued(session, new_job(project)).job.id


def test_swapped_cuda_order_still_reserves_the_named_physical_card(world) -> None:
    from sqlalchemy.orm import Session

    pg, project = world
    # The trainer container is pinned to host cards 2,0 (device_ids order): inside it, host card 2
    # is NVML position 0 and host card 0 is position 1, so the CUDA/NVML number of each physical
    # card is the opposite of its host number. Reserving host card 0 must take card B.
    order = parse_order('2,0')
    with Session(pg.engine('app')) as session, sync_project_tx(session, project):
        gpu_devices.upsert(session, [_card(CARD_A, 0), _card(CARD_B, 1)])  # A=host 2, B=host 0
    job = _job(pg, project)

    with Session(pg.engine('app')) as session, sync_project_tx(session, project):
        wanted = gpu_devices.on_host_at(session, HOST, [position_of(order, 0)])
        assert [c.gpu_uuid for c in wanted] == [CARD_B]
        reservations.reserve(session, job, wanted)

    assert sql(pg, 'SELECT gpu_uuid, holder_job_id FROM gpu_reservations') == [(CARD_B, job)]
    assert host_id_at(order, 1) == 0  # the reverse map the API uses to report host numbers


def test_a_card_held_by_another_job_refuses_and_a_repeat_by_its_holder_is_a_no_op(world) -> None:
    from sqlalchemy.orm import Session

    pg, project = world
    first, second = _job(pg, project), _job(pg, project)
    card = _card(CARD_A, 0)
    with Session(pg.engine('app')) as session, sync_project_tx(session, project):
        reservations.reserve(session, first, [card])
        reservations.reserve(session, first, [card])  # a redelivered job
    with (
        pytest.raises(reservations.CardHeldError) as refused,
        Session(pg.engine('app')) as session,
        sync_project_tx(session, project),
    ):
        reservations.reserve(session, second, [card])
    assert refused.value.gpu_uuid == CARD_A


def test_release_frees_every_card_the_job_holds(world) -> None:
    from sqlalchemy.orm import Session

    pg, project = world
    job = _job(pg, project)
    with Session(pg.engine('app')) as session, sync_project_tx(session, project):
        reservations.reserve(session, job, [_card(CARD_A, 0), _card(CARD_B, 1)])
        assert reservations.release(session, job) == 2
    assert sql(pg, 'SELECT count(*) FROM gpu_reservations') == [(0,)]


def test_host_numbers_map_through_the_order() -> None:
    assert parse_order('') == ()
    assert parse_order('2, 0') == (2, 0)
    assert position_of((), 3) == 3  # unrestricted: every card at its host number
    assert position_of((0, 2), 2) == 1
    with pytest.raises(ValueError, match='not attached'):
        position_of((0, 2), 1)
    assert host_id_at((), 5) == 5
