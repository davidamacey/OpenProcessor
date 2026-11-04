"""Create-only generation claims: one trainer replica runs a job, a stale holder is replaced by
exactly one reclaimer, and a replaced holder knows it.

Races use spawned processes; the claim files are real (``O_CREAT | O_EXCL`` on a shared dir).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import job_claim
import pytest


ROUNDS = 10
PROCS = 4


def _job(tmp_path: Path, name: str = 'j1') -> Path:
    path = tmp_path / f'{name}.job.json'
    path.write_text('{}')
    return path


def _reclaim(directory: str, barrier: Any, out: Any, rounds: int) -> None:
    for n in range(rounds):
        barrier.wait(timeout=60)
        out.put((n, job_claim.claim(Path(directory) / f'r{n}.job.json', stale_after_s=60.0)))


def test_a_single_trainer_claims_g1_runs_and_removes_its_claim(tmp_path: Path) -> None:
    job = _job(tmp_path)

    assert job_claim.claim(job) == 1
    assert (tmp_path / 'j1.claim.g1').exists()
    job_claim.release(job)

    assert list(tmp_path.glob('*.claim.*')) == []


def test_a_live_holder_blocks_a_second_claim(tmp_path: Path) -> None:
    job = _job(tmp_path)
    assert job_claim.claim(job) == 1

    assert job_claim.claim(job) is None
    assert not job_claim.superseded(job, 1)


def test_a_stale_holder_is_replaced_and_knows_it(tmp_path: Path) -> None:
    job = _job(tmp_path)
    assert job_claim.claim(job) == 1
    old = time.time() - 600
    os.utime(tmp_path / 'j1.claim.g1', (old, old))

    assert job_claim.claim(job) == 2

    assert job_claim.superseded(job, 1)
    assert not job_claim.superseded(job, 2)
    assert sorted(p.name for p in tmp_path.glob('*.claim.*')) == ['j1.claim.g2']


def test_exactly_one_of_four_reclaimers_wins_each_stale_claim(tmp_path: Path) -> None:
    for n in range(ROUNDS):
        job = _job(tmp_path, f'r{n}')
        assert job_claim.claim(job) == 1
        old = time.time() - 600
        os.utime(tmp_path / f'r{n}.claim.g1', (old, old))
    ctx = mp.get_context('spawn')
    barrier, out = ctx.Barrier(PROCS), ctx.Queue()
    procs = [
        ctx.Process(target=_reclaim, args=(str(tmp_path), barrier, out, ROUNDS))
        for _ in range(PROCS)
    ]
    for p in procs:
        p.start()
    results = [out.get(timeout=120) for _ in range(PROCS * ROUNDS)]
    for p in procs:
        p.join(timeout=30)

    for n in range(ROUNDS):
        won = [gen for r, gen in results if r == n and gen is not None]
        assert won == [2], (n, won)
        assert sorted(p.name for p in tmp_path.glob(f'r{n}.claim.*')) == [f'r{n}.claim.g2']


def _process_one(directory: str, barrier: Any, out: Any) -> None:
    import trainer

    ran: list[str] = []

    def run(spec: Any, generation: int = 1) -> None:
        ran.append(spec.job_id)
        time.sleep(1.0)

    trainer.parse_and_validate_job = lambda path: SimpleNamespace(job_id=path.name.split('.')[0])
    trainer.run_job = run
    barrier.wait(timeout=60)
    trainer.process_one(Path(directory))
    out.put(ran)


def test_two_process_one_calls_in_two_processes_run_the_job_once(tmp_path: Path) -> None:
    _job(tmp_path)
    ctx = mp.get_context('spawn')
    barrier, out = ctx.Barrier(2), ctx.Queue()
    procs = [ctx.Process(target=_process_one, args=(str(tmp_path), barrier, out)) for _ in range(2)]
    for p in procs:
        p.start()
    ran = sorted(job for _ in procs for job in out.get(timeout=120))
    for p in procs:
        p.join(timeout=30)

    assert ran == ['j1']


@pytest.mark.asyncio
async def test_the_api_create_collision_is_the_duplicate_error(tmp_path: Path) -> None:
    from src.services.training.job_files import _create_json

    target = tmp_path / 'a.job.json'
    await _create_json(target, {'job_id': 'a'})

    with pytest.raises(FileExistsError):
        await _create_json(target, {'job_id': 'a'})
    assert 'a' in target.read_text()
