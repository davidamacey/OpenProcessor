"""ST-4: keep-last retention for finished training runs + bake-off output.

Locks the planning logic in isolation: a promoted run and a
running/queued run must survive regardless of age; the oldest terminal
run beyond keep_last is planned; bake-off output prunes are pure
keep-last with no pin logic.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.services.training.run_retention import (
    apply_bakeoff_out_prune,
    apply_run_prune,
    plan_bakeoff_out_prune,
    plan_run_prune,
)


if TYPE_CHECKING:
    from pathlib import Path


def _write_status(jobs_dir: Path, job_id: str, state: str) -> None:
    (jobs_dir / f'{job_id}.job.json').write_text(json.dumps({'job_id': job_id}))
    (jobs_dir / f'{job_id}.status.json').write_text(json.dumps({'job_id': job_id, 'state': state}))
    (jobs_dir / f'{job_id}.manifest.json').write_text(json.dumps({'job_id': job_id}))


def _job_ids(n: int) -> list[str]:
    # ISO-timestamp-prefixed, strictly increasing (oldest first).
    return [f'202601{(i // 24) + 1:02d}T{i % 24:02d}0000Z-run' for i in range(n)]


class TestPlanRunPrune:
    def test_keeps_last_n_and_never_prunes_non_terminal(self, tmp_path: Path) -> None:
        jobs_dir = tmp_path / 'jobs'
        jobs_dir.mkdir()
        model_repo = tmp_path / 'models'
        bakeoff_jobs_dir = tmp_path / 'bakeoff_jobs'

        ids = _job_ids(10)
        for job_id in ids[:8]:
            _write_status(jobs_dir, job_id, 'finished')
        # Two old-looking ids that are still active -- must never be planned.
        _write_status(jobs_dir, ids[8], 'running')
        _write_status(jobs_dir, ids[9], 'queued')

        plan = plan_run_prune(
            jobs_dir=jobs_dir, model_repo=model_repo, bakeoff_jobs_dir=bakeoff_jobs_dir, keep_last=3
        )
        for job_id in (ids[8], ids[9]):
            assert job_id not in plan

    def test_promoted_run_survives(self, tmp_path: Path) -> None:
        jobs_dir = tmp_path / 'jobs'
        jobs_dir.mkdir()
        model_repo = tmp_path / 'models'
        (model_repo / 'my_model').mkdir(parents=True)
        bakeoff_jobs_dir = tmp_path / 'bakeoff_jobs'

        ids = _job_ids(6)
        for job_id in ids:
            _write_status(jobs_dir, job_id, 'finished')
        promoted_id = ids[0]  # the OLDEST -- would otherwise be pruned first
        (model_repo / 'my_model' / 'promote.json').write_text(json.dumps({'job_id': promoted_id}))

        plan = plan_run_prune(
            jobs_dir=jobs_dir, model_repo=model_repo, bakeoff_jobs_dir=bakeoff_jobs_dir, keep_last=2
        )
        assert promoted_id not in plan
        assert ids[1] in plan  # the next-oldest, unprotected, IS planned

    def test_bakeoff_active_run_survives(self, tmp_path: Path) -> None:
        jobs_dir = tmp_path / 'jobs'
        jobs_dir.mkdir()
        model_repo = tmp_path / 'models'
        bakeoff_jobs_dir = tmp_path / 'bakeoff_jobs'
        bakeoff_jobs_dir.mkdir()

        ids = _job_ids(6)
        for job_id in ids:
            _write_status(jobs_dir, job_id, 'finished')
        active_id = ids[0]
        (bakeoff_jobs_dir / 'b1.job.json').write_text(
            json.dumps({'models': [{'run_id': active_id}]})
        )

        plan = plan_run_prune(
            jobs_dir=jobs_dir, model_repo=model_repo, bakeoff_jobs_dir=bakeoff_jobs_dir, keep_last=2
        )
        assert active_id not in plan
        assert ids[1] in plan

    def test_oldest_terminal_run_beyond_keep_last_is_planned(self, tmp_path: Path) -> None:
        jobs_dir = tmp_path / 'jobs'
        jobs_dir.mkdir()
        model_repo = tmp_path / 'models'
        bakeoff_jobs_dir = tmp_path / 'bakeoff_jobs'

        ids = _job_ids(5)
        for job_id in ids:
            _write_status(jobs_dir, job_id, 'finished')

        plan = plan_run_prune(
            jobs_dir=jobs_dir, model_repo=model_repo, bakeoff_jobs_dir=bakeoff_jobs_dir, keep_last=3
        )
        assert set(plan) == {ids[0], ids[1]}

    def test_keep_last_zero_keeps_everything(self, tmp_path: Path) -> None:
        jobs_dir = tmp_path / 'jobs'
        jobs_dir.mkdir()
        for job_id in _job_ids(5):
            _write_status(jobs_dir, job_id, 'finished')
        plan = plan_run_prune(
            jobs_dir=jobs_dir,
            model_repo=tmp_path / 'models',
            bakeoff_jobs_dir=tmp_path / 'bakeoff_jobs',
            keep_last=0,
        )
        assert plan == []


class TestApplyRunPrune:
    def test_removes_every_job_state_file(self, tmp_path: Path) -> None:
        jobs_dir = tmp_path / 'jobs'
        jobs_dir.mkdir()
        job_id = _job_ids(1)[0]
        _write_status(jobs_dir, job_id, 'finished')
        (jobs_dir / f'{job_id}.run.log').write_text('log')

        result = apply_run_prune([job_id], jobs_dir=jobs_dir)
        assert result['removed_runs'] == 1
        assert not (jobs_dir / f'{job_id}.job.json').exists()
        assert not (jobs_dir / f'{job_id}.status.json').exists()
        assert not (jobs_dir / f'{job_id}.run.log').exists()


class TestBakeoffOutPrune:
    def test_keeps_last_n_by_name(self, tmp_path: Path) -> None:
        out_dir = tmp_path / 'bakeoff_out'
        out_dir.mkdir()
        names = [f'2026010{i}T000000Z-job' for i in range(1, 6)]
        for name in names:
            (out_dir / name).mkdir()

        plan = plan_bakeoff_out_prune(out_dir, keep_last=2)
        assert {p.name for p in plan} == set(names[:3])

        result = apply_bakeoff_out_prune(plan)
        assert result['removed'] == 3
        assert len(list(out_dir.iterdir())) == 2
