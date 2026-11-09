"""ST-4: keep-last retention for finished training runs + bake-off output.

Locks the planning logic in isolation: a promoted run and a
running/queued run must survive regardless of age; the oldest terminal
run beyond keep_last is planned; bake-off output prunes are pure
keep-last with no pin logic. The database side (listing and deleting rows) is in
``tests/training/test_run_retention_pg.py``.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from src.services.training.run_retention import (
    apply_bakeoff_out_prune,
    plan_bakeoff_out_prune,
    plan_run_prune,
)


if TYPE_CHECKING:
    from pathlib import Path


def _runs(*statuses: str) -> list[tuple[uuid.UUID, str]]:
    """Runs newest first, the order ``list_runs`` answers."""
    return [(uuid.uuid4(), status) for status in statuses]


class TestPlanRunPrune:
    def test_keeps_last_n_and_never_prunes_non_terminal(self) -> None:
        runs = _runs(
            'queued', 'running', 'paused', 'cancelling', 'interrupted', 'completed', 'failed'
        )
        plan = plan_run_prune(runs, promoted_job_ids=set(), bakeoff_run_ids=set(), keep_last=1)
        assert plan == [runs[5][0], runs[6][0]]

    def test_promoted_run_survives(self) -> None:
        runs = _runs('completed', 'completed', 'completed', 'completed')
        promoted = str(runs[3][0])  # the OLDEST -- would otherwise be pruned first
        plan = plan_run_prune(runs, promoted_job_ids={promoted}, bakeoff_run_ids=set(), keep_last=1)
        assert plan == [runs[1][0], runs[2][0]]

    def test_bakeoff_active_run_survives(self) -> None:
        runs = _runs('completed', 'failed', 'cancelled', 'completed_with_errors')
        pinned = str(runs[3][0])
        plan = plan_run_prune(runs, promoted_job_ids=set(), bakeoff_run_ids={pinned}, keep_last=1)
        assert plan == [runs[1][0], runs[2][0]]

    def test_oldest_terminal_run_beyond_keep_last_is_planned(self) -> None:
        runs = _runs('completed', 'completed', 'completed', 'completed', 'completed')
        plan = plan_run_prune(runs, promoted_job_ids=set(), bakeoff_run_ids=set(), keep_last=3)
        assert plan == [runs[3][0], runs[4][0]]

    def test_keep_last_zero_keeps_everything(self) -> None:
        runs = _runs('completed', 'completed')
        assert (
            plan_run_prune(runs, promoted_job_ids=set(), bakeoff_run_ids=set(), keep_last=0) == []
        )


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
