"""Fixtures of the promote tests: the job fixtures, a valid queue environment and a plan."""

from __future__ import annotations

from src.schemas.train_promote import PromoteRequest
from src.services.training.class_remap import ClassRemapResult
from src.services.training.job_models import TrainJobStatus
from src.services.training.promote_run import PromotePlan


def make_plan(
    triton_name: str = 'cars__det', *, run: str = 'run-1', labels: dict[int, str] | None = None
) -> tuple[PromotePlan, PromoteRequest]:
    plan = PromotePlan(
        job_id=run,
        triton_name=triton_name,
        job_status=TrainJobStatus(
            job_id=run, state='finished', checkpoint_path=f'/jobs/{run}/best.pt'
        ),
        class_id_to_name=labels or {0: 'car', 1: 'truck'},
        class_remap=ClassRemapResult(
            mapping={3: 0, 5: 1},
            names=['car', 'truck'],
            single_cls=False,
            include_classes=[3, 5],
            source='manifest',
        ),
        gate_report={'thresholds': {}, 'failures': []},
        gate_failures=[],
    )
    return plan, PromoteRequest(triton_name='det')


__all__ = ['make_plan']
