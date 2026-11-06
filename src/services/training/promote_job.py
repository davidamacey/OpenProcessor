"""The `promote` job: start it, read it, and record what it promoted.

A promote copies the ONNX into the Triton repo, asks Triton to load it, then pays the TensorRT
engine build with throwaway inferences: 2-3 minutes, so it is always a job on ``jobs`` (queue
``jobs``, exclusive per Triton model name, 02b section 2.1). The route validates first (every 4xx
a caller can fix is synchronous), :func:`start_promote` stores the validated plan as the job's
parameters, and the handler (``src/tasks/promote.py``) rebuilds the plan and runs the existing
:func:`~src.services.training.promote_run.execute_promote`.

What a promote records lives in PostgreSQL (``models``, ``promotions``): :class:`DbPromotionRecorder`
inserts the rows once the version directory is known, so ``labels.txt`` and the class remap are
written from the rows, and withdraws them when a later step fails.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from src.db.repos import models as repo
from src.db.session import project_tx
from src.schemas.train_promote import PromoteJobStatus, PromoteRequest, PromoteResponse
from src.services.jobs import status as job_status
from src.services.jobs.service import start_job
from src.services.training.class_remap import ClassRemapResult
from src.services.training.job_models import TrainJobStatus
from src.services.training.promote_gate import PromoteGateFailure
from src.services.training.promote_run import PromotePlan


if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable, Mapping
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.schemas.jobs import JobView
    from src.services.jobs.service import StartResult

KIND = 'promote'
# Order is the order a healthy promote moves through. ``building`` is the first warm-up
# inference (the TensorRT accelerator compiles the engine there, not in Triton's /load);
# ``warming`` is the max-batch pass.
PHASES = ('queued', 'exporting', 'loading', 'building', 'warming', 'done', 'failed')
RUNNING_PHASES = ('exporting', 'loading', 'building', 'warming')
_FAILED = frozenset({'failed', 'cancelled', 'interrupted', 'completed_with_errors'})


def plan_to_params(plan: PromotePlan, payload: PromoteRequest) -> dict[str, Any]:
    """The job parameters: the validated plan and the request, both JSON-able."""
    return {
        'run_job_id': plan.job_id,
        'triton_name': plan.triton_name,
        'request': payload.model_dump(),
        'job_status': plan.job_status.model_dump(mode='json'),
        'class_id_to_name': {str(i): n for i, n in plan.class_id_to_name.items()},
        'class_remap': asdict(plan.class_remap),
        'gate_report': plan.gate_report,
        'gate_failures': [f.model_dump() for f in plan.gate_failures],
    }


def plan_from_params(params: Mapping[str, Any]) -> tuple[PromotePlan, PromoteRequest]:
    remap = params['class_remap']
    plan = PromotePlan(
        job_id=str(params['run_job_id']),
        triton_name=str(params['triton_name']),
        job_status=TrainJobStatus.model_validate(params['job_status']),
        class_id_to_name={int(i): str(n) for i, n in params['class_id_to_name'].items()},
        class_remap=ClassRemapResult(
            mapping={int(k): int(v) for k, v in remap['mapping'].items()},
            names=remap['names'],
            single_cls=bool(remap['single_cls']),
            include_classes=remap['include_classes'],
            source=str(remap['source']),
        ),
        gate_report=dict(params['gate_report']),
        gate_failures=[PromoteGateFailure.model_validate(f) for f in params['gate_failures']],
    )
    return plan, PromoteRequest.model_validate(params['request'])


async def start_promote(
    session: AsyncSession,
    project_id: uuid.UUID,
    plan: PromotePlan,
    payload: PromoteRequest,
    *,
    requested_by: str,
) -> StartResult:
    """Queue the promote; 409 ``job_busy`` while one of the same Triton name is active."""
    return await start_job(
        session,
        KIND,
        project_id,
        params=plan_to_params(plan, payload),
        requested_by=requested_by,
    )


def wire(view: JobView) -> PromoteJobStatus:
    """The promote status wire model of one ``promote`` job."""
    params = view.params or {}
    stage = (view.progress or {}).get('stage')
    if view.status == 'completed':
        status = 'done'
    elif view.status in _FAILED:
        status = 'failed'
    elif view.status == 'queued':
        status = 'queued'
    else:
        status = stage if stage in RUNNING_PHASES else 'exporting'
    error = 'cancelled' if view.status == 'cancelled' else view.error_message
    return PromoteJobStatus(
        promote_id=view.id,
        job_id=str(params.get('run_job_id', '')),
        triton_name=str(params.get('triton_name', '')),
        status=status,
        error=error,
        result=PromoteResponse.model_validate(view.result) if status == 'done' else None,
        started_at=None if view.started_at is None else view.started_at.isoformat(),
        updated_at=view.updated_at.isoformat(),
        finished_at=None if view.finished_at is None else view.finished_at.isoformat(),
        poll_after_s=3 if status in {'queued', *RUNNING_PHASES} else None,
    )


async def read_job(
    session: AsyncSession, project_id: uuid.UUID, promote_id: uuid.UUID
) -> PromoteJobStatus | None:
    """One promote job of the project, or ``None`` for an unknown id or another kind of job."""
    view = await job_status.get_job(session, promote_id, project=project_id)
    return wire(view) if view.kind == KIND else None


async def latest_for_run(
    session: AsyncSession, project_id: uuid.UUID, run_job_id: str
) -> PromoteJobStatus | None:
    """The most recently started promote of ``run_job_id`` (for the train status route)."""
    async with project_tx(session, project_id):
        found = await repo.latest_promote_job_id(session, run_job_id)
    if found is None:
        return None
    return wire(await job_status.get_job(session, found, project=project_id))


@dataclass(frozen=True)
class Recorded:
    """What the promoter writes the model directory from: the stored rows."""

    labels: tuple[str, ...]
    class_remap: Mapping[str, Any] | None


def remap_document(remap: ClassRemapResult, run_job_id: str) -> dict[str, Any] | None:
    """``promotions.class_remap``: the resolved remap by name, ``None`` for a full-class run."""
    if remap.source == 'none':
        return None
    return {
        'original_to_new': {str(k): v for k, v in remap.mapping.items()},
        'single_cls': remap.single_cls,
        'names': remap.names,
        'include_classes': remap.include_classes,
        'source': remap.source,
        'job_id': run_job_id,
    }


class DbPromotionRecorder:
    """Inserts the ``models`` and ``promotions`` rows of one promote and can withdraw them.

    ``sessions`` opens a session in the job's project scope that commits on a clean exit
    (``JobContext.session``). ``record`` runs in its own transaction once the version directory exists; the previous live
    promotion of the Triton name is superseded in the same transaction. ``withdraw`` undoes it
    when a later step fails, so a promote that never served never stays live.
    """

    def __init__(
        self,
        sessions: Callable[[], AbstractAsyncContextManager[AsyncSession]],
        *,
        project_id: uuid.UUID,
        job_id: uuid.UUID,
        plan: PromotePlan,
        requested_by: str,
    ) -> None:
        self._sessions = sessions
        self._project_id = project_id
        self._job_id = job_id
        self._run_job_id = plan.job_id
        self._requested_by = requested_by
        self._labels = [plan.class_id_to_name[i] for i in sorted(plan.class_id_to_name)]
        self._remap = plan.class_remap
        self._promotion_id: uuid.UUID | None = None

    async def record(self, *, triton_name: str, triton_version: int) -> Recorded:
        new = repo.NewModel(
            project_id=self._project_id,
            name=triton_name,
            origin='trained',
            family='yolo26',
            task='detect',
            labels=self._labels,
            artifact_storage_key=f'models/{triton_name}/{triton_version}/model.onnx',
            created_by=self._requested_by,
        )
        async with self._sessions() as session:
            model, promotion = await repo.record_promotion(
                session,
                model=new,
                triton_model_name=triton_name,
                triton_version=triton_version,
                class_remap=remap_document(self._remap, self._run_job_id),
                job_id=self._job_id,
                promoted_by=self._requested_by,
            )
        self._promotion_id = promotion.id
        return Recorded(labels=model.labels, class_remap=promotion.class_remap)

    async def withdraw(self) -> None:
        if self._promotion_id is None:
            return
        async with self._sessions() as session:
            await repo.withdraw_promotion(session, self._promotion_id)
        self._promotion_id = None


__all__ = [
    'KIND',
    'PHASES',
    'DbPromotionRecorder',
    'Recorded',
    'latest_for_run',
    'plan_from_params',
    'plan_to_params',
    'read_job',
    'remap_document',
    'start_promote',
    'wire',
]
