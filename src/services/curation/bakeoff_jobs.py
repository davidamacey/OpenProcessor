"""Model-comparison (bake-off) job building: trained-model listing and job spec v2.

The API resolves everything it can at enqueue time, so the evaluator
container only scores:

* a training run -> checkpoint, imgsz, class names, the export it trained on
  (:func:`~src.services.curation.bakeoff_resolve.resolve_run_model`);
* a model x dataset pair -> its class mapping
  (:mod:`~src.services.curation.bakeoff_mapping`);
* a ``POST /bakeoff/run`` body -> the job file the evaluator reads plus the
  response (:func:`build_job_spec`).

Errors a caller can fix raise
:class:`~src.services.curation.bakeoff_resolve.BakeoffRequestError` with the HTTP
status the router should answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.schemas.bakeoff.job_spec import BakeoffJobSpec, JobDataset, JobModel, JobQuantize
from src.schemas.bakeoff.models import (
    BaselineModel,
    TrainedModel,
    TrainedModelForDataset,
    TrainTestOverlap,
)
from src.schemas.bakeoff.run import (
    AcceptedDataset,
    AcceptedModel,
    BakeoffRunAccepted,
    ClassMapping as ClassMappingWire,
)
from src.services.curation import eval_datasets as evd
from src.services.curation.bakeoff_mapping import (
    _overlap,
    explicit_class_mapping,
    run_class_mapping,
    scored_class_ids,
)
from src.services.curation.bakeoff_resolve import (
    BakeoffRequestError,
    host_checkpoint,
    resolve_baseline,
    resolve_profile,
    resolve_run_model,
    safe_job_id,
)
from src.services.training import jobs as train_jobs


if TYPE_CHECKING:
    from scripts.curation.bakeoff import class_map as cmap
    from src.schemas.bakeoff.run import (
        BakeoffRunRequest,
        BaselineModelRef,
        CustomModelRef,
        RunModelRef,
    )
    from src.services.curation.eval_datasets import EvalDatasetRecord


logger = get_logger(__name__)


# =============================================================================
# Trained-model listing
# =============================================================================


async def list_trained_models(dataset_id: str | None, limit: int) -> list[TrainedModel]:
    dataset = None
    if dataset_id:
        try:
            dataset = evd.resolve_dataset_id(dataset_id)
        except evd.UnknownDatasetError as exc:
            raise BakeoffRequestError(400, str(exc)) from exc
    out: list[TrainedModel] = []
    for status in await train_jobs.list_runs(limit=limit, offset=0):
        if status.state != 'finished' or not status.checkpoint_path:
            continue
        if not host_checkpoint(status.checkpoint_path).is_file():
            continue
        try:
            run = await resolve_run_model(status.job_id)
        except BakeoffRequestError as exc:
            logger.warning('bakeoff_trained_model_skipped', run_id=status.job_id, error=exc.detail)
            continue
        for_dataset = None
        if dataset is not None:
            try:
                n_mapped = len(
                    run_class_mapping(run, dataset, dataset.present_class_ids).model_to_eval or {}
                )
            except BakeoffRequestError:
                n_mapped = 0
            same_frozen = (
                run.frozen_test_sha == dataset.frozen_test_sha
                if run.frozen_test_sha and dataset.frozen_test_sha
                else None
            )
            for_dataset = TrainedModelForDataset(
                dataset_id=dataset.id,
                same_export=run.train_export_id == dataset.id,
                same_frozen_test=same_frozen,
                n_classes_mapped=n_mapped,
                train_test_overlap=_overlap(run, dataset),
            )
        out.append(
            TrainedModel(
                run_id=run.run_id,
                display_name=run.run_id,
                model_family=run.model_family,
                model_size=run.model_size,
                imgsz=run.imgsz,
                checkpoint_path=run.checkpoint_path,
                finished_at=run.finished_at,
                campaign_id=run.campaign_id,
                train_export_id=run.train_export_id,
                dataset_sha=run.dataset_sha,
                frozen_test_sha=run.frozen_test_sha,
                class_names=[run.class_names[k] for k in sorted(run.class_names)],
                single_cls=run.single_cls,
                trainer_map50=run.trainer_map50,
                trainer_map50_split=run.trainer_map50_split,
                for_dataset=for_dataset,
            )
        )
    return out


# =============================================================================
# Run request to job spec v2
# =============================================================================


@dataclass
class _Resolved:
    job: JobModel
    accepted: AcceptedModel


async def _resolve_datasets(request: BakeoffRunRequest) -> list[EvalDatasetRecord]:
    out: list[EvalDatasetRecord] = []
    for ref in request.datasets:
        try:
            if ref.id.startswith('run:'):
                run = await resolve_run_model(ref.id[len('run:') :])
                if run.train_export_id is None:
                    raise BakeoffRequestError(
                        400,
                        f'{ref.id}: the export that run trained on is not under the export '
                        f'root {evd.export_root()}',
                    )
                out.append(evd.resolve_dataset_id(run.train_export_id))
            else:
                out.append(evd.resolve_dataset_id(ref.id))
        except evd.UnknownDatasetError as exc:
            raise BakeoffRequestError(400, str(exc)) from exc
    ids = [d.id for d in out]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise BakeoffRequestError(400, f'duplicate dataset ids: {", ".join(dupes)}')
    return out


def _mappings(
    fn: Any, datasets: list[EvalDatasetRecord], scored: dict[str, list[int]]
) -> dict[str, cmap.ClassMapping]:
    return {d.id: fn(d, scored[d.id]) for d in datasets}


def _job_maps(mappings: dict[str, cmap.ClassMapping]) -> dict[str, dict[str, Any] | None]:
    return {k: (m.to_dict() if m.model_to_eval is not None else None) for k, m in mappings.items()}


def _resolved(
    fields: dict[str, Any],
    mappings: dict[str, cmap.ClassMapping],
    overlaps: dict[str, TrainTestOverlap | None],
) -> _Resolved:
    job = JobModel(
        **fields,
        class_map_by_dataset=_job_maps(mappings),
        train_test_overlap_by_dataset=overlaps,
    )
    accepted = AcceptedModel(
        model=job.model,
        display_name=job.display_name,
        source=job.source,
        class_mapping={k: ClassMappingWire(**m.to_dict()) for k, m in mappings.items()},
        train_test_overlap=overlaps,
    )
    return _Resolved(job, accepted)


async def _resolve_run_ref(
    ref: RunModelRef, datasets: list[EvalDatasetRecord], scored: dict[str, list[int]]
) -> _Resolved:
    run = await resolve_run_model(ref.run_id)
    key, weights = f'run:{run.run_id}', run.checkpoint_path
    if ref.backend == 'onnxruntime':
        key, weights = f'{key}:onnx', str(Path(run.checkpoint_path).with_suffix('.onnx'))
        if not host_checkpoint(weights).is_file():
            raise BakeoffRequestError(
                400, f'training run {run.run_id!r} has no ONNX export at {weights}'
            )
    fields = {
        'model': key,
        'display_name': ref.display_name or run.run_id,
        'source': 'run',
        'run_id': run.run_id,
        'backend': ref.backend,
        'weights': weights,
        'imgsz': run.imgsz,
        'mode': ref.mode,
        'backend_options': {},
        'triton_model': None,
        'training_data': None,
    }
    mappings = _mappings(lambda d, s: run_class_mapping(run, d, s), datasets, scored)
    return _resolved(fields, mappings, {d.id: _overlap(run, d) for d in datasets})


def _resolve_external_ref(
    ref: BaselineModelRef | CustomModelRef,
    model: BaselineModel | CustomModelRef,
    datasets: list[EvalDatasetRecord],
    scored: dict[str, list[int]],
) -> _Resolved:
    fields = {
        'model': f'{ref.source}:{model.name}',
        'display_name': ref.display_name or model.name,
        'source': ref.source,
        'run_id': None,
        'backend': model.backend,
        'weights': model.weights,
        'imgsz': model.imgsz,
        'mode': model.mode,
        'backend_options': dict(model.backend_options),
        'triton_model': model.triton_model,
        'training_data': getattr(model, 'training_data', None),
    }
    mappings = _mappings(
        lambda d, s: explicit_class_mapping(model.class_map, d, s), datasets, scored
    )
    return _resolved(fields, mappings, dict.fromkeys(scored))


async def _resolve_quantize(
    request: BakeoffRunRequest,
    datasets: list[EvalDatasetRecord],
    scored: dict[str, list[int]],
    out_dir: Path,
) -> JobQuantize | None:
    q = request.quantize
    if q is None:
        return None
    run = await resolve_run_model(q.run_id)
    mappings = _mappings(lambda d, s: run_class_mapping(run, d, s), datasets, scored)
    return JobQuantize(
        run_id=run.run_id,
        model_key_prefix=f'run:{run.run_id}',
        checkpoint=run.checkpoint_path,
        imgsz=run.imgsz,
        calib_dataset=str(run.train_export_dir or datasets[0].path),
        calib_split=q.calib_split,
        n_calib=q.n_calib,
        formats=list(q.formats),
        throughput=q.throughput,
        out_root=str(out_dir / 'quant'),
        class_map_by_dataset=_job_maps(mappings),
    )


async def build_job_spec(
    request: BakeoffRunRequest, *, out_root: Path
) -> tuple[BakeoffJobSpec, BakeoffRunAccepted]:
    """Validate + resolve a run request into the evaluator job spec and the response."""
    if not request.models and request.quantize is None:
        raise BakeoffRequestError(400, 'no models specified (give models or a quantize block)')
    if not request.datasets:
        raise BakeoffRequestError(400, 'no datasets specified')
    profile = resolve_profile(request.profile)
    job_id = safe_job_id(request.job_id or datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ'))
    out_dir = out_root / job_id

    datasets = await _resolve_datasets(request)
    scored = {d.id: scored_class_ids(d, profile) for d in datasets}
    resolved: list[_Resolved] = []
    for ref in request.models:
        if ref.source == 'run':
            resolved.append(await _resolve_run_ref(ref, datasets, scored))
        elif ref.source == 'baseline':
            baseline = resolve_baseline(ref.name, profile)
            resolved.append(_resolve_external_ref(ref, baseline, datasets, scored))
        else:
            resolved.append(_resolve_external_ref(ref, ref, datasets, scored))
    keys = [r.job.model for r in resolved]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise BakeoffRequestError(400, f'duplicate model keys: {", ".join(dupes)}')
    quantize = await _resolve_quantize(request, datasets, scored, out_dir)

    warnings = [
        f'{r.job.model}: {o.n_images} of the test images of {ds_id} are in its '
        'training/validation splits'
        for r in resolved
        for ds_id, o in r.job.train_test_overlap_by_dataset.items()
        if o is not None and o.n_images
    ]
    spec = BakeoffJobSpec(
        job_id=job_id,
        profile=request.profile,
        out_dir=str(out_dir),
        datasets=[
            JobDataset(
                id=d.id,
                dir_name=d.dir_name,
                path=str(d.path),
                test_label_sha=d.test_label_sha,
                frozen_test_sha=d.frozen_test_sha,
                eval_class_ids=scored[d.id],
            )
            for d in datasets
        ],
        models=[r.job for r in resolved],
        quantize=quantize,
    )
    accepted = BakeoffRunAccepted(
        status='enqueued',
        job_id=job_id,
        profile=profile.name,
        datasets=[
            AcceptedDataset(
                id=d.id,
                path=str(d.path),
                frozen_test_sha=d.frozen_test_sha,
                test_label_sha=d.test_label_sha,
                n_eval_classes=len(scored[d.id]),
            )
            for d in datasets
        ],
        models=[r.accepted for r in resolved],
        warnings=warnings,
    )
    return spec, accepted
