"""Model-comparison (bake-off) job resolution: runs, baselines, class maps, job spec v2.

The API resolves everything it can at enqueue time, so the evaluator
container only scores (plan ``generic_model_comparison_plan.md`` sections
3.3, 5, 7.7, 7.12):

* a training run -> checkpoint, imgsz, class names, the export it trained on
  (:func:`resolve_run_model`);
* a model x dataset pair -> its class mapping (:func:`build_class_mapping`,
  the plan's five-rule order, algorithm in ``scripts/curation/bakeoff/class_map.py``);
* a ``POST /bakeoff/run`` body -> the job file the evaluator reads plus the
  response (:func:`build_job_spec`).

Errors a caller can fix raise :class:`BakeoffRequestError` with the HTTP
status the router should answer.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.curation.bakeoff import class_map as cmap
from src.config import get_curation_config
from src.core.logging import get_logger
from src.routers.curation._bakeoff_models import (
    AcceptedDataset,
    AcceptedModel,
    BakeoffJobSpec,
    BakeoffRunAccepted,
    BaselineModel,
    ClassMapping as ClassMappingWire,
    JobDataset,
    JobModel,
    JobQuantize,
    TrainedModel,
    TrainedModelForDataset,
    TrainTestOverlap,
)
from src.services.curation import eval_datasets as evd
from src.services.training import jobs as train_jobs
from src.services.training.triton_promote import ClassRemapUnreadableError, resolve_class_remap


if TYPE_CHECKING:
    from scripts.curation.bakeoff.profile import BakeoffProfile
    from src.routers.curation._bakeoff_models import (
        BakeoffRunRequest,
        BaselineModelRef,
        CustomModelRef,
        RunModelRef,
    )
    from src.services.curation.eval_datasets import EvalDatasetRecord
    from src.services.training.triton_promote import ClassRemapResult


logger = get_logger(__name__)
_config = get_curation_config()

# Runs record checkpoint_path as the trainer saw it. Older runs used a /runs/...
# container path; the API sees the same files under this root.
RUNS_HOST_ROOT = Path(
    os.environ.get('OP_TRAIN_RUNS_ROOT', str(_config.state_dir / 'training_runs'))
)
# Baseline registry (external models). Empty by default; a profile may name
# its own via baselines_path, OP_BAKEOFF_BASELINES_PATH replaces the default.
_DEFAULT_BASELINES_PATH = (
    Path(__file__).resolve().parents[3] / 'scripts/curation/bakeoff/baselines.json'
)
BASELINES_PATH = Path(os.environ.get('OP_BAKEOFF_BASELINES_PATH') or _DEFAULT_BASELINES_PATH)

JOB_ID_RE = re.compile(r'[A-Za-z0-9_.:-]{1,64}')
NAME_MATCH_WARNING = 'resolved by name inside the evaluator'


class BakeoffRequestError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def safe_job_id(job_id: str) -> str:
    if not JOB_ID_RE.fullmatch(job_id):
        raise BakeoffRequestError(400, f'invalid job_id: {job_id!r}')
    return job_id


def host_checkpoint(checkpoint_path: str) -> Path:
    """A run's checkpoint path as this process sees it (``/runs/...`` -> host root)."""
    if checkpoint_path.startswith('/runs/'):
        return RUNS_HOST_ROOT / checkpoint_path[len('/runs/') :]
    return Path(checkpoint_path)


def infer_model_size(job_id: str, model_size: str | None) -> str | None:
    """Size from the manifest/status, else parsed from a ``..._yolo26n`` job-id suffix."""
    if model_size:
        return model_size
    m = re.search(r'yolo\d+([nsmlx])$', job_id)
    return m.group(1) if m else None


# =============================================================================
# Profiles + baselines
# =============================================================================


def resolve_profile(spec: str | None) -> BakeoffProfile:
    from scripts.curation.bakeoff.profile import resolve_profile as _resolve

    try:
        return _resolve(spec)
    except (OSError, TypeError, ValueError) as exc:
        raise BakeoffRequestError(400, str(exc)) from exc


def baselines_path_for(profile_name: str | None) -> Path:
    """The baseline registry of a registered profile (``None`` = the default registry)."""
    if not profile_name:
        return BASELINES_PATH
    if '/' in profile_name or profile_name.endswith('.json'):
        raise BakeoffRequestError(400, 'profile must be a profile name')
    from scripts.curation.bakeoff.profile import resolve_baselines_path

    return resolve_baselines_path(resolve_profile(profile_name), BASELINES_PATH)


def load_baselines(path: Path) -> list[BaselineModel]:
    """Registry entries that match the 7.6 schema; others are logged and skipped."""
    try:
        reg = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        logger.warning('bakeoff_baselines_read_failed', path=str(path), error=str(exc))
        return []
    out: list[BaselineModel] = []
    for entry in reg.get('baselines', []) if isinstance(reg, dict) else []:
        try:
            out.append(BaselineModel.model_validate(entry))
        except ValueError as exc:
            logger.warning('bakeoff_baseline_invalid', path=str(path), error=str(exc))
    return out


def resolve_baseline(name: str, profile: BakeoffProfile) -> BaselineModel:
    from scripts.curation.bakeoff.profile import resolve_baselines_path

    path = resolve_baselines_path(profile, BASELINES_PATH)
    for b in load_baselines(path):
        if b.name == name:
            return b
    raise BakeoffRequestError(400, f'unknown baseline {name!r} (registry {path})')


# =============================================================================
# Training runs
# =============================================================================


@dataclass(frozen=True)
class RunModel:
    """A finished training run, resolved for scoring."""

    run_id: str
    checkpoint_path: str
    imgsz: int
    model_family: str | None
    model_size: str | None
    finished_at: str | None
    campaign_id: str | None
    single_cls: bool
    include_classes: list[int] | None
    remap: ClassRemapResult
    class_names: dict[int, str]
    train_export_dir: Path | None
    train_export_id: str | None
    train_export_id_map: dict[int, int]
    dataset_sha: str | None
    frozen_test_sha: str | None
    trainer_map50: float | None
    trainer_map50_split: str | None


def _model_names(remap: ClassRemapResult, train_dir: Path | None) -> dict[int, str]:
    if remap.names:
        return dict(enumerate(remap.names))
    train_names = cmap.read_names(train_dir / 'data.yaml') if train_dir else {}
    if remap.mapping:
        id_map = cmap.read_export_id_map(train_dir) if train_dir else {}
        return {
            new: train_names.get(id_map.get(orig, -1), f'class_{orig}')
            for orig, new in remap.mapping.items()
        }
    return train_names


async def resolve_run_model(run_id: str) -> RunModel:
    """Status + manifest + class remap of a finished run with its checkpoint on disk."""
    try:
        status = await train_jobs.read_status(run_id)
    except ValueError:
        status = None
    if status is None:
        raise BakeoffRequestError(400, f'unknown training run {run_id!r}')
    if status.state != 'finished' or not status.checkpoint_path:
        raise BakeoffRequestError(400, f'training run {run_id!r} is {status.state}, not finished')
    ckpt = host_checkpoint(status.checkpoint_path)
    if not ckpt.is_file():
        raise BakeoffRequestError(400, f'training run {run_id!r} checkpoint is missing: {ckpt}')
    manifest = await train_jobs.read_manifest(run_id) or {}
    lineage = manifest.get('lineage') or {}
    spec = manifest.get('spec') or {}
    if not lineage.get('export_dir'):
        job = await train_jobs.read_job_spec(run_id) or {}
        lineage = {
            'export_dir': job.get('dataset_export_dir'),
            'single_cls': job.get('single_cls'),
            'include_classes': job.get('include_classes'),
            **lineage,
        }
    try:
        remap = resolve_class_remap(job_id=run_id, checkpoint_path=ckpt, manifest=manifest)
    except ClassRemapUnreadableError as exc:
        raise BakeoffRequestError(422, f'training run {run_id!r}: {exc}') from exc

    train_dir: Path | None = None
    train_id: str | None = None
    if lineage.get('export_dir'):
        try:
            train_id = evd.export_id_for_dir(lineage['export_dir'])
        except evd.UnknownDatasetError:
            train_id = None
        else:
            train_dir = evd.EXPORT_ROOT / train_id.split(':', 1)[1]
    run_eval = status.eval or {}
    hyper = spec.get('hyperparameters') or {}
    return RunModel(
        run_id=run_id,
        checkpoint_path=status.checkpoint_path,
        imgsz=int(hyper.get('imgsz') or 640),
        model_family=spec.get('model_family'),
        model_size=infer_model_size(run_id, spec.get('model_size')),
        finished_at=status.finished_at,
        campaign_id=status.campaign_id,
        single_cls=bool(lineage.get('single_cls') or remap.single_cls),
        include_classes=lineage.get('include_classes') or remap.include_classes,
        remap=remap,
        class_names=_model_names(remap, train_dir),
        train_export_dir=train_dir,
        train_export_id=train_id,
        train_export_id_map=cmap.read_export_id_map(train_dir) if train_dir else {},
        dataset_sha=lineage.get('dataset_sha'),
        frozen_test_sha=lineage.get('frozen_test_sha'),
        trainer_map50=run_eval.get('map50'),
        trainer_map50_split=run_eval.get('split'),
    )


# =============================================================================
# Class mapping: plan 3.3
# =============================================================================


def scored_class_ids(dataset: EvalDatasetRecord, profile: BakeoffProfile) -> list[int]:
    """Classes present in the test split, narrowed by the profile's ``class_filter``."""
    present = dataset.present_class_ids
    if not profile.class_filter:
        return present
    wanted = {cmap.normalize_name(n) for n in profile.class_filter}
    return [c for c in present if cmap.normalize_name(dataset.class_names.get(c, '')) in wanted]


def _single_cls_mapping(
    run: RunModel, dataset: EvalDatasetRecord, scored: list[int]
) -> cmap.ClassMapping | None:
    """``None`` when the run covers exactly one class (then rules 2/3 apply)."""
    n = len(run.include_classes) if run.include_classes else len(run.class_names)
    if n == 1:
        return None
    if len(scored) == 1:
        return cmap.finalize(
            'single_class_fallback',
            {0: scored[0]},
            model_names=None,
            eval_names=dataset.class_names,
            scored_class_ids=scored,
        )
    raise BakeoffRequestError(
        422,
        f'single_cls run over {n} classes cannot be scored per class; '
        'compare it on a single-class export',
    )


def run_class_mapping(
    run: RunModel, dataset: EvalDatasetRecord, scored: list[int]
) -> cmap.ClassMapping:
    """Rules 2-4 of plan 3.3 for a training run."""
    if run.single_cls:
        fallback = _single_cls_mapping(run, dataset, scored)
        if fallback is not None:
            return fallback
    names = run.class_names
    # A single_cls model is named 'object'; do not cross-check that name.
    check_names = None if run.single_cls else names
    eval_map, eval_names = dataset.export_id_map, dataset.class_names
    warnings: list[str] = []
    if run.remap.mapping and eval_map:
        new_to_orig = {new: orig for orig, new in run.remap.mapping.items()}
        m2e, warnings = cmap.resolve_by_registry(
            new_to_orig, eval_map, model_names=check_names, eval_names=eval_names
        )
        method = 'run_class_remap'
    elif not run.remap.mapping and run.train_export_id_map and eval_map:
        m2e, warnings = cmap.resolve_by_registry(
            cmap.invert_export_id_map(run.train_export_id_map),
            eval_map,
            model_names=check_names,
            eval_names=eval_names,
        )
        method = 'registry_ids'
    else:
        m2e, method = cmap.resolve_by_names(names, eval_names), 'names'
    return cmap.finalize(
        method,
        m2e,
        model_names=names,
        eval_names=eval_names,
        scored_class_ids=scored,
        warnings=warnings,
    )


def explicit_class_mapping(
    class_map: dict[str, str] | None, dataset: EvalDatasetRecord, scored: list[int]
) -> cmap.ClassMapping:
    """Rules 1 and 5: an explicit name map, else a name-match placeholder for the evaluator."""
    if class_map is None:
        return cmap.ClassMapping('names', None, warnings=[NAME_MATCH_WARNING])
    m2e, warnings = cmap.resolve_explicit(class_map, dataset.class_names)
    return cmap.finalize(
        'explicit',
        m2e,
        model_names={int(k): v for k, v in class_map.items()},
        eval_names=dataset.class_names,
        scored_class_ids=scored,
        warnings=warnings,
    )


def _overlap(run: RunModel, dataset: EvalDatasetRecord) -> TrainTestOverlap | None:
    if run.train_export_dir is None:
        return None
    return TrainTestOverlap(**evd.train_test_overlap(run.train_export_dir, dataset))


# =============================================================================
# Trained-model listing: plan 7.3
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
# Run request to job spec v2: plan 7.7, 7.12
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
                        f'root {evd.EXPORT_ROOT}',
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


__all__ = [
    'BASELINES_PATH',
    'RUNS_HOST_ROOT',
    'BakeoffRequestError',
    'RunModel',
    'baselines_path_for',
    'build_job_spec',
    'explicit_class_mapping',
    'host_checkpoint',
    'list_trained_models',
    'load_baselines',
    'resolve_baseline',
    'resolve_profile',
    'resolve_run_model',
    'run_class_mapping',
    'safe_job_id',
    'scored_class_ids',
]
