"""Run spec: constants, the :class:`JobSpec` dataclass and validation of ``training_runs.spec``.

The API-side half lives in :mod:`src.services.training.jobs`; the constants here mirror it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from logutil import get_logger


logger = get_logger('trainer.job_spec')


# ---------------------------------------------------------------------------
# Constants + environment
# ---------------------------------------------------------------------------

# Mirrors src.services.training.jobs -- keep both in sync.
VALID_FAMILIES = {'yolo26'}
VALID_SIZES = {'n', 's', 'm', 'l', 'x'}
VALID_PROFILES = {'probe', 'nano', 'small', 'medium', 'large', 'xlarge', 'custom'}

# Scratch root for per-job subset/augmentation rewrites. Inside the container
# this is the image's own tmpfs, not a shared host /tmp, and every job writes
# into its own ``<TMP_ROOT>/<job_id>/`` subdirectory which ``run_job`` rmtree's
# on exit. Overridable via OP_TRAIN_TMP_ROOT for a deployment that wants the
# scratch space on a larger volume.
TMP_ROOT = Path(os.environ.get('OP_TRAIN_TMP_ROOT', '/tmp'))  # nosec B108 - container tmpfs


# Ultralytics' ONNX export batch cap. Matches
# src.services.training.yolo_triton_config.DEFAULT_MAX_BATCH so a promote with
# default settings serves an engine the export actually supports.
DEFAULT_EXPORT_BATCH = 8
DEFAULT_INPUT_SIZE = 640


@dataclass
class JobSpec:
    """A run's ``training_runs.spec``, validated.

    Field-for-field a subset of :class:`src.services.training.job_models.TrainJobSpec`
    -- unknown keys are tolerated (kept in :attr:`raw`) so an API that grows a
    field doesn't break an older trainer image.
    """

    job_id: str
    model_family: str
    model_size: str
    profile: str
    dataset_export_dir: Path
    include_classes: list[int] | None
    single_cls: bool
    cuda_visible_devices: str | None
    hyperparameters: dict[str, Any]
    augmentation: dict[str, Any]
    mlflow_run_name: str
    submitted_at: str
    # Project isolation (the project-isolation design
    # §5.3). ``project`` names which project submitted the run;
    # ``project_export_root`` gates ``dataset_export_dir`` -- a job whose
    # export escapes its own project's export tree is refused rather than
    # trained (see ``parse_and_validate_job``'s ``ExportOutsideProjectError``);
    # ``mlflow_experiment`` is used instead of any env-var-only experiment
    # name.
    project: str | None = None
    project_export_root: Path | None = None
    mlflow_experiment: str | None = None
    # Campaign metadata. Only populated for jobs written by
    # ``POST {api_prefix}/train/start_campaign``.
    campaign_id: str | None = None
    stop_when: dict[str, float] | None = None
    auto_promote_best: bool = False
    is_last_in_campaign: bool = False
    # Opt-in: on a finished run, drop a bake-off job that exports this
    # checkpoint to portable ONNX (fp32/fp16/int8) and benchmarks
    # size/speed/accuracy, so a quantization panel updates with no manual step.
    auto_quantize_bakeoff: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def run_log_path(self) -> Path:
        """Scratch copy of the run log; the handler publishes it to the run's artifact prefix."""
        return self.tmp_root / 'run.log'

    @property
    def tmp_root(self) -> Path:
        return TMP_ROOT / self.job_id

    @property
    def subset_dir(self) -> Path:
        return self.tmp_root / 'subset'

    @property
    def is_subset_run(self) -> bool:
        return bool(self.include_classes) or self.single_cls

    @property
    def full_class_remap_path(self) -> Path:
        """Synthesized ``class_remap.json`` for a whole-export (non-subset)
        run -- see ``dataset_prep.write_full_class_remap``. Lives directly
        under ``tmp_root`` (not ``subset_dir``, which only exists for a
        subset rewrite)."""
        return self.tmp_root / 'full_class_remap.json'


def _utcnow_iso() -> str:
    return datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class JobValidationError(ValueError):
    """Base class for spec rejections that carry a stable error code."""

    code: str = 'validation_failed'


class ExportOutsideProjectError(JobValidationError):
    """``dataset_export_dir`` is not a subpath of the job's own
    ``project_export_root`` (the project-isolation design, §5.3)
    -- refused rather than trained, since it would
    otherwise let one project's training run read another project's
    dataset."""

    code = 'export_outside_project'


def parse_and_validate_spec(job_id: str, raw: dict[str, Any]) -> JobSpec:
    """Validate a run's stored spec and reject obviously-broken submissions.

    Most preflight checks (free disk, class counts, ...) live in the API's
    ``POST {api_prefix}/train/preflight``. The trainer only re-checks the few
    constraints that protect the training loop itself.
    """
    if not job_id:
        msg = 'a run needs a job id'
        raise ValueError(msg)

    family = raw.get('model_family') or 'yolo26'
    if family not in VALID_FAMILIES:
        msg = f'unsupported model_family={family!r}; expected one of {sorted(VALID_FAMILIES)}'
        raise ValueError(msg)

    size = raw.get('model_size') or 'm'
    if size not in VALID_SIZES:
        msg = f'unsupported model_size={size!r}; expected one of {sorted(VALID_SIZES)}'
        raise ValueError(msg)

    profile = raw.get('profile') or 'medium'
    if profile not in VALID_PROFILES:
        msg = f'unsupported profile={profile!r}; expected one of {sorted(VALID_PROFILES)}'
        raise ValueError(msg)

    export_dir_raw = raw.get('dataset_export_dir')
    if not export_dir_raw:
        msg = "run spec missing 'dataset_export_dir'"
        raise ValueError(msg)
    export_dir = Path(export_dir_raw)
    if not export_dir.is_dir():
        msg = f'dataset_export_dir does not exist: {export_dir}'
        raise ValueError(msg)

    project = raw.get('project')
    project_export_root_raw = raw.get('project_export_root')
    project_export_root = Path(project_export_root_raw) if project_export_root_raw else None
    if project_export_root is not None:
        try:
            export_dir.resolve().relative_to(project_export_root.resolve())
        except ValueError as exc:
            msg = (
                f"dataset_export_dir {export_dir} is not under this job's "
                f'project_export_root {project_export_root}'
            )
            raise ExportOutsideProjectError(msg) from exc

    include_classes = raw.get('include_classes')
    if include_classes is not None and (
        not isinstance(include_classes, list)
        or not all(isinstance(c, int) for c in include_classes)
    ):
        msg = 'include_classes must be a list[int]'
        raise ValueError(msg)

    hyperparameters = dict(raw.get('hyperparameters') or {})
    # YOLO26 + optimizer=auto silently trades MuSGD for AdamW on short runs
    # (Ultralytics issue #23696). The API rejects it too; re-checked here in
    # case a spec was stored by another client.
    if family == 'yolo26' and str(hyperparameters.get('optimizer', '')).lower() == 'auto':
        msg = (
            "optimizer='auto' is rejected for YOLO26 (Ultralytics issue #23696); "
            'set optimizer=MuSGD explicitly'
        )
        raise ValueError(msg)

    cuda_visible = raw.get('cuda_visible_devices')
    stop_when_raw = raw.get('stop_when')
    stop_when = (
        {str(k): float(v) for k, v in stop_when_raw.items()}
        if isinstance(stop_when_raw, dict)
        else None
    )
    campaign_id = raw.get('campaign_id')

    return JobSpec(
        job_id=job_id,
        model_family=family,
        model_size=size,
        profile=profile,
        dataset_export_dir=export_dir,
        include_classes=include_classes,
        single_cls=bool(raw.get('single_cls', False)),
        cuda_visible_devices=str(cuda_visible) if cuda_visible else None,
        hyperparameters=hyperparameters,
        augmentation=dict(raw.get('augmentation') or {}),
        mlflow_run_name=raw.get('mlflow_run_name') or job_id,
        submitted_at=raw.get('submitted_at') or _utcnow_iso(),
        campaign_id=str(campaign_id) if campaign_id else None,
        stop_when=stop_when,
        auto_promote_best=bool(raw.get('auto_promote_best', False)),
        is_last_in_campaign=bool(raw.get('is_last_in_campaign', False)),
        auto_quantize_bakeoff=bool(raw.get('auto_quantize_bakeoff', False)),
        project=str(project) if project else None,
        project_export_root=project_export_root,
        mlflow_experiment=raw.get('mlflow_experiment') or None,
        raw=raw,
    )
