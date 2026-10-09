"""Training-job wire models: the ``job.json`` / ``status.json`` shapes.

Split out of :mod:`src.services.training.jobs`; the file-protocol readers and
writers live there.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import get_gpu_arbiter_config
from src.services.training.augmentation_presets import DEFAULT_AUGMENTATION_PRESET


def _validate_gpu_device_string(v: str) -> str:
    """Validate + canonicalize a ``cuda_visible_devices`` value.

    Accepts any GPU id by default. A deployment that restricts training
    to specific GPUs (see ``GpuArbiterConfig.allowed_gpu_ids``) rejects
    any id outside that set. Returns a canonical, ascending,
    de-duplicated string so ``'2,0'`` and ``'0,2'`` compare equal
    downstream (arbiter lock, trainer manifest).
    """
    tokens = [t.strip() for t in v.split(',') if t.strip()]
    if not tokens:
        msg = 'cuda_visible_devices must not be empty'
        raise ValueError(msg)
    try:
        ids = [int(t) for t in tokens]
    except ValueError as exc:
        msg = f'cuda_visible_devices must be a comma-separated list of GPU ids, got {v!r}'
        raise ValueError(msg) from exc
    if len(ids) != len(set(ids)):
        msg = f'cuda_visible_devices must not repeat a GPU id: {v!r}'
        raise ValueError(msg)
    cfg = get_gpu_arbiter_config()
    bad = sorted(i for i in ids if not cfg.is_gpu_allowed(i))
    if bad:
        msg = (
            f'cuda_visible_devices may only reference an allowed GPU id; got {bad}. '
            f'Configured allowlist: {sorted(cfg.allowed_gpu_ids) or "unrestricted"}.'
        )
        raise ValueError(msg)
    return ','.join(str(i) for i in sorted(ids))


def default_train_gpu_value() -> str:
    """The ``cuda_visible_devices`` value a new spec defaults to.

    Precedence: ``GpuArbiterConfig.default_train_gpus`` (``OP_TRAIN_DEFAULT_GPUS``)
    if set; else the smallest id in ``allowed_gpu_ids`` if an allowlist is
    configured; else ``'0'`` (the generic, unrestricted-install default).
    Always validated through :func:`_validate_gpu_device_string` so a
    misconfigured default (outside the allowlist) fails loudly instead of
    quietly serving a GPU id training will then reject.
    """
    cfg = get_gpu_arbiter_config()
    if cfg.default_train_gpus:
        value = cfg.default_train_gpus
    elif cfg.allowed_gpu_ids:
        value = str(min(cfg.allowed_gpu_ids))
    else:
        value = '0'
    return _validate_gpu_device_string(value)


# =============================================================================
# Pydantic models -- wire format
# =============================================================================


class AugmentationSpec(BaseModel):
    """Optional augmentation block in ``job.json``."""

    model_config = ConfigDict(extra='allow')

    enabled: bool = True
    multiplier: int = Field(default=1, ge=1, le=20)
    # Validated against the catalog by preflight and /start (not here, so
    # preflight can report it as a check); GET /train/augmentation_presets.
    preset: str = Field(default=DEFAULT_AUGMENTATION_PRESET)
    albumentations: dict[str, Any] = Field(default_factory=dict)
    per_class_multiplier: dict[str, int] = Field(default_factory=dict)


def _reject_spec_fields_in_hyperparameters(
    value: dict[str, Any], spec_fields: set[str]
) -> dict[str, Any]:
    """``hyperparameters`` carries trainer arguments only; a top-level spec
    field placed inside it (``include_classes``) would be silently ignored."""
    misplaced = sorted(set(value) & spec_fields)
    if misplaced:
        msg = f'{misplaced} belong at the top level of the spec, not inside hyperparameters'
        raise ValueError(msg)
    return value


class TrainJobSpec(BaseModel):
    """Payload accepted by ``POST /curation/train/start``.

    All fields except the dataset path are optional -- the trainer
    applies profile defaults from :mod:`src.services.training.profiles`
    for anything not supplied.
    """

    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    # Identity ---------------------------------------------------------------
    job_id: str | None = None  # API fills this in if omitted
    campaign_id: str | None = None  # set by start_campaign()
    submitted_by: str = Field(default='labeler-ui')
    submitted_at: str | None = None  # ISO; API fills in

    # Campaign metadata -- copied onto each per-job spec by
    # ``write_campaign``. The trainer reads these between runs to decide
    # whether to auto-skip later siblings or auto-promote the best run.
    stop_when: dict[str, float] | None = None
    auto_promote_best: bool = False
    is_last_in_campaign: bool = False
    # Opt-in: on a finished run, auto-export the checkpoint to portable ONNX
    # (fp32/fp16/int8) and benchmark it via the evaluator so a frontend
    # QuantizationPanel updates with no manual step.
    auto_quantize_bakeoff: bool = False

    # Model selection --------------------------------------------------------
    model_family: Literal['yolo26'] = 'yolo26'
    model_size: Literal['n', 's', 'm', 'l', 'x'] = 'm'
    profile: Literal['probe', 'nano', 'small', 'medium', 'large', 'xlarge', 'custom'] = 'medium'

    # Data -------------------------------------------------------------------
    # F-73: optional, not required. Null/omitted defaults to the current
    # export (data/exports/current -- see
    # src.services.curation.export.resolve_current_export_dir), resolved by
    # _run_preflight before any check reads this field. When no export has
    # ever been run, preflight reports a single, clear blocking check
    # instead of the field forcing a 422 with no explanation of what's
    # actually missing or how to fix it (run POST /export/yolo).
    dataset_export_dir: str | None = Field(
        default=None,
        description=(
            'Absolute path (inside the trainer container) to a frozen export. '
            'Null/omitted defaults to the current export '
            "(the host's data/exports/current symlink target)."
        ),
    )
    include_classes: list[int] | None = Field(
        default=None,
        description='Subset of class_ids; null/empty means all classes in the export.',
    )
    single_cls: bool = False

    # Compute ----------------------------------------------------------------
    cuda_visible_devices: str = Field(default_factory=default_train_gpu_value)

    # Hyperparameters & augmentation ----------------------------------------
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    augmentation: AugmentationSpec | None = None

    # Tracking ---------------------------------------------------------------
    mlflow_run_name: str | None = None

    # Lineage -----------------------------------------------------------------
    # All of these are normally auto-filled by ``write_job`` at submit time
    # and should not be set by API callers directly:
    #   - dataset_sha / frozen_test_sha / test_label_sha / dataset_version_tag:
    #     copied from the export's manifest.json (``src.services.training.
    #     lineage.read_export_identity``), so the trainer's run manifest
    #     records exactly which export, and which frozen test split, this
    #     run trained/evaluated against instead of always recording None.
    #     ``dataset_sha`` is the export's own claim and is never computed;
    #     the two test-split hashes are computed from disk when an older
    #     export's manifest lacks them.
    #   - api_sha / trainer_image_id / trainer_image_revision: this API
    #     process's own build identity plus the trainer container's image
    #     id and baked-in revision label (``lineage.stamp_code_versions``).
    #   - registry_sha / registry_snapshot_path: a sha256 + on-disk copy of
    #     the class registry *at submit time*, so a class rename between
    #     export and promote can't silently relabel the served model
    #     (promote prefers this pinned snapshot over the live registry).
    dataset_sha: str | None = None
    frozen_test_sha: str | None = None
    test_label_sha: str | None = None
    dataset_version_tag: str | None = None
    api_sha: str | None = None
    trainer_image_id: str | None = None
    trainer_image_revision: str | None = None
    registry_sha: str | None = None
    registry_snapshot_path: str | None = None

    # Project isolation (docs/design/openprocessor_internal/projects_plan.md
    # §5.3) -- filled in by write_job() from the bound project, never set by
    # API callers directly. ``project_export_root`` lets the trainer refuse
    # a job whose ``dataset_export_dir`` escapes the project's own export
    # tree (``export_outside_project``); ``mlflow_experiment`` is the
    # project's own MLflow experiment name, read by the trainer instead of
    # any env-var-only experiment name.
    project: str | None = None
    project_export_root: str | None = None
    mlflow_experiment: str | None = None

    @field_validator('hyperparameters')
    @classmethod
    def _hyperparameters_exclude_spec_fields(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _reject_spec_fields_in_hyperparameters(
            v, set(cls.model_fields) - {'hyperparameters'}
        )

    @field_validator('include_classes')
    @classmethod
    def _validate_include_classes(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return None
        if any(c < 0 for c in v):
            msg = 'include_classes must be non-negative integers'
            raise ValueError(msg)
        if len(v) != len(set(v)):
            msg = 'include_classes must be unique'
            raise ValueError(msg)
        return v

    @field_validator('cuda_visible_devices')
    @classmethod
    def _validate_cuda_visible_devices(cls, v: str) -> str:
        return _validate_gpu_device_string(v)


class TrainJobStatus(BaseModel):
    """``status.json`` shape.

    Every field except ``job_id`` and ``state`` is nullable so the trainer
    can write partial updates while the run is still spinning up.
    """

    model_config = ConfigDict(extra='allow')

    job_id: str
    campaign_id: str | None = None
    state: Literal[
        'queued',
        'starting',
        'running',
        'exporting',
        'finished',
        'failed',
        'cancelled',
        'skipped',
        'lost',
    ]
    started_at: str | None = None
    finished_at: str | None = None
    current_epoch: int | None = None
    total_epochs: int | None = None
    epoch_time_s: float | None = None
    # The true last TRAINING epoch's metrics (``{'epoch': N, 'map50': ...,
    # 'map50_95': ...}``) -- distinct from best_checkpoint_metric because
    # Ultralytics re-validates best.pt once more after training and that
    # call does not advance the epoch counter (see docker/trainer/
    # trainer.py's on_fit_epoch_end for how the two are told apart).
    last_epoch_metric: dict[str, Any] | None = None
    # The best checkpoint's (best.pt) own re-validation metrics, as one
    # coherent row -- not a per-key running max across every epoch, which
    # could mix map50 from one epoch with map50_95 from another.
    best_checkpoint_metric: dict[str, Any] | None = None
    mlflow_run_id: str | None = None
    # Served value is rewritten before it reaches the wire -- see
    # ``_public_mlflow_url``/``_prepare_status_for_wire`` below. The trainer
    # writes an internal container hostname here (unreachable from a
    # browser); a caller of ``read_status``/``list_runs`` always gets either
    # a browser-reachable URL (``CurationConfig.mlflow_public_url`` set) or
    # ``null``, never the internal host.
    mlflow_run_url: str | None = None
    mlflow_experiment_id: str | None = None
    checkpoint_path: str | None = None
    gpu: list[dict[str, Any]] = Field(default_factory=list)
    # Served value has ``confusion_matrix_path`` (a server filesystem path)
    # replaced with ``confusion_matrix_url`` -- see ``_rewrite_eval_for_wire``.
    eval: dict[str, Any] | None = None
    # Side-by-side comparison vs the incumbent Triton model. Populated by
    # the trainer at the end of a finished run; null if Triton was
    # unreachable or the comparison failed for any reason.
    compare: dict[str, Any] | None = None
    error: str | None = None
    heartbeat_at: str | None = None
    # Latest background promote of this run (``GET /train/status/{job_id}``
    # only; null in lists and when the run was never promoted async).
    promote: dict[str, Any] | None = None

    @model_validator(mode='before')
    @classmethod
    def _drop_retired_metric_keys(cls, data: Any) -> Any:
        """Drop ``best_metric`` / ``last_metric`` from older status files.

        ``extra='allow'`` would otherwise serve them. They are not mapped
        onto the new fields: ``best_metric`` was a per-key running max that
        could pair map50 and map50_95 from different epochs. An older run
        leaves both new fields null; its ``eval`` block (labelled by
        ``eval.split``) still carries the final score.
        """
        if isinstance(data, dict) and ('best_metric' in data or 'last_metric' in data):
            data = {k: v for k, v in data.items() if k not in ('best_metric', 'last_metric')}
        return data


class CampaignRunSpec(BaseModel):
    """One row inside a campaign payload."""

    model_config = ConfigDict(extra='forbid')

    profile: str
    model_size: Literal['n', 's', 'm', 'l', 'x'] | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator('hyperparameters')
    @classmethod
    def _hyperparameters_exclude_spec_fields(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _reject_spec_fields_in_hyperparameters(
            v, set(TrainJobSpec.model_fields) - {'hyperparameters'}
        )


class TrainCampaignSpec(BaseModel):
    """Payload for ``POST /curation/train/start_campaign``."""

    model_config = ConfigDict(extra='forbid')

    campaign_id: str | None = None  # API fills in
    dataset_export_dir: str
    include_classes: list[int] | None = None
    single_cls: bool = False
    cuda_visible_devices: str = Field(default_factory=default_train_gpu_value)
    augmentation: AugmentationSpec | None = None
    runs: list[CampaignRunSpec]
    stop_when: dict[str, float] | None = None
    auto_promote_best: bool = False
    submitted_by: str = 'labeler-ui'

    @field_validator('runs')
    @classmethod
    def _at_least_one_run(cls, v: list[CampaignRunSpec]) -> list[CampaignRunSpec]:
        if not v:
            msg = 'campaign must include at least one run'
            raise ValueError(msg)
        if len(v) > 16:
            msg = 'campaign may not exceed 16 runs'
            raise ValueError(msg)
        return v

    @field_validator('cuda_visible_devices')
    @classmethod
    def _validate_cuda_visible_devices(cls, v: str) -> str:
        return _validate_gpu_device_string(v)


class Profile(BaseModel):
    """Profile envelope returned by ``GET /curation/train/profiles``."""

    name: str
    description: str = ''
    defaults: dict[str, Any]
