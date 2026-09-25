"""Wire models for the model-comparison (bake-off) routes.

Shapes are specified in ``docs/design/generic_model_comparison_plan.md``
section 7. Result files written by the evaluator (``status.json``,
``<dataset>/comparison.json``, ``matrix.json``) are validated with these
models on read; a file that does not validate is an unsupported (pre-v2)
schema and the route answers 409.

Id spaces used below: an **eval class id** is the eval dataset's dense class
id (its label files / ``data.yaml``); a **model class id** is the model's
own output class id; a **registry class id** is a class-registry id.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MODEL_NAME_PATTERN = r'^[A-Za-z0-9_.\-]{1,64}$'

ModelSource = Literal['run', 'baseline', 'custom']
ModelMode = Literal['full', 'crop', 'both']
QuantFormat = Literal['fp32_onnx', 'fp16_onnx', 'int8_onnx']
QUANT_FORMATS: tuple[QuantFormat, ...] = ('fp32_onnx', 'fp16_onnx', 'int8_onnx')
MappingMethod = Literal[
    'explicit', 'run_class_remap', 'registry_ids', 'names', 'single_class_fallback'
]


# =============================================================================
# Eval datasets: plan 7.2
# =============================================================================


class EvalDatasetClass(BaseModel):
    """One class with at least one object in the dataset's test split."""

    eval_class_id: int
    name: str
    registry_class_id: int | None = Field(
        description='Class-registry id (inverse of class_registry.json export_id_map); '
        'null when the export has no registry mapping for it'
    )
    n_objects: int
    n_images: int


class EvalDataset(BaseModel):
    id: str = Field(description='export:<path under export_root> or external:<group>/<name>')
    source: Literal['export', 'external']
    group: str | None = Field(description='curated | public | sample for external; null for export')
    name: str
    path: str
    is_current: bool
    dataset_kind: Literal['multi_class', 'single_class', 'external']
    nc: int = Field(description='Size of the label space (data.yaml names)')
    classes: list[EvalDatasetClass]
    n_images: int
    n_objects: int
    n_background_images: int
    frozen_test_sha: str | None = Field(
        description='Test-split identity: which frames (filenames only), 16 hex'
    )
    test_label_sha: str | None = Field(
        description='Test-split content: which boxes (label file contents), 16 hex; '
        'the hash the evaluator verifies before scoring'
    )
    sha_source: Literal['manifest', 'computed'] = Field(
        description='Where test_label_sha came from: the export manifest / freeze lock, '
        'or computed from the label files'
    )
    dataset_sha: str | None
    exported_at: str | None
    unlabeled_items_on_exported_images: int | None
    frozen_ok: bool | None = Field(
        description='External only: the test split still matches its freeze lock'
    )


class EvalDatasetList(BaseModel):
    datasets: list[EvalDataset]
    count: int


# =============================================================================
# Trained models: plan 7.3
# =============================================================================


class TrainTestOverlap(BaseModel):
    """Eval test images also present in a run's training export train/val splits."""

    n_images: int
    fraction: float


class TrainedModelForDataset(BaseModel):
    dataset_id: str
    same_export: bool
    same_frozen_test: bool | None = Field(
        description="Run lineage frozen_test_sha == the dataset's; null when either is unknown"
    )
    n_classes_mapped: int
    train_test_overlap: TrainTestOverlap | None


class TrainedModel(BaseModel):
    run_id: str
    display_name: str
    model_family: str | None
    model_size: str | None
    imgsz: int
    checkpoint_path: str
    finished_at: str | None
    campaign_id: str | None
    train_export_id: str | None = Field(
        description='export: id of the export the run trained on; null if it is not '
        'under the export root'
    )
    dataset_sha: str | None
    frozen_test_sha: str | None
    class_names: list[str] = Field(description="The model's classes, by model class id")
    single_cls: bool
    trainer_map50: float | None = Field(
        description="The trainer's own Ultralytics mAP50 (status eval.map50); not a "
        'comparison metric'
    )
    trainer_map50_split: str | None = Field(
        description='Which split trainer_map50 was measured on (status eval.split)'
    )
    for_dataset: TrainedModelForDataset | None = Field(
        default=None, description='Present only when ?dataset_id= is given'
    )


class TrainedModelList(BaseModel):
    models: list[TrainedModel]
    count: int


# =============================================================================
# Profiles and baselines: plan 7.4 to 7.6
# =============================================================================


class BakeoffProfileRow(BaseModel):
    name: str
    description: str
    kind: Literal['registered', 'configured']
    default: bool
    class_filter: list[str]
    imgsz: int
    conf_floor: float
    nms_iou: float
    op_conf: float
    op_iou: float
    rank_metric: str
    default_backend: str
    triton_model: str
    context_class_ids: list[int]
    baselines_path: str


class BakeoffProfileList(BaseModel):
    profiles: list[BakeoffProfileRow]
    count: int
    default_profile: str | None
    default_error: str | None = None


class BaselineModel(BaseModel):
    """A configured external model (also the baseline-registry entry schema)."""

    model_config = ConfigDict(extra='forbid')

    name: str = Field(pattern=MODEL_NAME_PATTERN)
    backend: str = Field(description='A built-in or plugin-registered harness backend')
    weights: str | None = Field(default=None, description='Path inside the evaluator')
    imgsz: int | None = None
    mode: ModelMode = 'full'
    class_map: dict[str, str] | None = Field(
        default=None, description='{"<model class id>": "<eval class name>"}; null = by name'
    )
    backend_options: dict[str, Any] = Field(default_factory=dict)
    training_data: str | None = None
    triton_model: str | None = None


class BaselineModelList(BaseModel):
    baselines: list[BaselineModel]
    count: int


# =============================================================================
# Run request and response: plan 7.7
# =============================================================================


class DatasetRef(BaseModel):
    model_config = ConfigDict(extra='forbid')

    id: str = Field(
        description='An eval dataset id, or run:<job_id> = the export that run trained on'
    )


class RunModelRef(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source: Literal['run']
    run_id: str
    display_name: str | None = None
    backend: Literal['ultralytics', 'onnxruntime'] = Field(
        default='ultralytics',
        description="onnxruntime scores the run's exported ONNX next to best.pt",
    )
    mode: ModelMode = 'full'


class BaselineModelRef(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source: Literal['baseline']
    name: str
    display_name: str | None = None


class CustomModelRef(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source: Literal['custom']
    name: str = Field(pattern=MODEL_NAME_PATTERN)
    backend: str
    weights: str | None = None
    triton_model: str | None = None
    imgsz: int | None = None
    mode: ModelMode = 'full'
    class_map: dict[str, str] | None = None
    backend_options: dict[str, Any] = Field(default_factory=dict)
    display_name: str | None = None


ModelRef = Annotated[RunModelRef | BaselineModelRef | CustomModelRef, Field(discriminator='source')]


class QuantizeRequest(BaseModel):
    """Export a run to ONNX variants and score them in the same job."""

    model_config = ConfigDict(extra='forbid')

    run_id: str
    formats: list[QuantFormat] = Field(default_factory=lambda: list(QUANT_FORMATS))
    n_calib: int = Field(default=1000, ge=1)
    calib_split: Literal['train', 'val'] = 'train'
    throughput: bool = False


class BakeoffRunRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    job_id: str | None = None
    profile: str | None = Field(
        default=None,
        description='BakeoffProfile name (GET /bakeoff/profiles) or a profile .json path; '
        'omitted = the configured default',
    )
    datasets: list[DatasetRef] = Field(default_factory=list)
    models: list[ModelRef] = Field(default_factory=list)
    quantize: QuantizeRequest | None = None


class NotCoveredClass(BaseModel):
    eval_class_id: int
    name: str


class UnmappedModelClass(BaseModel):
    model_class_id: int
    name: str | None = None
    n_predictions: int | None = Field(
        default=None, description='Predictions of this class that could not be scored'
    )


class ClassMapping(BaseModel):
    method: MappingMethod
    model_to_eval: dict[str, int] | None = Field(
        description='{"<model class id>": <eval class id>}; null = matched by name in the evaluator'
    )
    unmapped_model_classes: list[UnmappedModelClass]
    not_covered_eval_classes: list[NotCoveredClass]
    warnings: list[str]


class AcceptedDataset(BaseModel):
    id: str
    path: str
    frozen_test_sha: str | None
    test_label_sha: str | None
    n_eval_classes: int


class AcceptedModel(BaseModel):
    model: str = Field(
        description='Model key: run:<id>, run:<id>:onnx, baseline:<name>, custom:<name>'
    )
    display_name: str
    source: ModelSource
    class_mapping: dict[str, ClassMapping] = Field(description='Per dataset id')
    train_test_overlap: dict[str, TrainTestOverlap | None] = Field(description='Per dataset id')


class BakeoffRunAccepted(BaseModel):
    status: Literal['enqueued']
    job_id: str
    profile: str
    datasets: list[AcceptedDataset]
    models: list[AcceptedModel]
    warnings: list[str]


# =============================================================================
# Status and runs: plan 7.8, 7.9
# =============================================================================


class JobProgress(BaseModel):
    done: int
    total: int


class CompletedTask(BaseModel):
    dataset: str
    model: str


class FailedTask(BaseModel):
    stage: str | None
    dataset: str | None
    model: str | None
    error: str


class BakeoffStatus(BaseModel):
    schema_version: Literal[2]
    job_id: str
    state: Literal['queued', 'running', 'done', 'error']
    profile: str | None
    datasets: list[str]
    models: list[str]
    started_at: str | None
    finished_at: str | None
    progress: JobProgress
    completed: list[CompletedTask]
    failed: list[FailedTask]
    error: str | None


class BakeoffRunRow(BaseModel):
    job_id: str
    state: Literal['queued', 'running', 'done', 'error']
    profile: str | None
    datasets: list[str]
    models: list[str]
    started_at: str | None
    finished_at: str | None


class BakeoffRunList(BaseModel):
    runs: list[BakeoffRunRow]


# =============================================================================
# Comparison: plan 7.10
# =============================================================================


class MetricBlock(BaseModel):
    """Metrics over a model's own covered classes (AP: class mean; P/R/F1: micro)."""

    n_classes: int
    map_50: float | None
    map_50_95: float | None
    map_75: float | None
    ap_small: float | None
    ap_medium: float | None
    ap_large: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    mean_iou: float | None
    tp: int
    fp: int
    fn: int


class CommonMetricBlock(BaseModel):
    """The same metrics restricted to the comparison's common_classes."""

    n_classes: int
    map_50: float | None
    map_50_95: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    tp: int
    fp: int
    fn: int


class PerClassRow(BaseModel):
    eval_class_id: int
    name: str
    n_gt: int
    covered: bool
    model_class_ids: list[int]
    ap50: float | None
    ap50_95: float | None
    ap75: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    tp: int | None
    fp: int | None
    fn: int | None


class Coverage(BaseModel):
    n_eval_classes: int
    n_covered: int
    not_covered: list[NotCoveredClass]
    unmapped_model_classes: list[UnmappedModelClass]
    predictions_outside_scored_classes: int


class RowClassMapping(BaseModel):
    method: MappingMethod
    warnings: list[str]


class LatencyStats(BaseModel):
    mean: float
    p50: float
    p90: float
    p99: float


class StratumMetrics(BaseModel):
    n_images: int
    map_50: float | None
    recall: float | None
    precision: float | None


class ComparisonRow(BaseModel):
    rank: int | None = Field(description='Competition rank (1, 1, 3); null = not rankable')
    model: str
    display_name: str
    source: ModelSource
    run_id: str | None
    runtime: str
    imgsz: int | None
    training_data: str | None
    overall: MetricBlock
    common: CommonMetricBlock
    per_class: list[PerClassRow]
    coverage: Coverage
    class_mapping: RowClassMapping
    train_test_overlap: TrainTestOverlap | None
    latency_ms: LatencyStats
    fps: float
    size_mb: float | None
    per_stratum: dict[str, StratumMetrics]


class ComparisonDataset(BaseModel):
    id: str
    frozen_test_sha: str | None = None
    test_label_sha: str | None = None
    n_images: int | None = None
    n_objects: int | None = None
    n_background_images: int | None = None


class EvalClassGt(BaseModel):
    eval_class_id: int
    name: str
    n_gt: int


class FailedModel(BaseModel):
    model: str
    error: str | None


class BakeoffComparison(BaseModel):
    schema_version: Literal[2]
    job_id: str | None
    profile: str | None
    thresholds: dict[str, float]
    dataset: ComparisonDataset
    eval_classes: list[EvalClassGt]
    common_classes: list[int]
    rank_by: str
    rank_scope: Literal['common', 'overall']
    models: list[ComparisonRow]
    failed: list[FailedModel]
    warnings: list[str]
    n_models: int


# =============================================================================
# Matrix: plan 7.11
# =============================================================================


class MatrixDataset(BaseModel):
    id: str
    frozen_test_sha: str | None
    test_label_sha: str | None
    rank_scope: Literal['common', 'overall'] | None
    n_common_classes: int


class MatrixModel(BaseModel):
    model: str
    display_name: str
    source: ModelSource


class MatrixCell(BaseModel):
    map_50: float | None
    map_50_95: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    latency_ms: float | None
    size_mb: float | None
    coverage: float | None = Field(description='n_covered / n_eval_classes')
    rank: int | None


class BakeoffMatrix(BaseModel):
    schema_version: Literal[2]
    job_id: str | None
    rank_by: str | None
    datasets: list[MatrixDataset]
    models: list[MatrixModel]
    metrics: list[str]
    cells: dict[str, dict[str, MatrixCell]] = Field(description='cells[model key][dataset id]')
    best: dict[str, dict[str, list[str]]] = Field(
        description='best[dataset id][metric] = every tied winning model key'
    )


# =============================================================================
# Job spec v2, API to evaluator (not a route): plan 7.12
# =============================================================================


class JobDataset(BaseModel):
    id: str
    dir_name: str
    path: str
    test_label_sha: str | None
    frozen_test_sha: str | None
    eval_class_ids: list[int]


class JobModel(BaseModel):
    model: str
    display_name: str
    source: ModelSource
    run_id: str | None
    backend: str
    weights: str | None
    imgsz: int | None
    mode: ModelMode
    backend_options: dict[str, Any]
    triton_model: str | None
    training_data: str | None
    class_map_by_dataset: dict[str, dict[str, Any] | None] = Field(
        description='ClassMapping dict per dataset id; null = name-match in the evaluator'
    )
    train_test_overlap_by_dataset: dict[str, TrainTestOverlap | None]


class JobQuantize(BaseModel):
    run_id: str
    model_key_prefix: str
    checkpoint: str
    imgsz: int
    calib_dataset: str
    calib_split: Literal['train', 'val']
    n_calib: int
    formats: list[QuantFormat]
    throughput: bool
    out_root: str
    class_map_by_dataset: dict[str, dict[str, Any] | None]


class BakeoffJobSpec(BaseModel):
    schema_version: Literal[2] = 2
    job_id: str
    profile: str | None
    out_dir: str
    datasets: list[JobDataset]
    models: list[JobModel]
    quantize: JobQuantize | None = None


__all__ = [
    'QUANT_FORMATS',
    'AcceptedDataset',
    'AcceptedModel',
    'BakeoffComparison',
    'BakeoffJobSpec',
    'BakeoffMatrix',
    'BakeoffProfileList',
    'BakeoffProfileRow',
    'BakeoffRunAccepted',
    'BakeoffRunList',
    'BakeoffRunRequest',
    'BakeoffRunRow',
    'BakeoffStatus',
    'BaselineModel',
    'BaselineModelList',
    'BaselineModelRef',
    'ClassMapping',
    'CommonMetricBlock',
    'ComparisonRow',
    'Coverage',
    'CustomModelRef',
    'EvalDataset',
    'EvalDatasetClass',
    'EvalDatasetList',
    'JobDataset',
    'JobModel',
    'JobQuantize',
    'MetricBlock',
    'PerClassRow',
    'QuantizeRequest',
    'RunModelRef',
    'TrainTestOverlap',
    'TrainedModel',
    'TrainedModelForDataset',
    'TrainedModelList',
]
