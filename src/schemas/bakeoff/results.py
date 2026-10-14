"""Wire models of the bake-off result files: per-dataset comparison and the matrix."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.bakeoff.models import MappingMethod, ModelSource, TrainTestOverlap
from src.schemas.bakeoff.run import NotCoveredClass, UnmappedModelClass


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
