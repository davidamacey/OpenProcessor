"""Wire models of the bake-off run request, its response and the run status."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.bakeoff.models import (
    MODEL_NAME_PATTERN,
    QUANT_FORMATS,
    MappingMethod,
    ModelMode,
    ModelSource,
    QuantFormat,
    TrainTestOverlap,
)


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
    model_to_eval_names: dict[str, str] | None = Field(
        description='{"<model class id>": "<eval class name>"}; same keys as model_to_eval, '
        'paired by name (never by raw index) so a consumer never has to re-derive the name '
        'from the eval class id; null under the same condition as model_to_eval'
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
