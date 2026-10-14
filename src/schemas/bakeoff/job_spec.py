"""The job spec the API writes for the evaluator (not a route)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.schemas.bakeoff.models import ModelMode, ModelSource, QuantFormat, TrainTestOverlap


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
