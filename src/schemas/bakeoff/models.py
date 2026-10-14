"""Wire models of the bake-off model rosters (trained models, profiles, baselines) and the shared enums."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MODEL_NAME_PATTERN = r'^[A-Za-z0-9_.\-]{1,64}$'

ModelSource = Literal['run', 'baseline', 'custom']
ModelMode = Literal['full', 'crop', 'both']
QuantFormat = Literal['fp32_onnx', 'fp16_onnx', 'int8_onnx']
QUANT_FORMATS: tuple[QuantFormat, ...] = ('fp32_onnx', 'fp16_onnx', 'int8_onnx')
MappingMethod = Literal[
    'explicit', 'run_class_remap', 'registry_ids', 'names', 'single_class_fallback'
]


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
    context_class_names: list[str] = Field(
        description='Registry names for context_class_ids, same order/length, resolved by '
        "id lookup; an entry is the id's string form when it's no longer in the registry"
    )
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
