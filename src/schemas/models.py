"""
Pydantic schemas for Model Management API.

Defines request/response models for:
- Model upload and export
- Export task tracking
- Model listing and management
"""

from datetime import datetime
from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExportFormat(str, Enum):  # noqa: UP042  # (str, Enum) keeps str() and format() as Class.MEMBER; StrEnum would change both
    """Available export formats."""

    TRT = 'trt'
    ONNX = 'onnx'
    ALL = 'all'


class ExportStatus(str, Enum):  # noqa: UP042  # (str, Enum) keeps str() and format() as Class.MEMBER; StrEnum would change both
    """Export task status."""

    PENDING = 'pending'
    VALIDATING = 'validating'
    EXPORTING = 'exporting'
    LOADING = 'loading'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class ExportTaskResponse(BaseModel):
    """Response for export task creation."""

    task_id: str = Field(description='Unique task identifier for status polling')
    model_name: str = Field(description='Model name being exported')
    triton_name: str = Field(description='Triton model name')
    status: ExportStatus = Field(description='Current task status')
    message: str = Field(description='Status message')
    created_at: datetime = Field(description='Task creation timestamp')


class ExportTaskStatus(BaseModel):
    """Detailed export task status."""

    task_id: str
    model_name: str
    triton_name: str
    status: ExportStatus
    progress: float = Field(ge=0.0, le=100.0, description='Export progress percentage')
    current_step: str = Field(description='Current export step')
    message: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    formats_completed: list[str] = Field(default_factory=list)
    formats_pending: list[str] = Field(default_factory=list)
    num_classes: int | None = None
    class_names: list[str] | None = None
    triton_loaded: bool = False
    # Timing information
    export_duration_seconds: float | None = Field(
        default=None, description='Total export duration in seconds'
    )
    step_times: dict[str, float] | None = Field(
        default=None, description='Timing for each export step in seconds'
    )


class ModelInfo(BaseModel):
    """Information about a deployed model."""

    name: str = Field(description='Model name in Triton')
    status: str = Field(description='Model status (READY, LOADING, UNAVAILABLE)')
    versions: list[str] = Field(default_factory=list, description='Available model versions')
    backend: str | None = Field(default=None, description='Model backend (tensorrt, onnxruntime)')
    max_batch_size: int | None = None
    has_labels: bool = Field(default=False, description='Whether labels.txt exists')
    num_classes: int | None = None
    input_shape: list[int] | None = None


class ModelUsageInstructions(BaseModel):
    """Usage instructions for models."""

    detection: str = Field(
        description='How to use model for object detection',
        default='curl -X POST http://localhost:4603/detect -F "image=@your_image.jpg" -F "model_name={model_name}"',
    )
    analysis: str = Field(
        description='How to use model for full analysis',
        default='curl -X POST http://localhost:4603/analyze -F "image=@your_image.jpg" -F "model_name={model_name}"',
    )
    batch: str = Field(
        description='How to use model for batch detection',
        default='curl -X POST http://localhost:4603/detect/batch -F "images=@image1.jpg" -F "images=@image2.jpg" -F "model_name={model_name}"',
    )


class ModelListResponse(BaseModel):
    """Response for listing models."""

    models: list[ModelInfo]
    total: int
    triton_status: str = Field(description='Triton server status')
    usage_instructions: ModelUsageInstructions = Field(
        default_factory=ModelUsageInstructions,
        description='Instructions for using models in API requests',
    )
    notes: list[str] = Field(
        default_factory=lambda: [
            'Model names ending with _trt_end2end or _trt serve the same fused output0 [300, 6]',
            'Use model_name parameter to specify which model to use',
            'Default model is yolov11_small_trt_end2end if not specified',
            'Uploaded models appear here after successful export',
        ],
        description='Important notes about model usage',
    )


class ModelLoadResponse(BaseModel):
    """Response for model load/unload operations."""

    model_name: str
    action: str = Field(description='load or unload')
    success: bool
    message: str


class ModelDeleteResponse(BaseModel):
    """Response for model deletion."""

    model_name: str
    deleted_files: list[str]
    unloaded_from_triton: bool
    message: str


# --- Engine and encoder contracts (research seams S-1, S-2, X-5) -------------------------------
# Stored as jsonb on model_engines and embedding_model_versions and validated on write and read,
# so a decoder or kernel change is a contract diff instead of an edit hidden in two files.


class _Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class SizeRange(_Contract):
    """A dynamic dimension as TensorRT optimization-profile ``min <= opt <= max``."""

    min: int = Field(ge=1)
    opt: int = Field(ge=1)
    max: int = Field(ge=1)

    @model_validator(mode='after')
    def _ordered(self) -> Self:
        if not self.min <= self.opt <= self.max:
            msg = (
                f'size range must satisfy min <= opt <= max, got {self.min}, {self.opt}, {self.max}'
            )
            raise ValueError(msg)
        return self


class NormalizeSpec(_Contract):
    """``(pixel * scale - mean) / std`` per channel, applied outside the graph."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    scale: float = Field(gt=0)


class EngineInputContract(_Contract):
    """Every pixel input of an engine: what a caller must produce for it."""

    dtype: Literal['uint8', 'fp16', 'fp32']
    layout: Literal['NCHW']
    size: tuple[int | SizeRange, int | SizeRange] | SizeRange
    color: Literal['RGB', 'BGR']
    normalize: Literal['in_graph'] | NormalizeSpec
    resize: Literal['letterbox', 'short_edge_center_crop', 'stretch', 'native_width']
    interpolation: Literal['pil_bilinear', 'bilinear', 'area']
    fill: int = Field(default=0, ge=0, le=255)
    pad_split: Literal['floor_half', 'ultralytics_round'] | None = None
    max_side: int | None = Field(default=None, ge=1)

    @model_validator(mode='after')
    def _fixed_dims_positive(self) -> Self:
        if isinstance(self.size, tuple) and any(isinstance(d, int) and d < 1 for d in self.size):
            msg = 'a fixed size dimension must be at least 1'
            raise ValueError(msg)
        return self


class EngineOutputContract(_Contract):
    """What an engine returns, so consumers check the contract instead of the model name."""

    kind: Literal['end2end_boxes', 'embedding', 'anchors_raw', 'ctc_logits', 'ctc_argmax']
    shapes: list[list[int]] = Field(min_length=1)  # -1 marks a dynamic dimension
    dtypes: list[Literal['uint8', 'int32', 'int64', 'fp16', 'fp32']] = Field(min_length=1)

    @model_validator(mode='after')
    def _one_dtype_per_output(self) -> Self:
        if len(self.shapes) != len(self.dtypes):
            msg = 'shapes and dtypes must list the same outputs'
            raise ValueError(msg)
        return self


class TextPrefixes(_Contract):
    query: str = ''
    document: str = ''


class TextInputSpec(_Contract):
    context_length: int = Field(ge=1)
    prefixes: TextPrefixes = TextPrefixes()


class EncoderInputSpec(_Contract):
    """The logical encoder input; ``image`` is the contract of its ready Triton image engine."""

    image: EngineInputContract | None = None
    text: TextInputSpec | None = None

    @model_validator(mode='after')
    def _has_a_modality(self) -> Self:
        if self.image is None and self.text is None:
            msg = 'an encoder input spec needs an image or a text input'
            raise ValueError(msg)
        return self
