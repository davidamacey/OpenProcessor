"""Request and response bodies of the model administration routes (status, export, test)."""

from pydantic import BaseModel, Field

from src.schemas.models import ExportFormat, ExportStatus


class ModelStatusResponse(BaseModel):
    """Response for model status query."""

    name: str = Field(description='Model name')
    exists: bool = Field(description='Whether model files exist')
    pytorch_model: bool = Field(description='Whether .pt file exists in pytorch_models/')
    triton_trt: bool = Field(description='Whether TRT model exists in Triton')
    triton_trt_end2end: bool = Field(description='Whether TRT End2End model exists in Triton')
    triton_status: str | None = Field(
        default=None, description='Triton model status (READY, LOADING, etc)'
    )
    export_status: ExportStatus | None = Field(
        default=None, description='Current export status if exporting'
    )
    export_task_id: str | None = Field(default=None, description='Export task ID if exporting')
    num_classes: int | None = Field(default=None, description='Number of classes in model')
    file_size_mb: float | None = Field(default=None, description='PyTorch model file size in MB')


class TestInferenceResponse(BaseModel):
    """Response for test inference."""

    model_name: str
    track: str = Field(description='Inference track used (B or C)')
    backend: str = Field(description='Inference backend (triton)')
    num_detections: int
    detections: list[dict]
    image: dict = Field(description='Image dimensions (height, width)')
    inference_time_ms: float


class ExportRequest(BaseModel):
    """Request body for model export."""

    formats: list[ExportFormat] = Field(
        default=[ExportFormat.TRT_END2END],
        description='Export formats to generate',
    )
    max_batch: int = Field(
        default=32,
        ge=1,
        le=128,
        description='Maximum batch size for dynamic batching',
    )
    normalize_boxes: bool = Field(
        default=True,
        description='Output boxes in [0,1] normalized range',
    )
    auto_load: bool = Field(
        default=True,
        description='Automatically load model into Triton after export',
    )
