"""Response models for the /embed endpoints."""

from pydantic import BaseModel, Field


class ImageEmbeddingResponse(BaseModel):
    """Response for image embedding endpoint."""

    embedding: list[float] = Field(..., description='512-dim L2-normalized embedding')
    embedding_norm: float = Field(..., description='L2 norm (should be ~1.0)')
    indexed: bool = Field(default=False, description='Whether image was stored')
    image_id: str | None = Field(default=None, description='ID if indexed')
    status: str = Field(default='success')


class TextEmbeddingResponse(BaseModel):
    """Response for text embedding endpoint."""

    embedding: list[float] = Field(..., description='512-dim L2-normalized embedding')
    embedding_norm: float = Field(..., description='L2 norm (should be ~1.0)')
    text: str = Field(..., description='Input text (echoed)')
    status: str = Field(default='success')


class BatchEmbeddingResult(BaseModel):
    """Single result in batch embedding response."""

    index: int = Field(..., description='Index in batch')
    filename: str = Field(..., description='Original filename')
    embedding: list[float] = Field(..., description='512-dim L2-normalized embedding')
    embedding_norm: float = Field(..., description='L2 norm (should be ~1.0)')
    status: str = Field(default='success')
    error: str | None = Field(default=None, description='Error message if failed')


class BatchEmbeddingResponse(BaseModel):
    """Response for batch embedding endpoint."""

    total: int = Field(..., description='Total images submitted')
    successful: int = Field(..., description='Successfully processed images')
    failed: int = Field(..., description='Failed images count')
    results: list[BatchEmbeddingResult] = Field(default_factory=list)
    status: str = Field(default='success')


class BoxEmbedding(BaseModel):
    """Embedding for a single bounding box crop."""

    box: list[float] = Field(..., description='Bounding box [x1, y1, x2, y2] normalized [0,1]')
    embedding: list[float] = Field(..., description='512-dim L2-normalized embedding')


class BoxEmbeddingsResponse(BaseModel):
    """Response for box embeddings endpoint."""

    boxes: list[BoxEmbedding] = Field(default_factory=list)
    num_boxes: int = Field(..., description='Number of boxes processed')
    status: str = Field(default='success')
