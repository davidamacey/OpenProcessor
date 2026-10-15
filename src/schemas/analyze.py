"""Response bodies of the combined analysis routes (single and batch)."""

from pydantic import BaseModel, Field

from src.schemas.detection import ImageMetadata


class DetectionResult(BaseModel):
    """Single detection from YOLO."""

    box: list[float] = Field(..., description='Bounding box [x1, y1, x2, y2] normalized [0,1]')
    confidence: float = Field(..., description='Detection confidence score')
    class_id: int = Field(..., description='COCO class ID (0-79)')
    class_name: str | None = Field(default=None, description='Human-readable class name')
    embedding: list[float] | None = Field(
        default=None, description='512-dim MobileCLIP crop embedding (if include_embedding=True)'
    )


class FaceResult(BaseModel):
    """Single face detection with embedding."""

    box: list[float] = Field(..., description='Face bounding box [x1, y1, x2, y2] normalized [0,1]')
    landmarks: list[float] = Field(
        ..., description='5-point facial landmarks [lx1,ly1,...,lx5,ly5] normalized [0,1]'
    )
    score: float = Field(..., description='Detection confidence score')
    quality: float | None = Field(default=None, description='Face quality score')
    embedding: list[float] | None = Field(
        default=None, description='512-dim ArcFace embedding (if include_embeddings=True)'
    )


class OcrResult(BaseModel):
    """OCR extraction result."""

    texts: list[str] = Field(default_factory=list, description='Detected text strings')
    boxes: list[list[float]] = Field(
        default_factory=list, description='Quad boxes [x1,y1,x2,y2,x3,y3,x4,y4]'
    )
    boxes_normalized: list[list[float]] = Field(
        default_factory=list, description='Axis-aligned boxes [x1,y1,x2,y2] normalized'
    )
    det_scores: list[float] = Field(default_factory=list, description='Detection confidence scores')
    rec_scores: list[float | None] = Field(
        default_factory=list,
        description='Recognition confidence scores; an entry is null when '
        'recognition failed for that line (see rec_errors)',
    )
    rec_errors: list[str | None] = Field(
        default_factory=list,
        description="Same length/order as rec_scores; e.g. 'recognition_failed' where "
        'rec_scores is null, null otherwise',
    )
    full_text: str = Field(default='', description='All text concatenated')
    num_texts: int = Field(default=0, description='Number of text regions')


class AnalyzeResponse(BaseModel):
    """Response for combined analysis endpoint."""

    status: str = Field(default='success', description="'success' or 'error'")
    image: ImageMetadata | None = Field(default=None, description='Original image dimensions')

    # YOLO object detection
    detections: list[DetectionResult] = Field(
        default_factory=list, description='YOLO object detections'
    )
    num_detections: int = Field(default=0, description='Number of objects detected')

    # Face detection/recognition
    faces: list[FaceResult] = Field(default_factory=list, description='Detected faces')
    num_faces: int = Field(default=0, description='Number of faces detected')

    # CLIP embedding
    global_embedding: list[float] | None = Field(
        default=None, description='512-dim MobileCLIP embedding (if include_embedding=True)'
    )
    embedding_norm: float | None = Field(default=None, description='L2 norm of embedding')

    # OCR
    ocr: OcrResult | None = Field(default=None, description='OCR results (if enable_ocr=True)')
    ocr_error: str | None = Field(
        default=None,
        description='Why OCR produced no result when enable_ocr=True and it failed; '
        'null when OCR ran (even with no text) or was not requested',
    )

    # Timing
    total_time_ms: float | None = Field(default=None, description='Total processing time in ms')

    # Per-component timing
    timing: dict[str, float] | None = Field(
        default=None, description='Per-component timing breakdown in ms'
    )


class BatchAnalyzeResult(BaseModel):
    """Single image result in batch analysis."""

    filename: str = Field(..., description='Original filename')
    image_index: int = Field(..., description='Index in batch')
    status: str = Field(default='success', description="'success' or 'error'")
    error: str | None = Field(default=None, description='Error message if failed')

    image: ImageMetadata | None = Field(default=None, description='Image dimensions')
    num_detections: int = Field(default=0, description='Number of objects detected')
    num_faces: int = Field(default=0, description='Number of faces detected')
    has_text: bool = Field(default=False, description='Whether OCR found text')

    # Full results only if requested
    detections: list[DetectionResult] | None = Field(default=None)
    faces: list[FaceResult] | None = Field(default=None)
    global_embedding: list[float] | None = Field(default=None)
    ocr: OcrResult | None = Field(default=None)
    ocr_error: str | None = Field(default=None, description='Why OCR failed for this image')


class BatchAnalyzeResponse(BaseModel):
    """Response for batch combined analysis."""

    status: str = Field(default='success', description='Overall status')
    total_images: int = Field(..., description='Total images in request')
    processed_images: int = Field(..., description='Successfully processed images')
    failed_images: int = Field(..., description='Failed images count')

    results: list[BatchAnalyzeResult] = Field(default_factory=list, description='Per-image results')
    failures: list[dict] | None = Field(default=None, description='Details of failed images')

    # Aggregated stats
    total_detections: int = Field(default=0, description='Sum of detections across all images')
    total_faces: int = Field(default=0, description='Sum of faces across all images')
    images_with_text: int = Field(default=0, description='Number of images with OCR text')

    total_time_ms: float | None = Field(
        default=None, description='Total batch processing time in ms'
    )
