"""Response models for the /ingest endpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IndexedCounts(BaseModel):
    """Counts of documents indexed per category."""

    global_: bool = Field(..., alias='global', description='Whether global embedding was indexed')
    vehicles: int = Field(default=0, description='Number of vehicle detections indexed')
    people: int = Field(default=0, description='Number of person detections indexed')
    faces: int = Field(default=0, description='Number of faces indexed')

    model_config = ConfigDict(populate_by_name=True)


class NearDuplicateInfo(BaseModel):
    """Information about near-duplicate detection result."""

    action: Literal['joined_group', 'created_group'] = Field(
        ..., description='Action taken for duplicate grouping'
    )
    group_id: str = Field(..., description='Duplicate group ID')
    similarity: float = Field(
        ..., ge=0.0, le=1.0, description='Similarity score with matched image'
    )
    matched_image: str = Field(..., description='Image ID of the matched duplicate')


class OCRInfo(BaseModel):
    """OCR processing result information."""

    num_texts: int = Field(..., ge=0, description='Number of text regions detected')
    full_text: str = Field(..., description='Concatenated extracted text (truncated to 200 chars)')
    indexed: bool = Field(..., description='Whether OCR results were indexed')


class IngestResponse(BaseModel):
    """
    Response for single image ingestion.

    Includes status, indexing results, and optional duplicate/OCR information.
    """

    status: Literal['success', 'duplicate', 'error'] = Field(..., description='Ingestion status')
    image_id: str = Field(..., description='Image identifier')
    num_detections: int = Field(default=0, ge=0, description='Number of YOLO detections')
    num_faces: int = Field(default=0, ge=0, description='Number of faces detected')
    embedding_norm: float = Field(default=0.0, description='L2 norm of global embedding')
    imohash: str | None = Field(
        default=None, description='Image content hash for duplicate detection'
    )
    indexed: IndexedCounts | None = Field(
        default=None, description='Counts of documents indexed per category'
    )
    near_duplicate: NearDuplicateInfo | None = Field(
        default=None, description='Near-duplicate grouping result (if detected)'
    )
    ocr: OCRInfo | None = Field(default=None, description='OCR processing result (if enabled)')
    existing_image_id: str | None = Field(
        default=None, description='ID of existing image (if duplicate)'
    )
    existing_image_path: str | None = Field(
        default=None, description='Path of existing image (if duplicate)'
    )
    message: str | None = Field(default=None, description='Additional status message')
    error: str | None = Field(default=None, description='Error message (if status is error)')
    errors: list[str] | None = Field(default=None, description='List of non-fatal errors')
    total_time_ms: float | None = Field(default=None, description='Processing time in milliseconds')


class BatchIndexedCounts(BaseModel):
    """Aggregate counts for batch ingestion."""

    global_: int = Field(
        default=0, alias='global', description='Number of global embeddings indexed'
    )
    vehicles: int = Field(default=0, description='Number of vehicle detections indexed')
    people: int = Field(default=0, description='Number of person detections indexed')
    faces: int = Field(default=0, description='Number of faces indexed')
    ocr: int = Field(default=0, description='Number of images with OCR indexed')

    model_config = ConfigDict(populate_by_name=True)


class DuplicateDetail(BaseModel):
    """Details about a detected duplicate."""

    image_id: str = Field(..., description='ID of the duplicate image')
    existing_image_id: str | None = Field(default=None, description='ID of existing image in index')
    imohash: str = Field(..., description='Content hash')


class ErrorDetail(BaseModel):
    """Details about an ingestion error."""

    image_id: str = Field(..., description='ID of the failed image')
    error: str = Field(..., description='Error message')


class NearDuplicateDetail(BaseModel):
    """Details about a near-duplicate assignment."""

    image_id: str = Field(..., description='ID of the image')
    action: str = Field(..., description='Action taken (joined_group or created_group)')
    group_id: str = Field(..., description='Duplicate group ID')
    similarity: float = Field(..., description='Similarity score')
    matched_image: str = Field(..., description='ID of matched image')


class BatchIngestResponse(BaseModel):
    """
    Response for batch image ingestion.

    Includes summary statistics and detailed lists of duplicates, errors,
    and near-duplicate assignments.
    """

    status: Literal['success', 'partial', 'error'] = Field(..., description='Overall batch status')
    total: int = Field(..., ge=0, description='Total images submitted')
    processed: int = Field(default=0, ge=0, description='Successfully processed images')
    duplicates: int = Field(default=0, ge=0, description='Exact duplicates skipped')
    errors_count: int = Field(default=0, ge=0, description='Failed images count')
    indexed: BatchIndexedCounts = Field(
        default_factory=BatchIndexedCounts, description='Aggregate indexed counts'
    )
    near_duplicates: int = Field(default=0, ge=0, description='Images assigned to duplicate groups')
    deferred_ops: int = Field(
        default=0, ge=0, description='Images with deferred near-duplicate detection'
    )
    duplicate_details: list[DuplicateDetail] | None = Field(
        default=None, description='Details of detected duplicates'
    )
    error_details: list[ErrorDetail] | None = Field(
        default=None, description='Details of failed images'
    )
    near_duplicate_details: list[NearDuplicateDetail] | None = Field(
        default=None, description='Details of near-duplicate assignments'
    )
    total_time_ms: float | None = Field(default=None, description='Total batch processing time')


class DirectoryIngestResponse(BaseModel):
    """
    Response for directory bulk ingestion.

    Extends batch response with directory-specific metadata.
    """

    status: Literal['success', 'partial', 'error'] = Field(
        ..., description='Overall ingestion status'
    )
    directory: str = Field(..., description='Source directory path')
    total_files: int = Field(..., ge=0, description='Total image files found')
    total: int = Field(..., ge=0, description='Total images submitted for processing')
    processed: int = Field(default=0, ge=0, description='Successfully processed images')
    duplicates: int = Field(default=0, ge=0, description='Exact duplicates skipped')
    errors_count: int = Field(default=0, ge=0, description='Failed images count')
    indexed: BatchIndexedCounts = Field(
        default_factory=BatchIndexedCounts, description='Aggregate indexed counts'
    )
    near_duplicates: int = Field(default=0, ge=0, description='Images assigned to duplicate groups')
    skipped_extensions: list[str] | None = Field(
        default=None, description='File extensions that were skipped'
    )
    total_time_ms: float | None = Field(default=None, description='Total processing time')
    error: str | None = Field(default=None, description='Error message if status is error')
