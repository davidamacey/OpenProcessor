"""Response models for the /ingest endpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config.faces import faces_enabled
from src.schemas.jobs import JobView


class IndexedCounts(BaseModel):
    """Counts of documents indexed per category."""

    global_: bool = Field(..., alias='global', description='Whether global embedding was indexed')
    vehicles: int = Field(default=0, description='Number of vehicle detections indexed')
    people: int = Field(default=0, description='Number of person detections indexed')
    faces: int = Field(default=0, description='Number of faces indexed')

    model_config = ConfigDict(populate_by_name=True)


class NearDuplicateCandidate(BaseModel):
    """A stored image above the core near-duplicate cut."""

    image_id: str = Field(..., description='ID of the stored image')
    cosine: float = Field(..., ge=-1.0, le=1.0, description='Cosine similarity to the new image')


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
    near_duplicate_candidates: list[NearDuplicateCandidate] | None = Field(
        default=None,
        description='Stored images above the core cut, closest first. The group itself appears '
        'once the near_dup_core_group job has run (GET /query/duplicates).',
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
    faces_enabled: bool = Field(
        default_factory=faces_enabled,
        description='False when the deployment has face models off (OP_FACES_ENABLED=false): no face was detected or indexed',
    )


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


class ImageNearDuplicateCandidates(BaseModel):
    """The near-duplicate candidates of one image of a batch."""

    image_id: str = Field(..., description='ID of the new image')
    candidates: list[NearDuplicateCandidate] = Field(
        ..., description='Stored images above the core cut, closest first'
    )


class BatchIngestResponse(BaseModel):
    """
    Response for batch image ingestion.

    Includes summary statistics and detailed lists of duplicates, errors,
    and near-duplicate candidates.
    """

    status: Literal['success', 'partial', 'error'] = Field(..., description='Overall batch status')
    total: int = Field(..., ge=0, description='Total images submitted')
    processed: int = Field(default=0, ge=0, description='Successfully processed images')
    duplicates: int = Field(default=0, ge=0, description='Exact duplicates skipped')
    errors_count: int = Field(default=0, ge=0, description='Failed images count')
    indexed: BatchIndexedCounts = Field(
        default_factory=BatchIndexedCounts, description='Aggregate indexed counts'
    )
    near_duplicates: int = Field(
        default=0, ge=0, description='Images with near-duplicate candidates'
    )
    deferred_ops: int = Field(
        default=0, ge=0, description='Images with skipped near-duplicate candidate lookup'
    )
    duplicate_details: list[DuplicateDetail] | None = Field(
        default=None, description='Details of detected duplicates'
    )
    error_details: list[ErrorDetail] | None = Field(
        default=None, description='Details of failed images'
    )
    near_duplicate_candidates: list[ImageNearDuplicateCandidates] | None = Field(
        default=None,
        description='Stored images above the core cut for each new image. The groups appear once '
        'the near_dup_core_group job has run (GET /query/duplicates).',
    )
    total_time_ms: float | None = Field(default=None, description='Total batch processing time')
    faces_enabled: bool = Field(
        default_factory=faces_enabled,
        description='False when the deployment has face models off (OP_FACES_ENABLED=false): no face was detected or indexed',
    )


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
    near_duplicates: int = Field(
        default=0, ge=0, description='Images with near-duplicate candidates'
    )
    skipped_extensions: list[str] | None = Field(
        default=None, description='File extensions that were skipped'
    )
    total_time_ms: float | None = Field(default=None, description='Total processing time')
    faces_enabled: bool = Field(
        default_factory=faces_enabled,
        description='False when the deployment has face models off (OP_FACES_ENABLED=false): no face was detected or indexed',
    )
    error: str | None = Field(default=None, description='Error message if status is error')


class DirectoryIngestAccepted(BaseModel):
    """``202`` of ``POST /ingest/directory``: the job that ingests the directory."""

    job_id: str = Field(..., description='The directory_ingest job')
    status: Literal['queued'] = Field(..., description='Always queued at the start')
    status_url: str = Field(..., description='Where to read the job (GET)')


class DirectoryIngestStatus(BaseModel):
    """``GET /ingest/directory/{job_id}``: the job, and its result once it has one."""

    job: JobView = Field(..., description='The job row: status, progress, errors')
    directory: str | None = Field(default=None, description='Source directory path')
    result: DirectoryIngestResponse | None = Field(
        default=None, description='The ingest result, once the job finished'
    )
