"""Response models for the /search endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Single search result with image metadata and similarity score."""

    image_id: str = Field(..., description='Unique image identifier')
    image_path: str | None = Field(default=None, description='File path to the image')
    score: float = Field(..., ge=0.0, le=1.0, description='Similarity score (cosine similarity)')
    metadata: dict | None = Field(default=None, description='Additional image metadata')


class SearchResponse(BaseModel):
    """Standard response for all visual search endpoints."""

    status: Literal['success', 'error'] = Field(..., description='Request status')
    query_type: str = Field(..., description='Type of search performed')
    results: list[SearchResult] = Field(
        default_factory=list, description='Search results ordered by score'
    )
    total_results: int = Field(..., ge=0, description='Number of results returned')
    search_time_ms: float = Field(..., ge=0.0, description='Search execution time in milliseconds')


class FaceSearchResult(SearchResult):
    """Extended search result for face similarity search."""

    face_id: str = Field(..., description='Unique face identifier')
    box: list[float] = Field(..., description='Face bounding box [x1, y1, x2, y2] normalized [0,1]')
    confidence: float = Field(..., ge=0.0, le=1.0, description='Original detection confidence')
    person_id: str | None = Field(default=None, description='Person cluster ID if assigned')
    person_name: str | None = Field(default=None, description='Person name if known')


class FaceSearchResponse(BaseModel):
    """Response for face similarity search with query face info."""

    status: Literal['success', 'error'] = Field(..., description='Request status')
    query_type: str = Field(default='face', description='Type of search performed')
    query_face: dict | None = Field(
        default=None, description='Query face info (box, landmarks, score)'
    )
    results: list[FaceSearchResult] = Field(
        default_factory=list, description='Search results ordered by score'
    )
    total_results: int = Field(..., ge=0, description='Number of results returned')
    search_time_ms: float = Field(..., ge=0.0, description='Search execution time in milliseconds')


class ObjectSearchResult(SearchResult):
    """Extended search result for object-level search."""

    box: list[float] = Field(
        ..., description='Object bounding box [x1, y1, x2, y2] normalized [0,1]'
    )
    class_id: int = Field(..., ge=0, description='COCO class ID')
    category: str = Field(..., description='Detection category (vehicle, person, other)')


class ObjectSearchResponse(BaseModel):
    """Response for object-level similarity search."""

    status: Literal['success', 'error'] = Field(..., description='Request status')
    query_type: str = Field(default='object', description='Type of search performed')
    query_object: dict | None = Field(
        default=None, description='Query object info (box, class_id, category)'
    )
    results: list[ObjectSearchResult] = Field(
        default_factory=list, description='Search results ordered by score'
    )
    total_results: int = Field(..., ge=0, description='Number of results returned')
    search_time_ms: float = Field(..., ge=0.0, description='Search execution time in milliseconds')


class OCRSearchResult(SearchResult):
    """Extended search result for OCR text search."""

    # Text relevance is a BM25 score, not a [0, 1] similarity.
    score: float = Field(..., ge=0.0, description='Text relevance score (BM25, unbounded)')
    matched_text: str = Field(..., description='Text that matched the query')
    text_box: list[float] | None = Field(
        default=None, description='Text bounding box [x1, y1, x2, y2] normalized'
    )
    full_text: str | None = Field(default=None, description='Full OCR text from the image')


class OCRSearchResponse(BaseModel):
    """Response for OCR text search."""

    status: Literal['success', 'error'] = Field(..., description='Request status')
    query_type: str = Field(default='ocr', description='Type of search performed')
    query_text: str = Field(..., description='Search query text')
    results: list[OCRSearchResult] = Field(
        default_factory=list, description='Search results ordered by relevance'
    )
    total_results: int = Field(..., ge=0, description='Number of results returned')
    search_time_ms: float = Field(..., ge=0.0, description='Search execution time in milliseconds')
