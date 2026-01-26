"""
Visual Search Router.

Provides visual similarity search endpoints using MobileCLIP and ArcFace embeddings
with OpenSearch k-NN backend.

Endpoints:
- POST /search/image - Image-to-image similarity search
- POST /search/text - Text-to-image search (CLIP text search)
- POST /search/face - Face similarity search
- POST /search/ocr - Search images by text content
- POST /search/object - Object-level similarity (vehicles, people)

All endpoints use the VisualSearchService for OpenSearch operations and
InferenceService for embedding generation.
"""

import logging
import time
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.core.dependencies import VisualSearchDep


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/search',
    tags=['Visual Search'],
    default_response_class=ORJSONResponse,
)


# =============================================================================
# Response Models
# =============================================================================


class SearchResult(BaseModel):
    """Single search result with image metadata and similarity score."""

    image_id: str = Field(..., description='Unique image identifier')
    image_path: str | None = Field(None, description='File path to the image')
    score: float = Field(..., ge=0.0, le=1.0, description='Similarity score (cosine similarity)')
    metadata: dict | None = Field(None, description='Additional image metadata')


class SearchResponse(BaseModel):
    """Standard response for all visual search endpoints."""

    status: Literal['success', 'error'] = Field(..., description='Request status')
    query_type: str = Field(..., description='Type of search performed')
    results: list[SearchResult] = Field(default_factory=list, description='Search results ordered by score')
    total_results: int = Field(..., ge=0, description='Number of results returned')
    search_time_ms: float = Field(..., ge=0.0, description='Search execution time in milliseconds')


class FaceSearchResult(SearchResult):
    """Extended search result for face similarity search."""

    face_id: str = Field(..., description='Unique face identifier')
    box: list[float] = Field(..., description='Face bounding box [x1, y1, x2, y2] normalized [0,1]')
    confidence: float = Field(..., ge=0.0, le=1.0, description='Original detection confidence')
    person_id: str | None = Field(None, description='Person cluster ID if assigned')
    person_name: str | None = Field(None, description='Person name if known')


class FaceSearchResponse(BaseModel):
    """Response for face similarity search with query face info."""

    status: Literal['success', 'error'] = Field(..., description='Request status')
    query_type: str = Field(default='face', description='Type of search performed')
    query_face: dict | None = Field(None, description='Query face info (box, landmarks, score)')
    results: list[FaceSearchResult] = Field(default_factory=list, description='Search results ordered by score')
    total_results: int = Field(..., ge=0, description='Number of results returned')
    search_time_ms: float = Field(..., ge=0.0, description='Search execution time in milliseconds')


class ObjectSearchResult(SearchResult):
    """Extended search result for object-level search."""

    box: list[float] = Field(..., description='Object bounding box [x1, y1, x2, y2] normalized [0,1]')
    class_id: int = Field(..., ge=0, description='COCO class ID')
    category: str = Field(..., description='Detection category (vehicle, person, other)')


class ObjectSearchResponse(BaseModel):
    """Response for object-level similarity search."""

    status: Literal['success', 'error'] = Field(..., description='Request status')
    query_type: str = Field(default='object', description='Type of search performed')
    query_object: dict | None = Field(None, description='Query object info (box, class_id, category)')
    results: list[ObjectSearchResult] = Field(default_factory=list, description='Search results ordered by score')
    total_results: int = Field(..., ge=0, description='Number of results returned')
    search_time_ms: float = Field(..., ge=0.0, description='Search execution time in milliseconds')


class OCRSearchResult(SearchResult):
    """Extended search result for OCR text search."""

    matched_text: str = Field(..., description='Text that matched the query')
    text_box: list[float] | None = Field(None, description='Text bounding box [x1, y1, x2, y2] normalized')
    full_text: str | None = Field(None, description='Full OCR text from the image')


class OCRSearchResponse(BaseModel):
    """Response for OCR text search."""

    status: Literal['success', 'error'] = Field(..., description='Request status')
    query_type: str = Field(default='ocr', description='Type of search performed')
    query_text: str = Field(..., description='Search query text')
    results: list[OCRSearchResult] = Field(default_factory=list, description='Search results ordered by relevance')
    total_results: int = Field(..., ge=0, description='Number of results returned')
    search_time_ms: float = Field(..., ge=0.0, description='Search execution time in milliseconds')


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    '/image',
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary='Image-to-image similarity search',
    description='Find visually similar images using MobileCLIP embeddings. '
                'Encodes the query image and searches the global visual search index.',
    response_description='List of similar images with similarity scores',
)
async def search_by_image(
    search_service: VisualSearchDep,
    image: UploadFile = File(..., description='Query image file (JPEG/PNG)'),
    top_k: int = Query(
        10,
        ge=1,
        le=100,
        description='Maximum number of results to return',
    ),
    min_score: float = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description='Minimum similarity score threshold',
    ),
) -> SearchResponse:
    """
    Find visually similar images using MobileCLIP global embeddings.

    Pipeline:
    1. Encode query image via MobileCLIP image encoder
    2. k-NN search on visual_search_global index in OpenSearch
    3. Return top-k results above min_score threshold

    Use cases:
    - Find similar scenes or compositions
    - Discover related images in a collection
    - Reverse image search

    Args:
        image: Query image file (JPEG/PNG)
        top_k: Maximum number of results to return (1-100)
        min_score: Minimum similarity score threshold (0.0-1.0)

    Returns:
        SearchResponse with similar images ordered by similarity score.

    Raises:
        HTTPException 400: Empty or invalid image file
        HTTPException 500: Search or embedding generation failed
    """
    start_time = time.perf_counter()

    try:
        image_bytes = image.file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Empty image file',
            )

        # Search using visual search service
        results = await search_service.search_by_image(
            image_bytes=image_bytes,
            top_k=top_k,
            min_score=min_score,
        )

        search_time_ms = (time.perf_counter() - start_time) * 1000

        # Format results
        formatted_results = [
            SearchResult(
                image_id=r.get('image_id', ''),
                image_path=r.get('image_path'),
                score=r.get('score', 0.0),
                metadata=r.get('metadata'),
            )
            for r in results
        ]

        return SearchResponse(
            status='success',
            query_type='image',
            results=formatted_results,
            total_results=len(formatted_results),
            search_time_ms=round(search_time_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Image search failed: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Image search failed: {e!s}',
        ) from e


@router.post(
    '/text',
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary='Text-to-image search',
    description='Search for images using natural language queries. '
                'Encodes the text query using MobileCLIP and searches the global index.',
    response_description='List of matching images with similarity scores',
)
async def search_by_text(
    search_service: VisualSearchDep,
    text: str = Query(
        ...,
        min_length=1,
        max_length=500,
        description='Search query text (natural language)',
    ),
    top_k: int = Query(
        10,
        ge=1,
        le=100,
        description='Maximum number of results to return',
    ),
    min_score: float = Query(
        0.2,
        ge=0.0,
        le=1.0,
        description='Minimum similarity score threshold',
    ),
    use_cache: bool = Query(
        True,
        description='Use cached text embeddings for faster repeated queries',
    ),
) -> SearchResponse:
    """
    Search images using natural language text queries (CLIP text-to-image).

    Pipeline:
    1. Tokenize and encode query text via MobileCLIP text encoder
    2. k-NN search on visual_search_global index in OpenSearch
    3. Return top-k results above min_score threshold

    Use cases:
    - Semantic image search ("beach sunset", "red sports car")
    - Find images by content description
    - Zero-shot image retrieval

    Note: Text-to-image scores are typically lower than image-to-image.
    A min_score of 0.2 is usually appropriate for text queries.

    Args:
        text: Search query in natural language (max 500 chars, truncated to 77 tokens)
        top_k: Maximum number of results to return (1-100)
        min_score: Minimum similarity score threshold (0.0-1.0)
        use_cache: Use cached text embeddings for repeated queries

    Returns:
        SearchResponse with matching images ordered by relevance score.

    Raises:
        HTTPException 400: Empty or invalid query text
        HTTPException 500: Search or embedding generation failed
    """
    start_time = time.perf_counter()

    try:
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Query text cannot be empty',
            )

        # Search using visual search service
        results = await search_service.search_by_text(
            text=text.strip(),
            top_k=top_k,
            min_score=min_score,
            use_cache=use_cache,
        )

        search_time_ms = (time.perf_counter() - start_time) * 1000

        # Format results
        formatted_results = [
            SearchResult(
                image_id=r.get('image_id', ''),
                image_path=r.get('image_path'),
                score=r.get('score', 0.0),
                metadata=r.get('metadata'),
            )
            for r in results
        ]

        return SearchResponse(
            status='success',
            query_type='text',
            results=formatted_results,
            total_results=len(formatted_results),
            search_time_ms=round(search_time_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Text search failed: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Text search failed: {e!s}',
        ) from e


@router.post(
    '/face',
    response_model=FaceSearchResponse,
    status_code=status.HTTP_200_OK,
    summary='Face similarity search',
    description='Find similar faces using ArcFace embeddings. '
                'Detects faces in the query image and searches the face index.',
    response_description='List of similar faces with identity scores',
)
async def search_by_face(
    search_service: VisualSearchDep,
    image: UploadFile = File(..., description='Query image file with face (JPEG/PNG)'),
    face_index: int = Query(
        0,
        ge=0,
        description='Which detected face to use as query (0-indexed)',
    ),
    top_k: int = Query(
        10,
        ge=1,
        le=100,
        description='Maximum number of results to return',
    ),
    min_score: float = Query(
        0.7,
        ge=0.0,
        le=1.0,
        description='Minimum similarity score (0.7 recommended for identity matching)',
    ),
) -> FaceSearchResponse:
    """
    Find similar faces using ArcFace identity embeddings.

    Pipeline:
    1. Detect faces in query image using YOLO11-face
    2. Extract ArcFace 512-dim embedding for selected face
    3. k-NN search on visual_search_faces index in OpenSearch
    4. Return top-k faces above min_score threshold

    Use cases:
    - Find all photos of a specific person
    - Face identification (1:N matching)
    - Photo organization by face

    Threshold guidelines:
    - 0.7+: High confidence identity match
    - 0.6: Balanced precision/recall
    - 0.5: More permissive (may include siblings/lookalikes)

    Args:
        image: Query image file containing a face (JPEG/PNG)
        face_index: Which detected face to use as query (0 = most confident)
        top_k: Maximum number of results to return (1-100)
        min_score: Minimum similarity score threshold (0.7 recommended)

    Returns:
        FaceSearchResponse with query face info and similar faces.

    Raises:
        HTTPException 400: No faces detected or invalid face_index
        HTTPException 500: Face detection or search failed
    """
    start_time = time.perf_counter()

    try:
        image_bytes = image.file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Empty image file',
            )

        # Use search_faces_by_image from visual search service
        result = await search_service.search_faces_by_image(
            image_bytes=image_bytes,
            face_index=face_index,
            top_k=top_k,
            min_score=min_score,
        )

        search_time_ms = (time.perf_counter() - start_time) * 1000

        if result.get('status') == 'error':
            error_msg = result.get('error', 'Face search failed')
            if 'No faces detected' in error_msg:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg,
                )
            if 'out of range' in error_msg:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg,
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg,
            )

        # Format results
        formatted_results = [
            FaceSearchResult(
                image_id=r.get('image_id', ''),
                image_path=r.get('image_path'),
                score=r.get('score', 0.0),
                metadata=r.get('metadata'),
                face_id=r.get('face_id', r.get('_id', '')),
                box=r.get('box', [0, 0, 0, 0]),
                confidence=r.get('confidence', 0.0),
                person_id=r.get('person_id'),
                person_name=r.get('person_name'),
            )
            for r in result.get('results', [])
        ]

        return FaceSearchResponse(
            status='success',
            query_type='face',
            query_face=result.get('query_face'),
            results=formatted_results,
            total_results=len(formatted_results),
            search_time_ms=round(search_time_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Face search failed: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Face search failed: {e!s}',
        ) from e


@router.post(
    '/ocr',
    response_model=OCRSearchResponse,
    status_code=status.HTTP_200_OK,
    summary='Search images by text content',
    description='Find images containing specific text using OCR index. '
                'Searches the full-text OCR index with trigram matching.',
    response_description='List of images containing matching text',
)
async def search_by_ocr(
    search_service: VisualSearchDep,
    text: str = Query(
        ...,
        min_length=1,
        max_length=500,
        description='Text to search for in images',
    ),
    top_k: int = Query(
        10,
        ge=1,
        le=100,
        description='Maximum number of results to return',
    ),
    min_score: float = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description='Minimum relevance score threshold',
    ),
) -> OCRSearchResponse:
    """
    Search for images containing specific text using OCR index.

    Pipeline:
    1. Query OpenSearch visual_search_ocr index using full-text search
    2. Match against extracted text content (trigram-based fuzzy matching)
    3. Return images with matching text ordered by relevance

    Use cases:
    - Find screenshots with specific error messages
    - Search for images with signs, labels, or documents
    - Locate photos with visible text

    The OCR index supports:
    - Exact phrase matching
    - Partial word matching (trigram-based)
    - Case-insensitive search

    Args:
        text: Text to search for in images (supports partial matching)
        top_k: Maximum number of results to return (1-100)
        min_score: Minimum relevance score threshold (0.0-1.0)

    Returns:
        OCRSearchResponse with images containing matching text.

    Raises:
        HTTPException 400: Empty query text
        HTTPException 500: OCR search failed
    """
    start_time = time.perf_counter()

    try:
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Query text cannot be empty',
            )

        # Search OCR index via OpenSearch
        results = await search_service.opensearch.search_ocr(
            query_text=text.strip(),
            top_k=top_k,
            min_score=min_score,
        )

        search_time_ms = (time.perf_counter() - start_time) * 1000

        # Format results
        formatted_results = [
            OCRSearchResult(
                image_id=r.get('image_id', ''),
                image_path=r.get('image_path'),
                score=r.get('score', 0.0),
                metadata=r.get('metadata'),
                matched_text=r.get('matched_text', r.get('full_text', '')),
                text_box=r.get('text_box'),
                full_text=r.get('full_text'),
            )
            for r in results
        ]

        return OCRSearchResponse(
            status='success',
            query_type='ocr',
            query_text=text.strip(),
            results=formatted_results,
            total_results=len(formatted_results),
            search_time_ms=round(search_time_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'OCR search failed: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'OCR search failed: {e!s}',
        ) from e


@router.post(
    '/object',
    response_model=ObjectSearchResponse,
    status_code=status.HTTP_200_OK,
    summary='Object-level similarity search',
    description='Find similar objects (vehicles, people) using per-detection embeddings. '
                'Automatically routes to vehicles or people index based on detection class.',
    response_description='List of similar objects with similarity scores',
)
async def search_by_object(
    search_service: VisualSearchDep,
    image: UploadFile = File(..., description='Query image file (JPEG/PNG)'),
    box_index: int = Query(
        0,
        ge=0,
        description='Which detected object to use as query (0-indexed)',
    ),
    top_k: int = Query(
        10,
        ge=1,
        le=100,
        description='Maximum number of results to return',
    ),
    min_score: float = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description='Minimum similarity score threshold',
    ),
    class_filter: list[int] | None = Query(
        None,
        description='Filter by COCO class IDs [2=car, 3=motorcycle, 5=bus, 7=truck, 8=boat]',
    ),
) -> ObjectSearchResponse:
    """
    Find similar objects using per-detection MobileCLIP embeddings.

    Pipeline:
    1. Run YOLO detection on query image
    2. Extract MobileCLIP embedding for selected detection box
    3. Auto-route to appropriate index based on class:
       - Vehicles (car, truck, motorcycle, bus, boat) -> visual_search_vehicles
       - People -> visual_search_people
       - Other classes -> visual_search_global (fallback)
    4. k-NN search and return top-k results

    Use cases:
    - Find similar vehicles ("red sports cars like this one")
    - Find people with similar appearance (clothing, pose)
    - Object-level reverse search

    Supported categories:
    - Vehicles: car (2), motorcycle (3), bus (5), truck (7), boat (8)
    - People: person (0)

    Args:
        image: Query image file containing objects (JPEG/PNG)
        box_index: Which detected object to use as query (0 = first/most confident)
        top_k: Maximum number of results to return (1-100)
        min_score: Minimum similarity score threshold (0.0-1.0)
        class_filter: Optional filter by COCO class IDs

    Returns:
        ObjectSearchResponse with query object info and similar objects.

    Raises:
        HTTPException 400: No objects detected or invalid box_index
        HTTPException 500: Detection or search failed
    """
    start_time = time.perf_counter()

    try:
        image_bytes = image.file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Empty image file',
            )

        # Use search_by_object from visual search service
        result = await search_service.search_by_object(
            image_bytes=image_bytes,
            box_index=box_index,
            top_k=top_k,
            min_score=min_score,
            class_filter=class_filter,
        )

        search_time_ms = (time.perf_counter() - start_time) * 1000

        if result.get('status') == 'error':
            error_msg = result.get('error', 'Object search failed')
            if 'No objects detected' in error_msg:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg,
                )
            if 'out of range' in error_msg:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg,
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg,
            )

        # Format results
        formatted_results = [
            ObjectSearchResult(
                image_id=r.get('image_id', ''),
                image_path=r.get('image_path'),
                score=r.get('score', 0.0),
                metadata=r.get('metadata'),
                box=r.get('box', [0, 0, 0, 0]),
                class_id=r.get('class_id', 0),
                category=r.get('category', 'other'),
            )
            for r in result.get('results', [])
        ]

        return ObjectSearchResponse(
            status='success',
            query_type='object',
            query_object=result.get('query_object'),
            results=formatted_results,
            total_results=len(formatted_results),
            search_time_ms=round(search_time_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Object search failed: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Object search failed: {e!s}',
        ) from e
