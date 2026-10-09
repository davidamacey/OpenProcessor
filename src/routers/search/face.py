"""POST /search/face: face similarity search."""

import logging
import time

from fastapi import File, HTTPException, Query, UploadFile, status

from src.core.dependencies import VisualSearchDep
from src.routers.search._router import router
from src.routers.search.models import FaceSearchResponse, FaceSearchResult


logger = logging.getLogger(__name__)


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
    1. Detect faces in query image using SCRFD
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
