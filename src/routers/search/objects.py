"""POST /search/object: object-level similarity (vehicles, people)."""

import logging
import time

from fastapi import File, HTTPException, Query, UploadFile, status

from src.core.dependencies import VisualSearchDep
from src.routers.search._router import router
from src.routers.search.models import ObjectSearchResponse, ObjectSearchResult


logger = logging.getLogger(__name__)


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
