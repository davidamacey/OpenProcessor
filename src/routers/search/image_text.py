"""POST /search/image and POST /search/text: image-to-image and text-to-image search."""

import logging
import time

from fastapi import File, HTTPException, Query, UploadFile, status

from src.core.dependencies import VisualSearchDep
from src.routers.search._router import router
from src.routers.search.models import SearchResponse, SearchResult


logger = logging.getLogger(__name__)


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
