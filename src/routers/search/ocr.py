"""POST /search/ocr: search images by OCR text content."""

import logging
import time

from fastapi import HTTPException, Query, status

from src.core.dependencies import VisualSearchDep
from src.routers.search._router import router
from src.routers.search.models import OCRSearchResponse, OCRSearchResult


logger = logging.getLogger(__name__)


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
        description='Minimum text relevance score (BM25, unbounded)',
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
        min_score: Minimum text relevance score (BM25, unbounded)

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
                matched_text=r.get('text', ''),
                text_box=r.get('box_normalized'),
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
