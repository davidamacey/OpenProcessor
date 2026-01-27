"""
OCR Router - Text Detection and Recognition.

Provides OCR endpoints using PP-OCRv5 for text extraction from images.

Endpoints:
- POST /ocr/predict - Extract text from single image
- POST /ocr/batch - Batch OCR processing
- POST /ocr/search - Search images by text content
"""

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.schemas.detection import ImageMetadata
from src.services.ocr_service import OcrService


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/ocr',
    tags=['OCR Text Extraction'],
    default_response_class=ORJSONResponse,
)


# =============================================================================
# Response Models
# =============================================================================


class TextRegion(BaseModel):
    """Single detected text region."""

    text: str = Field(..., description='Recognized text string')
    box: list[float] = Field(..., description='Quad box [x1,y1,x2,y2,x3,y3,x4,y4] in pixels')
    box_normalized: list[float] = Field(
        ..., description='Axis-aligned box [x1,y1,x2,y2] normalized [0,1]'
    )
    det_score: float = Field(..., description='Detection confidence score')
    rec_score: float = Field(..., description='Recognition confidence score')


class OcrPredictResponse(BaseModel):
    """Response for single image OCR."""

    status: str = Field(..., description="'success' or 'error'")

    # OCR results
    texts: list[str] = Field(default_factory=list, description='All detected text strings')
    regions: list[TextRegion] = Field(
        default_factory=list, description='Text regions with boxes and scores'
    )
    full_text: str = Field(default='', description='All text concatenated')
    num_texts: int = Field(default=0, description='Number of text regions detected')

    # Image metadata
    image: ImageMetadata | None = Field(default=None, description='Original image dimensions')

    # Timing
    total_time_ms: float | None = Field(default=None, description='Processing time in ms')

    # Error info
    error: str | None = Field(default=None, description='Error message if failed')


class BatchOcrResult(BaseModel):
    """Single image result in batch OCR."""

    filename: str = Field(..., description='Original filename')
    image_index: int = Field(..., description='Index in batch')
    status: str = Field(default='success', description="'success' or 'error'")
    error: str | None = Field(default=None, description='Error message if failed')

    # OCR results
    texts: list[str] = Field(default_factory=list, description='Detected text strings')
    full_text: str = Field(default='', description='All text concatenated')
    num_texts: int = Field(default=0, description='Number of text regions')

    # Image metadata
    image: ImageMetadata | None = Field(default=None, description='Image dimensions')


class BatchOcrResponse(BaseModel):
    """Response for batch OCR processing."""

    status: str = Field(..., description='Overall status')
    total_images: int = Field(..., description='Total images in request')
    processed_images: int = Field(..., description='Successfully processed')
    failed_images: int = Field(..., description='Failed count')

    results: list[BatchOcrResult] = Field(default_factory=list, description='Per-image results')
    failures: list[dict] | None = Field(default=None, description='Failure details')

    # Aggregated stats
    total_text_regions: int = Field(default=0, description='Sum of text regions')
    images_with_text: int = Field(default=0, description='Images where text was found')

    total_time_ms: float | None = Field(default=None, description='Total processing time in ms')


# =============================================================================
# Endpoints
# =============================================================================


@router.post('/predict', response_model=OcrPredictResponse)
def ocr_predict(
    image: Annotated[UploadFile, File(description='Image file (JPEG/PNG)')],
    min_det_score: Annotated[
        float, Query(ge=0.0, le=1.0, description='Minimum detection confidence')
    ] = 0.5,
    min_rec_score: Annotated[
        float, Query(ge=0.0, le=1.0, description='Minimum recognition confidence')
    ] = 0.8,
    filter_by_score: Annotated[
        bool, Query(description='Filter results by confidence thresholds')
    ] = True,
):
    """
    Extract text from a single image using PP-OCRv5.

    Pipeline:
    1. Image preprocessing
    2. PP-OCRv5 detection (text localization)
    3. PP-OCRv5 recognition (text decoding)
    4. Results with quad boxes and confidence scores

    PP-OCRv5 supports multiple languages and handles various text styles
    including curved text, rotated text, and dense text layouts.

    Args:
        image: Image file (JPEG/PNG)
        min_det_score: Minimum detection confidence (default: 0.5)
        min_rec_score: Minimum recognition confidence (default: 0.8)
        filter_by_score: Filter results by confidence thresholds

    Returns:
        OCR results with text regions, boxes, and scores.
    """
    filename = image.filename or 'uploaded_image'

    try:
        image_bytes = image.file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail='Empty image file')

        # Get OCR service with configured thresholds
        service = OcrService(min_det_score=min_det_score, min_rec_score=min_rec_score)

        result = service.extract_text(image_bytes, filter_by_score=filter_by_score)

        if result.get('status') == 'error':
            return OcrPredictResponse(
                status='error',
                error=result.get('error', 'OCR extraction failed'),
            )

        # Build response
        texts = result.get('texts', [])
        boxes = result.get('boxes', [])
        boxes_normalized = result.get('boxes_normalized', [])
        det_scores = result.get('det_scores', [])
        rec_scores = result.get('rec_scores', [])

        regions = [
            TextRegion(
                text=texts[i] if i < len(texts) else '',
                box=boxes[i] if i < len(boxes) else [],
                box_normalized=boxes_normalized[i] if i < len(boxes_normalized) else [],
                det_score=det_scores[i] if i < len(det_scores) else 0.0,
                rec_score=rec_scores[i] if i < len(rec_scores) else 0.0,
            )
            for i in range(result.get('num_texts', 0))
        ]

        image_size = result.get('image_size', [0, 0])

        return OcrPredictResponse(
            status='success',
            texts=result.get('texts', []),
            regions=regions,
            full_text=' '.join(result.get('texts', [])),
            num_texts=result.get('num_texts', 0),
            image=ImageMetadata(width=image_size[1], height=image_size[0])
            if image_size[0] > 0
            else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'OCR failed for {filename}: {e}')
        raise HTTPException(status_code=500, detail=f'OCR extraction failed: {e!s}') from e


@router.post('/batch', response_model=BatchOcrResponse)
def ocr_batch(
    images: Annotated[list[UploadFile], File(description='Image files (JPEG/PNG), max 32')],
    min_det_score: Annotated[
        float, Query(ge=0.0, le=1.0, description='Minimum detection confidence')
    ] = 0.5,
    min_rec_score: Annotated[
        float, Query(ge=0.0, le=1.0, description='Minimum recognition confidence')
    ] = 0.8,
    max_workers: Annotated[int, Query(ge=1, le=32, description='Parallel processing threads')] = 16,
):
    """
    Batch OCR processing for multiple images.

    Processes up to 32 images in parallel using thread pool execution.
    Optimized for throughput when processing many documents.

    Args:
        images: List of image files (max 32)
        min_det_score: Minimum detection confidence
        min_rec_score: Minimum recognition confidence
        max_workers: Parallel processing threads

    Returns:
        Per-image OCR results and summary statistics.
    """
    MAX_BATCH = 32

    if len(images) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f'Batch size {len(images)} exceeds maximum {MAX_BATCH}',
        )

    if len(images) == 0:
        return BatchOcrResponse(
            status='success',
            total_images=0,
            processed_images=0,
            failed_images=0,
            results=[],
        )

    # Read all images first
    images_data = []
    failures = []

    for idx, img in enumerate(images):
        filename = img.filename or f'image_{idx}'
        try:
            image_bytes = img.file.read()
            if not image_bytes:
                failures.append(
                    {
                        'filename': filename,
                        'index': idx,
                        'error': 'Empty image file',
                    }
                )
                continue
            images_data.append((image_bytes, filename, idx))
        except Exception as e:
            failures.append(
                {
                    'filename': filename,
                    'index': idx,
                    'error': str(e),
                }
            )

    if not images_data:
        return BatchOcrResponse(
            status='all_failed',
            total_images=len(images),
            processed_images=0,
            failed_images=len(failures),
            results=[],
            failures=failures,
        )

    # Process with OCR service batch method
    service = OcrService(min_det_score=min_det_score, min_rec_score=min_rec_score)

    image_bytes_list = [d[0] for d in images_data]
    ocr_results = service.extract_text_batch(
        image_bytes_list,
        filter_by_score=True,
        max_workers=min(max_workers, len(images_data)),
    )

    # Build response
    results = []
    total_text_regions = 0
    images_with_text = 0

    for i, ocr_result in enumerate(ocr_results):
        _, filename, original_idx = images_data[i]

        if ocr_result.get('status') == 'error':
            failures.append(
                {
                    'filename': filename,
                    'index': original_idx,
                    'error': ocr_result.get('error', 'OCR failed'),
                }
            )
            continue

        num_texts = ocr_result.get('num_texts', 0)
        total_text_regions += num_texts
        if num_texts > 0:
            images_with_text += 1

        image_size = ocr_result.get('image_size', [0, 0])

        results.append(
            BatchOcrResult(
                filename=filename,
                image_index=original_idx,
                status='success',
                texts=ocr_result.get('texts', []),
                full_text=' '.join(ocr_result.get('texts', [])),
                num_texts=num_texts,
                image=ImageMetadata(width=image_size[1], height=image_size[0])
                if image_size[0] > 0
                else None,
            )
        )

    return BatchOcrResponse(
        status='success' if results else 'all_failed',
        total_images=len(images),
        processed_images=len(results),
        failed_images=len(failures),
        results=results,
        failures=failures if failures else None,
        total_text_regions=total_text_regions,
        images_with_text=images_with_text,
    )


@router.post('/search')
async def search_by_ocr(
    text: str = Query(..., min_length=1, description='Text to search for'),
    page: int = Query(0, ge=0, description='Page number (0-indexed)'),
    size: int = Query(20, ge=1, le=100, description='Results per page'),
    fuzzy: bool = Query(True, description='Enable fuzzy matching'),
):
    """
    Search images by OCR text content.

    Searches the visual_search_ocr index for images containing the specified
    text. Uses OpenSearch full-text search with optional fuzzy matching.

    Args:
        text: Text to search for
        page: Page number (0-indexed)
        size: Results per page (max 100)
        fuzzy: Enable fuzzy matching for typos

    Returns:
        Images containing matching text.
    """
    # Import here to avoid circular imports
    from src.core.dependencies import get_visual_search_service

    try:
        search_service = await get_visual_search_service()
        client = search_service.opensearch.client

        from src.clients.opensearch import IndexName

        # Build search query
        if fuzzy:
            query = {
                'multi_match': {
                    'query': text,
                    'fields': ['full_text', 'texts'],
                    'fuzziness': 'AUTO',
                }
            }
        else:
            query = {
                'multi_match': {
                    'query': text,
                    'fields': ['full_text', 'texts'],
                }
            }

        response = await client.search(
            index=IndexName.OCR.value,
            body={
                'query': query,
                'from': page * size,
                'size': size,
                '_source': ['image_id', 'image_path', 'full_text', 'texts', 'num_texts'],
                'highlight': {
                    'fields': {
                        'full_text': {},
                        'texts': {},
                    }
                },
            },
        )

        hits = response.get('hits', {})
        total = hits.get('total', {}).get('value', 0)

        results = []
        for hit in hits.get('hits', []):
            source = hit['_source']
            highlight = hit.get('highlight', {})

            results.append(
                {
                    'image_id': source.get('image_id', ''),
                    'image_path': source.get('image_path', ''),
                    'score': hit.get('_score', 0),
                    'full_text': source.get('full_text', ''),
                    'num_texts': source.get('num_texts', 0),
                    'highlight': highlight.get('full_text', highlight.get('texts', [])),
                }
            )

        return {
            'status': 'success',
            'query': text,
            'total_results': total,
            'page': page,
            'size': size,
            'results': results,
        }

    except Exception as e:
        logger.error(f'OCR search failed for "{text}": {e}')
        raise HTTPException(status_code=500, detail=f'Search failed: {e!s}') from e
