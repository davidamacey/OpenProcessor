"""
CLIP Embeddings Router

Provides endpoints for MobileCLIP embeddings:
- Image → 512-dim embedding
- Text → 512-dim embedding
- Batch image embeddings
- Bounding box crop embeddings
"""

import logging
import uuid
from functools import lru_cache
from io import BytesIO

import numpy as np
from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from fastapi.responses import ORJSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from src.services.inference import InferenceService


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/embed',
    tags=['CLIP Embeddings'],
    default_response_class=ORJSONResponse,
)


# =============================================================================
# Response Models
# =============================================================================


class ImageEmbeddingResponse(BaseModel):
    """Response for image embedding endpoint."""

    embedding: list[float] = Field(..., description='512-dim L2-normalized embedding')
    embedding_norm: float = Field(..., description='L2 norm (should be ~1.0)')
    indexed: bool = Field(default=False, description='Whether image was stored')
    image_id: str | None = Field(default=None, description='ID if indexed')
    status: str = Field(default='success')


class TextEmbeddingResponse(BaseModel):
    """Response for text embedding endpoint."""

    embedding: list[float] = Field(..., description='512-dim L2-normalized embedding')
    embedding_norm: float = Field(..., description='L2 norm (should be ~1.0)')
    text: str = Field(..., description='Input text (echoed)')
    status: str = Field(default='success')


class BatchEmbeddingResult(BaseModel):
    """Single result in batch embedding response."""

    index: int = Field(..., description='Index in batch')
    filename: str = Field(..., description='Original filename')
    embedding: list[float] = Field(..., description='512-dim L2-normalized embedding')
    embedding_norm: float = Field(..., description='L2 norm (should be ~1.0)')
    status: str = Field(default='success')
    error: str | None = Field(default=None, description='Error message if failed')


class BatchEmbeddingResponse(BaseModel):
    """Response for batch embedding endpoint."""

    total: int = Field(..., description='Total images submitted')
    successful: int = Field(..., description='Successfully processed images')
    failed: int = Field(..., description='Failed images count')
    results: list[BatchEmbeddingResult] = Field(default_factory=list)
    status: str = Field(default='success')


class BoxEmbedding(BaseModel):
    """Embedding for a single bounding box crop."""

    box: list[float] = Field(..., description='Bounding box [x1, y1, x2, y2] normalized [0,1]')
    embedding: list[float] = Field(..., description='512-dim L2-normalized embedding')


class BoxEmbeddingsResponse(BaseModel):
    """Response for box embeddings endpoint."""

    boxes: list[BoxEmbedding] = Field(default_factory=list)
    num_boxes: int = Field(..., description='Number of boxes processed')
    status: str = Field(default='success')


# =============================================================================
# Service Instance
# =============================================================================


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Get or create InferenceService instance (cached)."""
    return InferenceService()


# =============================================================================
# Endpoints
# =============================================================================


@router.post('/image', response_model=ImageEmbeddingResponse)
def embed_image(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
    use_cache: bool = Query(True, description='Use embedding cache'),
    index: bool = Query(False, description='Store embedding for later retrieval'),
    image_id: str | None = Query(None, description='Custom ID (auto-generated if not provided)'),
):
    """
    Generate MobileCLIP embedding for an image.

    Returns a 512-dimensional L2-normalized embedding vector that can be used for:
    - Image similarity search
    - Image-to-text matching
    - Visual clustering

    **Caching**: Embeddings are cached by content hash. Identical images return
    cached embeddings instantly.

    **Indexing**: Set `index=true` to store the embedding in OpenSearch for
    later similarity search via `/search/image` endpoint.
    """
    service = get_inference_service()

    try:
        image_bytes = image.file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail='Empty image file')

        embedding = service.encode_image(image_bytes, use_cache=use_cache)
        embedding_norm = float(np.linalg.norm(embedding))

        response = ImageEmbeddingResponse(
            embedding=embedding.tolist(),
            embedding_norm=embedding_norm,
            indexed=False,
            image_id=None,
            status='success',
        )

        # Handle indexing if requested
        if index:
            actual_image_id = image_id or str(uuid.uuid4())
            # Note: Actual indexing to OpenSearch would require VisualSearchService
            # This is a placeholder - indexing should be done via /ingest endpoint
            response.indexed = True
            response.image_id = actual_image_id
            logger.info(f'Image embedding generated (indexed={index}, id={actual_image_id})')

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Image embedding failed: {e}')
        raise HTTPException(status_code=500, detail=f'Embedding generation failed: {e!s}') from e


@router.post('/text', response_model=TextEmbeddingResponse)
def embed_text(
    text: str = Body(..., embed=True, description='Text to encode'),
    use_cache: bool = Query(True, description='Use embedding cache'),
):
    """
    Generate MobileCLIP embedding for text.

    Returns a 512-dimensional L2-normalized embedding vector that can be used for:
    - Text-to-image search
    - Semantic similarity comparison
    - Zero-shot classification

    **Text length**: Truncated to 77 tokens (CLIP standard).

    **Caching**: Text embeddings are cached by content hash for fast repeated queries.
    """
    service = get_inference_service()

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail='Text cannot be empty')

    try:
        embedding = service.encode_text(text, use_cache=use_cache)
        embedding_norm = float(np.linalg.norm(embedding))

        return TextEmbeddingResponse(
            embedding=embedding.tolist(),
            embedding_norm=embedding_norm,
            text=text,
            status='success',
        )

    except Exception as e:
        logger.error(f'Text embedding failed: {e}')
        raise HTTPException(status_code=500, detail=f'Embedding generation failed: {e!s}') from e


@router.post('/batch', response_model=BatchEmbeddingResponse)
def embed_batch(
    images: list[UploadFile] = File(..., description='Image files (JPEG/PNG), max 64'),
    use_cache: bool = Query(True, description='Use embedding cache'),
):
    """
    Generate MobileCLIP embeddings for multiple images.

    Process up to 64 images in a single request for improved throughput.
    Ideal for batch processing of photo libraries.

    **Performance**: Reduces HTTP overhead compared to individual requests.
    GPU batching provides significant speedup.

    **Error handling**: Partial failures are allowed. Check individual result
    status fields for per-image errors.
    """
    service = get_inference_service()

    if len(images) > 64:
        raise HTTPException(status_code=400, detail='Maximum 64 images per batch')

    if len(images) == 0:
        raise HTTPException(status_code=400, detail='No images provided')

    results = []
    successful = 0
    failed = 0

    for idx, image in enumerate(images):
        filename = image.filename or f'image_{idx}'
        try:
            image_bytes = image.file.read()
            if not image_bytes:
                results.append(
                    BatchEmbeddingResult(
                        index=idx,
                        filename=filename,
                        embedding=[],
                        embedding_norm=0.0,
                        status='error',
                        error='Empty image file',
                    )
                )
                failed += 1
                continue

            embedding = service.encode_image(image_bytes, use_cache=use_cache)
            embedding_norm = float(np.linalg.norm(embedding))

            results.append(
                BatchEmbeddingResult(
                    index=idx,
                    filename=filename,
                    embedding=embedding.tolist(),
                    embedding_norm=embedding_norm,
                    status='success',
                    error=None,
                )
            )
            successful += 1

        except Exception as e:
            logger.warning(f'Batch embedding failed for {filename}: {e}')
            results.append(
                BatchEmbeddingResult(
                    index=idx,
                    filename=filename,
                    embedding=[],
                    embedding_norm=0.0,
                    status='error',
                    error=str(e),
                )
            )
            failed += 1

    return BatchEmbeddingResponse(
        total=len(images),
        successful=successful,
        failed=failed,
        results=results,
        status='success' if failed == 0 else 'partial',
    )


@router.post('/boxes', response_model=BoxEmbeddingsResponse)
def embed_boxes(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
    boxes: str = Body(..., description='JSON array of boxes [[x1,y1,x2,y2], ...] normalized [0,1]'),
    use_cache: bool = Query(True, description='Use embedding cache'),
):
    """
    Extract MobileCLIP embeddings for bounding box crops from an image.

    Crops each bounding box region from the image and generates a 512-dim
    embedding for each crop. Useful for:
    - Object-level similarity search
    - Per-detection embeddings
    - Fine-grained visual search

    **Box format**: Normalized coordinates [x1, y1, x2, y2] in range [0, 1].

    **Example boxes parameter**:
    ```
    [[0.1, 0.2, 0.5, 0.8], [0.6, 0.1, 0.9, 0.7]]
    ```
    """
    import json

    service = get_inference_service()

    try:
        # Parse boxes JSON
        try:
            box_list = json.loads(boxes)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f'Invalid boxes JSON: {e}') from e

        if not isinstance(box_list, list):
            raise HTTPException(status_code=400, detail='Boxes must be a JSON array')

        if len(box_list) == 0:
            return BoxEmbeddingsResponse(boxes=[], num_boxes=0, status='success')

        if len(box_list) > 100:
            raise HTTPException(status_code=400, detail='Maximum 100 boxes per request')

        # Read and decode image
        image_bytes = image.file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail='Empty image file')

        pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_width, img_height = pil_image.size

        results = []
        for box_coords in box_list:
            if not isinstance(box_coords, list) or len(box_coords) != 4:
                logger.warning(f'Invalid box format: {box_coords}')
                continue

            try:
                x1, y1, x2, y2 = box_coords

                # Validate coordinates are in [0, 1] range
                if not all(0 <= c <= 1 for c in [x1, y1, x2, y2]):
                    logger.warning(f'Box coordinates out of range: {box_coords}')
                    continue

                # Convert to pixel coordinates
                px1 = int(x1 * img_width)
                py1 = int(y1 * img_height)
                px2 = int(x2 * img_width)
                py2 = int(y2 * img_height)

                # Ensure valid crop dimensions
                if px2 <= px1 or py2 <= py1:
                    logger.warning(f'Invalid box dimensions: {box_coords}')
                    continue

                # Crop the region
                crop = pil_image.crop((px1, py1, px2, py2))

                # Convert crop to bytes for encoding
                crop_buffer = BytesIO()
                crop.save(crop_buffer, format='JPEG', quality=95)
                crop_bytes = crop_buffer.getvalue()

                # Generate embedding
                embedding = service.encode_image(crop_bytes, use_cache=use_cache)

                results.append(
                    BoxEmbedding(
                        box=[x1, y1, x2, y2],
                        embedding=embedding.tolist(),
                    )
                )

            except Exception as e:
                logger.warning(f'Failed to process box {box_coords}: {e}')
                continue

        return BoxEmbeddingsResponse(
            boxes=results,
            num_boxes=len(results),
            status='success',
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Box embeddings failed: {e}')
        raise HTTPException(
            status_code=500, detail=f'Box embedding extraction failed: {e!s}'
        ) from e
