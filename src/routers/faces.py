"""
Face Detection and Recognition Router.

Provides face-related endpoints using YOLO11-face exclusively.
All endpoints use CPU preprocessing for stability at high concurrency.

Endpoints:
- POST /faces/detect - Face detection only (YOLO11-face)
- POST /faces/recognize - Detection + ArcFace 512-dim embeddings
- POST /faces/verify - 1:1 face comparison (two images, return similarity)
- POST /faces/search - Find similar faces in index (requires OpenSearch)
- POST /faces/identify - 1:N face identification
"""

import logging

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.core.dependencies import VisualSearchDep
from src.schemas.detection import ImageMetadata
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/faces',
    tags=['Face Recognition'],
    default_response_class=ORJSONResponse,
)


# =============================================================================
# Response Models
# =============================================================================


class FaceBox(BaseModel):
    """Face bounding box with landmarks."""

    box: list[float] = Field(..., description='Face bounding box [x1, y1, x2, y2] normalized [0,1]')
    landmarks: list[float] = Field(
        ..., description='5-point facial landmarks [lx1,ly1,...,lx5,ly5] normalized [0,1]'
    )
    score: float = Field(..., description='Detection confidence score')
    quality: float | None = Field(
        default=None, description='Face quality score (frontality, sharpness)'
    )


class FaceDetectResponse(BaseModel):
    """Response for face detection endpoint."""

    num_faces: int = Field(..., description='Number of faces detected')
    faces: list[FaceBox] = Field(default_factory=list, description='Detected faces')
    image: ImageMetadata | None = Field(default=None, description='Original image dimensions')
    total_time_ms: float | None = Field(default=None, description='Processing time in ms')


class FaceRecognizeResponse(BaseModel):
    """Response for face recognition endpoint (detection + embeddings)."""

    num_faces: int = Field(..., description='Number of faces detected')
    faces: list[FaceBox] = Field(default_factory=list, description='Detected faces')
    embeddings: list[list[float]] = Field(
        default_factory=list, description='512-dim ArcFace embeddings per face'
    )
    image: ImageMetadata | None = Field(default=None, description='Original image dimensions')
    total_time_ms: float | None = Field(default=None, description='Processing time in ms')


class FaceVerifyResponse(BaseModel):
    """Response for 1:1 face verification."""

    match: bool = Field(..., description='Whether faces match (above threshold)')
    similarity: float = Field(..., description='Cosine similarity score (0-1)')
    threshold: float = Field(..., description='Threshold used for matching')
    image1: dict = Field(..., description='Face info from first image')
    image2: dict = Field(..., description='Face info from second image')
    total_time_ms: float | None = Field(default=None, description='Processing time in ms')


class FaceSearchResult(BaseModel):
    """Single face search result."""

    face_id: str = Field(..., description='Face document ID')
    image_id: str = Field(..., description='Source image ID')
    image_path: str | None = Field(default=None, description='Source image path')
    score: float = Field(..., description='Similarity score')
    person_id: str | None = Field(default=None, description='Person ID if assigned')
    person_name: str | None = Field(default=None, description='Person name if known')
    box: list[float] = Field(..., description='Face box in source image')
    confidence: float = Field(..., description='Original detection confidence')


class FaceSearchResponse(BaseModel):
    """Response for face similarity search."""

    query_face: FaceBox = Field(..., description='Query face used for search')
    results: list[FaceSearchResult] = Field(default_factory=list, description='Search results')
    total_results: int = Field(..., description='Number of results returned')
    search_time_ms: float | None = Field(default=None, description='Search execution time')
    total_time_ms: float | None = Field(default=None, description='Total processing time')


class FaceIdentifyMatch(BaseModel):
    """Single match in face identification."""

    face_id: str = Field(..., description='Matched face ID')
    image_id: str = Field(..., description='Source image ID')
    image_path: str | None = Field(default=None, description='Source image path')
    score: float = Field(..., description='Similarity score')
    person_id: str | None = Field(default=None, description='Person ID if assigned')
    person_name: str | None = Field(default=None, description='Person name if known')


class FaceIdentifyResult(BaseModel):
    """Identification result for a single query face."""

    query_face: FaceBox = Field(..., description='Query face info')
    identified: bool = Field(..., description='Whether a match was found above threshold')
    best_match: FaceIdentifyMatch | None = Field(default=None, description='Best matching face')
    all_matches: list[FaceIdentifyMatch] = Field(
        default_factory=list, description='All matches above threshold'
    )


class FaceIdentifyResponse(BaseModel):
    """Response for 1:N face identification."""

    num_faces: int = Field(..., description='Number of faces in query image')
    results: list[FaceIdentifyResult] = Field(
        default_factory=list, description='Identification results per face'
    )
    total_time_ms: float | None = Field(default=None, description='Total processing time')


# =============================================================================
# Endpoints
# =============================================================================


@router.post('/detect', response_model=FaceDetectResponse)
def detect_faces(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
    confidence: float = Query(
        0.5,
        ge=0.1,
        le=0.99,
        description='Minimum face detection confidence threshold',
    ),
):
    """
    Detect faces in image using YOLO11-face.

    Pipeline:
    1. CPU image decode and preprocessing
    2. YOLO11-face detection via TensorRT
    3. GPU NMS (End2End TensorRT)
    4. Returns face boxes, 5-point landmarks, and confidence scores

    Args:
        image: Image file (JPEG/PNG)
        confidence: Minimum detection confidence threshold (0.1-0.99)

    Returns:
        Face detections with normalized [0,1] coordinates and landmarks.
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

        result = inference_service.detect_faces(image_bytes, confidence=confidence)

        if result.get('status') == 'error':
            raise HTTPException(500, result.get('error', 'Face detection failed'))

        orig_h, orig_w = result['orig_shape']

        faces = [
            FaceBox(
                box=f['box'],
                landmarks=f['landmarks'],
                score=f['score'],
                quality=f.get('quality'),
            )
            for f in result['faces']
        ]

        return FaceDetectResponse(
            num_faces=result['num_faces'],
            faces=faces,
            image=ImageMetadata(width=orig_w, height=orig_h),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Face detection failed: {e}')
        raise HTTPException(500, f'Face detection failed: {e!s}') from e


@router.post('/recognize', response_model=FaceRecognizeResponse)
def recognize_faces(
    image: UploadFile = File(..., description='Image file (JPEG/PNG)'),
    confidence: float = Query(
        0.5,
        ge=0.1,
        le=0.99,
        description='Minimum face detection confidence threshold',
    ),
):
    """
    Detect faces and extract ArcFace identity embeddings.

    Pipeline:
    1. CPU image decode and preprocessing
    2. YOLO11-face detection via TensorRT
    3. Face alignment from HD original (industry standard)
    4. ArcFace embedding extraction (512-dim L2-normalized)

    Args:
        image: Image file (JPEG/PNG)
        confidence: Minimum detection confidence threshold (0.1-0.99)

    Use embeddings for:
    - Face verification (1:1 matching) - cosine similarity > 0.6
    - Face identification (1:N search) - OpenSearch k-NN

    Returns:
        Face detections with normalized [0,1] coordinates and 512-dim embeddings per face.
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

        result = inference_service.detect_faces(image_bytes, confidence=confidence)

        if result.get('status') == 'error':
            raise HTTPException(500, result.get('error', 'Face recognition failed'))

        orig_h, orig_w = result['orig_shape']

        faces = [
            FaceBox(
                box=f['box'],
                landmarks=f['landmarks'],
                score=f['score'],
                quality=f.get('quality'),
            )
            for f in result['faces']
        ]

        return FaceRecognizeResponse(
            num_faces=result['num_faces'],
            faces=faces,
            embeddings=result['embeddings'],
            image=ImageMetadata(width=orig_w, height=orig_h),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Face recognition failed: {e}')
        raise HTTPException(500, f'Face recognition failed: {e!s}') from e


@router.post('/verify', response_model=FaceVerifyResponse)
def verify_faces(
    image1: UploadFile = File(..., description='First image with face'),
    image2: UploadFile = File(..., description='Second image with face'),
    threshold: float = Query(
        0.6,
        ge=0.0,
        le=1.0,
        description='Similarity threshold for match decision',
    ),
    confidence: float = Query(
        0.5,
        ge=0.1,
        le=0.99,
        description='Minimum face detection confidence',
    ),
):
    """
    Verify if two images contain the same person (1:1 verification).

    Extracts ArcFace embeddings from both images and compares using cosine similarity.

    Args:
        image1: First image with face
        image2: Second image with face
        threshold: Similarity threshold for match decision
        confidence: Minimum face detection confidence

    Threshold guidelines:
    - 0.6: High confidence (recommended for security)
    - 0.5: Balanced precision/recall
    - 0.4: More permissive (may have false positives)

    Returns:
        Match decision, similarity score, and face info from both images.
    """
    try:
        image1_bytes = image1.file.read()
        image2_bytes = image2.file.read()
        inference_service = InferenceService()

        result1 = inference_service.detect_faces(image1_bytes, confidence=confidence)
        result2 = inference_service.detect_faces(image2_bytes, confidence=confidence)

        if result1.get('status') == 'error':
            raise HTTPException(500, f'Face detection failed on image1: {result1.get("error")}')
        if result2.get('status') == 'error':
            raise HTTPException(500, f'Face detection failed on image2: {result2.get("error")}')

        if result1['num_faces'] == 0:
            raise HTTPException(400, 'No face detected in first image')
        if result2['num_faces'] == 0:
            raise HTTPException(400, 'No face detected in second image')

        # Use first (largest/most confident) face from each image
        emb1 = np.array(result1['embeddings'][0])
        emb2 = np.array(result2['embeddings'][0])

        # Cosine similarity (embeddings are L2-normalized)
        similarity = float(np.dot(emb1, emb2))
        is_match = similarity >= threshold

        return FaceVerifyResponse(
            match=is_match,
            similarity=round(similarity, 4),
            threshold=threshold,
            image1={
                'num_faces': result1['num_faces'],
                'face_used': result1['faces'][0],
            },
            image2={
                'num_faces': result2['num_faces'],
                'face_used': result2['faces'][0],
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Face verification failed: {e}')
        raise HTTPException(500, f'Face verification failed: {e!s}') from e


@router.post('/search', response_model=FaceSearchResponse)
async def search_faces(
    search_service: VisualSearchDep,
    image: UploadFile = File(..., description='Image file with face to search'),
    face_index: int = Query(
        0,
        ge=0,
        description='Which detected face to use as query (0-indexed)',
    ),
    top_k: int = Query(
        10,
        ge=1,
        le=100,
        description='Number of results to return',
    ),
    min_score: float = Query(
        0.7,
        ge=0.0,
        le=1.0,
        description='Minimum similarity score (0.7 recommended for identity)',
    ),
    confidence: float = Query(
        0.5,
        ge=0.1,
        le=0.99,
        description='Minimum face detection confidence',
    ),
):
    """
    Find similar faces in the indexed database.

    Pipeline:
    1. Detect faces in query image using YOLO11-face
    2. Extract ArcFace embedding for selected face
    3. Search visual_search_faces index via OpenSearch k-NN

    Args:
        image: Image file with face to search
        face_index: Which detected face to use as query (0-indexed)
        top_k: Number of results to return
        min_score: Minimum similarity score threshold
        confidence: Minimum face detection confidence

    Requires:
        OpenSearch with visual_search_faces index populated via /ingest.

    Returns:
        Query face info and list of similar faces from database.
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

        # Detect and recognize faces
        result = inference_service.detect_faces(image_bytes, confidence=confidence)

        if result.get('status') == 'error':
            raise HTTPException(500, result.get('error', 'Face detection failed'))

        if result['num_faces'] == 0:
            raise HTTPException(400, 'No faces detected in query image')

        if face_index >= result['num_faces']:
            raise HTTPException(
                400,
                f'Face index {face_index} out of range (0-{result["num_faces"] - 1})',
            )

        # Get query face info and embedding
        query_embedding = np.array(result['embeddings'][face_index])
        face_data = result['faces'][face_index]
        query_face = FaceBox(
            box=face_data['box'],
            landmarks=face_data['landmarks'],
            score=face_data['score'],
            quality=face_data.get('quality'),
        )

        # Search faces index via OpenSearch
        search_results = await search_service.opensearch.search_faces(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

        # Format results
        results = [
            FaceSearchResult(
                face_id=r.get('face_id', r.get('_id', '')),
                image_id=r.get('image_id', ''),
                image_path=r.get('image_path'),
                score=r.get('score', 0.0),
                person_id=r.get('person_id'),
                person_name=r.get('person_name'),
                box=r.get('box', [0, 0, 0, 0]),
                confidence=r.get('confidence', 0.0),
            )
            for r in search_results
        ]

        return FaceSearchResponse(
            query_face=query_face,
            results=results,
            total_results=len(results),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Face search failed: {e}')
        raise HTTPException(500, f'Face search failed: {e!s}') from e


@router.post('/identify', response_model=FaceIdentifyResponse)
async def identify_faces(
    search_service: VisualSearchDep,
    image: UploadFile = File(..., description='Image file with faces to identify'),
    top_k: int = Query(
        5,
        ge=1,
        le=100,
        description='Number of matches to return per face',
    ),
    threshold: float = Query(
        0.6,
        ge=0.0,
        le=1.0,
        description='Similarity threshold for positive identification',
    ),
    confidence: float = Query(
        0.5,
        ge=0.1,
        le=0.99,
        description='Minimum face detection confidence',
    ),
):
    """
    1:N face identification - identify all faces in image against database.

    For each detected face:
    1. Extract ArcFace embedding
    2. Search visual_search_faces index
    3. Return best match if above threshold

    Args:
        image: Image file with faces to identify
        top_k: Number of matches to return per face
        threshold: Similarity threshold for positive identification
        confidence: Minimum face detection confidence

    Use cases:
    - Access control: verify person identity
    - Photo organization: auto-tag known people
    - Security: identify persons of interest

    Returns:
        Identification results for each detected face.
    """
    try:
        image_bytes = image.file.read()
        inference_service = InferenceService()

        # Detect and recognize all faces
        result = inference_service.detect_faces(image_bytes, confidence=confidence)

        if result.get('status') == 'error':
            raise HTTPException(500, result.get('error', 'Face detection failed'))

        if result['num_faces'] == 0:
            return FaceIdentifyResponse(
                num_faces=0,
                results=[],
            )

        # Identify each face
        identification_results = []

        for i in range(result['num_faces']):
            query_embedding = np.array(result['embeddings'][i])
            face_data = result['faces'][i]

            query_face = FaceBox(
                box=face_data['box'],
                landmarks=face_data['landmarks'],
                score=face_data['score'],
                quality=face_data.get('quality'),
            )

            # Search for matches
            search_results = await search_service.opensearch.search_faces(
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=threshold,
            )

            # Format matches
            all_matches = [
                FaceIdentifyMatch(
                    face_id=r.get('face_id', r.get('_id', '')),
                    image_id=r.get('image_id', ''),
                    image_path=r.get('image_path'),
                    score=r.get('score', 0.0),
                    person_id=r.get('person_id'),
                    person_name=r.get('person_name'),
                )
                for r in search_results
            ]

            # Determine if identified (best match above threshold)
            identified = len(all_matches) > 0
            best_match = all_matches[0] if identified else None

            identification_results.append(
                FaceIdentifyResult(
                    query_face=query_face,
                    identified=identified,
                    best_match=best_match,
                    all_matches=all_matches,
                )
            )

        return FaceIdentifyResponse(
            num_faces=result['num_faces'],
            results=identification_results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Face identification failed: {e}')
        raise HTTPException(500, f'Face identification failed: {e!s}') from e
