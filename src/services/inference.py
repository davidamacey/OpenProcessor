"""
Inference service for YOLO detection, face recognition, and embeddings.

All methods use CPU preprocessing with direct TRT model calls via Triton gRPC.
"""

import hashlib
import logging
from functools import lru_cache
from typing import Any

import numpy as np

from src.config import get_settings
from src.config.settings import TritonModelConfig
from src.services.sync_triton import YOLO_CLIP, routed_face_client, routed_triton_client
from src.utils.cache import clip_text_tokens, get_image_cache, get_text_cache
from src.utils.face_response import face_landmark_mismatch
from src.utils.image_processing import decode_image, validate_image
from src.utils.retry import TRITON_CALL_ERRORS, RetryExhaustedError
from src.utils.upstream_errors import UpstreamError


logger = logging.getLogger(__name__)


def build_response(
    detections: list,
    image_shape: tuple[int, int],
    model_name: str,
    backend: str = 'triton',
    embedding: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Build standardized inference response.

    Args:
        detections: List of detection dicts
        image_shape: Original image (height, width)
        model_name: Model name used
        backend: Inference backend (triton)
        embedding: Optional image embedding

    Returns:
        Standardized response dict
    """
    response = {
        'detections': detections,
        'num_detections': len(detections),
        'status': 'success',
        'image': {
            'height': image_shape[0],
            'width': image_shape[1],
        },
        'model': {
            'name': model_name,
            'backend': backend,
        },
        # total_time_ms injected by middleware
    }

    if embedding is not None:
        response['embedding_norm'] = float(np.linalg.norm(embedding))

    return response


def _refuse_unusable_face_result(result: dict[str, Any]) -> None:
    """Raise for a face response that cannot be turned into faces: an undecodable image is the
    caller's fault (400); a response whose landmarks do not pair with its faces is the model's
    (503 with the stable reason). Neither may look like an image with no faces."""
    if result.get('status') == 'error':
        # src.core imports the services package, so this cannot be a module-level import.
        from src.core.exceptions import RequestRejectedError

        raise RequestRejectedError(400, result.get('error') or 'Failed to decode image')
    if reason := face_landmark_mismatch(result):
        raise UpstreamError('triton', reason)


class InferenceService:
    """
    Simplified inference service for object detection, face recognition, and embeddings.

    Public methods:
    - detect(): Object detection using YOLOv11 fused TRT detector
    - detect_faces(): SCRFD face detection with ArcFace embeddings
    - recognize_faces(): Full pipeline (YOLO + faces + CLIP embedding)
    - encode_image(): MobileCLIP image encoding
    - encode_text(): MobileCLIP text encoding

    All responses include:
    - detections: List of normalized [0,1] bounding boxes
    - num_detections: Count of detections
    - image: Original image dimensions
    - model: Model name and backend info
    - total_time_ms: Injected by middleware
    """

    def __init__(self) -> None:
        """Initialize inference service."""
        self.settings = get_settings()

    # =========================================================================
    # Object Detection
    # =========================================================================

    def detect(
        self,
        image_bytes: bytes,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Detect objects in image using YOLOv11 fused TensorRT detector.

        Uses CPU preprocessing with letterbox transform, then TensorRT inference
        with GPU NMS. Coordinates are normalized to [0, 1] range.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            model_name: Triton model name (default: settings YOLO_MODEL)

        Returns:
            Standardized response dict with detections
        """
        model_name = model_name or TritonModelConfig.YOLO_MODEL

        # Decode and validate image
        img = decode_image(image_bytes, 'image')
        validate_image(img, 'image')
        image_shape = img.shape[:2]  # (height, width)

        # Run inference via Triton
        client = routed_triton_client(model_name)
        result = client.infer_fused_detector(img, model_name)
        detections = client.format_detections(result, model_name=model_name)

        return build_response(
            detections=detections,
            image_shape=image_shape,
            model_name=model_name,
            backend='triton',
        )

    # =========================================================================
    # Face Detection and Recognition
    # =========================================================================

    def detect_faces(
        self,
        image_bytes: bytes,
        confidence: float = 0.5,
    ) -> dict[str, Any]:
        """
        Detect faces using SCRFD + Umeyama alignment + ArcFace embeddings.

        Pipeline:
        1. Decode and preprocess (CPU)
        2. SCRFD face detection with 5-point landmarks via TensorRT
        3. Umeyama affine alignment from HD original (industry standard)
        4. ArcFace embedding extraction via TensorRT (512-dim L2-normalized)

        Uses SCRFD-10G for face detection with native 5-point landmarks.
        Face coordinates are normalized to [0, 1] range relative to original image.

        Args:
            image_bytes: Raw JPEG/PNG bytes
            confidence: Minimum detection confidence (default 0.5)

        Returns:
            Dict with:
                - status: 'success'
                - num_faces: Number of faces detected
                - faces: List of face dicts with box, landmarks, score, quality
                - embeddings: List of 512-dim L2-normalized ArcFace embeddings
                - orig_shape: (height, width)

        Raises:
            RequestRejectedError: 400, the bytes do not decode as an image.
            UpstreamError: the face models failed or answered with unusable landmarks.
            RetryExhaustedError: Triton is unreachable (the app answers 503/504).
        """
        face_client = routed_face_client()

        try:
            result = face_client.recognize(image_bytes, confidence=confidence)
        except RetryExhaustedError:
            raise
        except TRITON_CALL_ERRORS as e:
            logger.error(f'Face detection failed: {e}')
            raise UpstreamError('triton', 'face_inference_failed') from e

        _refuse_unusable_face_result(result)

        # FastFaceClient returns already-normalized boxes, landmarks, embeddings
        faces = []
        for i in range(result['num_faces']):
            box = result['face_boxes'][i]
            landmarks = result['face_landmarks'][i] if result['face_landmarks'] else []
            score = float(result['face_scores'][i])
            quality = float(result['face_quality'][i]) if result['face_quality'] else None

            faces.append(
                {
                    'box': list(box),
                    'landmarks': list(landmarks) if landmarks else [0.0] * 10,
                    'score': score,
                    'quality': quality,
                }
            )

        embeddings = result.get('face_embeddings', [])
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        return {
            'status': 'success',
            'num_faces': result['num_faces'],
            'faces': faces,
            'embeddings': embeddings,
            'orig_shape': result['orig_shape'],
        }

    def recognize_faces(
        self,
        image_bytes: bytes,
        confidence: float = 0.5,
    ) -> dict[str, Any]:
        """
        Full pipeline: YOLO object detection + SCRFD face detection + MobileCLIP embedding.

        Runs in parallel:
        1. YOLO object detection + MobileCLIP global embedding
        2. SCRFD face detection + Umeyama alignment + ArcFace embeddings

        Args:
            image_bytes: Raw JPEG/PNG bytes
            confidence: Minimum face detection confidence (default 0.5)

        Returns:
            Dict with:
                - status: 'success'
                - detections: YOLO object detections
                - num_detections: Number of objects detected
                - num_faces: Number of faces detected
                - faces: List of face dicts with box, landmarks, score, quality
                - face_embeddings: List of 512-dim ArcFace embeddings
                - image_embedding: 512-dim MobileCLIP embedding
                - embedding_norm: L2 norm of image embedding
                - orig_shape: (height, width)

        Raises:
            RequestRejectedError: 400, the bytes do not decode as an image.
            UpstreamError: the models failed or the face response is unusable.
            RetryExhaustedError: Triton is unreachable (the app answers 503/504).
        """
        from concurrent.futures import ThreadPoolExecutor

        client = routed_triton_client(*YOLO_CLIP)
        face_client = routed_face_client()

        try:
            # Run YOLO+CLIP and face detection in parallel
            def run_yolo_clip() -> dict[str, Any]:
                return client.infer_yolo_clip_cpu(image_bytes)

            def run_face_detection() -> dict[str, Any]:
                return face_client.recognize(image_bytes, confidence=confidence)

            with ThreadPoolExecutor(max_workers=2) as executor:
                yolo_future = executor.submit(run_yolo_clip)
                face_future = executor.submit(run_face_detection)
                yolo_result = yolo_future.result()
                face_result = face_future.result()
        except RetryExhaustedError:
            raise
        except TRITON_CALL_ERRORS as e:
            logger.error(f'Face recognition failed: {e}')
            raise UpstreamError('triton', 'face_inference_failed') from e

        # Format YOLO detections
        detections = client.format_detections(yolo_result, model_name=TritonModelConfig.YOLO_MODEL)

        orig_h, orig_w = face_result.get('orig_shape', (0, 0))

        _refuse_unusable_face_result(face_result)

        # Format face detections (FastFaceClient returns already-normalized data)
        faces = []
        for i in range(face_result.get('num_faces', 0)):
            box = face_result['face_boxes'][i]
            landmarks = face_result['face_landmarks'][i] if face_result['face_landmarks'] else []
            score = float(face_result['face_scores'][i])
            quality = float(face_result['face_quality'][i]) if face_result['face_quality'] else None

            faces.append(
                {
                    'box': list(box),
                    'landmarks': list(landmarks) if landmarks else [0.0] * 10,
                    'score': score,
                    'quality': quality,
                }
            )

        face_embeddings = face_result.get('face_embeddings', [])
        if isinstance(face_embeddings, np.ndarray):
            face_embeddings = face_embeddings.tolist()

        return {
            'status': 'success',
            # YOLO detections
            'detections': detections,
            'num_detections': len(detections),
            # Face detections and embeddings
            'num_faces': face_result.get('num_faces', 0),
            'faces': faces,
            'face_embeddings': face_embeddings,
            # Global embedding
            'image_embedding': yolo_result.get('image_embedding'),
            'embedding_norm': float(np.linalg.norm(yolo_result.get('image_embedding', [0]))),
            # Image metadata
            'orig_shape': (orig_h, orig_w),
        }

    # =========================================================================
    # Embedding Encoding
    # =========================================================================

    def encode_image(
        self,
        image_bytes: bytes,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Encode image to 512-dim embedding using MobileCLIP.

        Args:
            image_bytes: Raw JPEG/PNG bytes
            use_cache: Use embedding cache (default True)

        Returns:
            512-dim L2-normalized embedding
        """
        # Check cache first
        if use_cache:
            cache = get_image_cache()
            cache_key = hashlib.sha256(image_bytes).hexdigest()
            cached_embedding = cache.get(cache_key)
            if cached_embedding is not None:
                return cached_embedding

        # Get Triton client and encode
        client = routed_triton_client(TritonModelConfig.CLIP_IMAGE_MODEL)
        embedding = client.encode_image(image_bytes)

        # Cache the result
        if use_cache:
            cache.set(cache_key, embedding)

        return embedding

    def encode_text(
        self,
        text: str,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Encode text to 512-dim embedding using MobileCLIP text encoder.

        Args:
            text: Query text
            use_cache: Use embedding cache (default True)

        Returns:
            512-dim L2-normalized embedding
        """
        # Check cache first
        if use_cache:
            cache = get_text_cache()
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            cached_embedding = cache.get(cache_key)
            if cached_embedding is not None:
                return cached_embedding

        client = routed_triton_client(TritonModelConfig.CLIP_TEXT_MODEL)
        embedding = client.encode_text(clip_text_tokens(text))

        # Cache the result
        if use_cache:
            cache.set(cache_key, embedding)

        return embedding


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Get singleton inference service instance (cached)."""
    return InferenceService()
