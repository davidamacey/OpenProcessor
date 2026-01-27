"""
Inference service for YOLO detection, face recognition, and embeddings.

All methods use CPU preprocessing with direct TRT model calls via Triton gRPC.
"""

import hashlib
import logging
from functools import lru_cache
from typing import Any

import numpy as np

from src.clients.fast_face_client import get_fast_face_client
from src.clients.triton_client import get_triton_client
from src.config import get_settings
from src.utils.cache import get_clip_tokenizer, get_image_cache, get_text_cache
from src.utils.image_processing import decode_image, validate_image


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


class InferenceService:
    """
    Simplified inference service for object detection, face recognition, and embeddings.

    Public methods:
    - detect(): Object detection using YOLOv11 TRT End2End
    - detect_batch(): Batch object detection
    - detect_faces(): SCRFD face detection with ArcFace embeddings
    - recognize_faces(): Full pipeline (YOLO + faces + CLIP embedding)
    - encode_image(): MobileCLIP image encoding
    - encode_text(): MobileCLIP text encoding
    - analyze_full(): Combined YOLO + faces + CLIP + OCR

    All responses include:
    - detections: List of normalized [0,1] bounding boxes
    - num_detections: Count of detections
    - image: Original image dimensions
    - model: Model name and backend info
    - total_time_ms: Injected by middleware
    """

    def __init__(self):
        """Initialize inference service."""
        self.settings = get_settings()

    # =========================================================================
    # Object Detection
    # =========================================================================

    def detect(
        self,
        image_bytes: bytes,
        model_name: str = 'yolov11_small_trt_end2end',
    ) -> dict[str, Any]:
        """
        Detect objects in image using YOLOv11 TensorRT End2End model.

        Uses CPU preprocessing with letterbox transform, then TensorRT inference
        with GPU NMS. Coordinates are normalized to [0, 1] range.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
            model_name: Triton model name (default: yolov11_small_trt_end2end)

        Returns:
            Standardized response dict with detections
        """
        # Decode and validate image
        img = decode_image(image_bytes, 'image')
        validate_image(img, 'image')
        image_shape = img.shape[:2]  # (height, width)

        # Run inference via Triton
        client = get_triton_client(self.settings.triton_url)
        result = client.infer_yolo_end2end(img, model_name)
        detections = client.format_detections(result)

        return build_response(
            detections=detections,
            image_shape=image_shape,
            model_name=model_name,
            backend='triton',
        )

    def detect_batch(
        self,
        images: list[bytes],
        model_name: str = 'yolov11_small_trt_end2end',
    ) -> list[dict[str, Any]]:
        """
        Detect objects in multiple images using batched inference.

        Args:
            images: List of raw image bytes (JPEG/PNG)
            model_name: Triton model name (default: yolov11_small_trt_end2end)

        Returns:
            List of standardized response dicts, one per image
        """
        # Decode and validate all images
        decoded_images = []
        image_shapes = []
        failed_indices = []

        for idx, image_bytes in enumerate(images):
            try:
                img = decode_image(image_bytes, f'image_{idx}')
                validate_image(img, f'image_{idx}')
                decoded_images.append(img)
                image_shapes.append(img.shape[:2])
            except ValueError as e:
                logger.warning(f'Failed to decode image {idx}: {e}')
                failed_indices.append(idx)

        if not decoded_images:
            return []

        # Run batch inference via Triton
        client = get_triton_client(self.settings.triton_url)
        results = client.infer_yolo_end2end_batch(decoded_images, model_name)

        # Format responses
        responses = []
        for result, shape in zip(results, image_shapes, strict=False):
            detections = client.format_detections(result)
            responses.append(
                build_response(
                    detections=detections,
                    image_shape=shape,
                    model_name=model_name,
                    backend='triton',
                )
            )

        return responses

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
                - status: 'success' or 'error'
                - num_faces: Number of faces detected
                - faces: List of face dicts with box, landmarks, score, quality
                - embeddings: List of 512-dim L2-normalized ArcFace embeddings
                - orig_shape: (height, width)
        """
        face_client = get_fast_face_client(self.settings.triton_url)

        try:
            result = face_client.recognize(image_bytes, confidence=confidence)
        except Exception as e:
            logger.error(f'Face detection failed: {e}')
            return {
                'status': 'error',
                'error': str(e),
                'num_faces': 0,
                'faces': [],
                'embeddings': [],
                'orig_shape': (0, 0),
            }

        if result.get('status') == 'error':
            return {
                'status': 'error',
                'error': result.get('error', 'Unknown error'),
                'num_faces': 0,
                'faces': [],
                'embeddings': [],
                'orig_shape': (0, 0),
            }

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
                - status: 'success' or 'error'
                - detections: YOLO object detections
                - num_detections: Number of objects detected
                - num_faces: Number of faces detected
                - faces: List of face dicts with box, landmarks, score, quality
                - face_embeddings: List of 512-dim ArcFace embeddings
                - image_embedding: 512-dim MobileCLIP embedding
                - embedding_norm: L2 norm of image embedding
                - orig_shape: (height, width)
        """
        from concurrent.futures import ThreadPoolExecutor

        client = get_triton_client(self.settings.triton_url)
        face_client = get_fast_face_client(self.settings.triton_url)

        try:
            # Run YOLO+CLIP and face detection in parallel
            def run_yolo_clip():
                return client.infer_yolo_clip_cpu(image_bytes)

            def run_face_detection():
                return face_client.recognize(image_bytes, confidence=confidence)

            with ThreadPoolExecutor(max_workers=2) as executor:
                yolo_future = executor.submit(run_yolo_clip)
                face_future = executor.submit(run_face_detection)
                yolo_result = yolo_future.result()
                face_result = face_future.result()
        except Exception as e:
            logger.error(f'Face recognition failed: {e}')
            return {
                'status': 'error',
                'error': str(e),
                'detections': [],
                'num_detections': 0,
                'num_faces': 0,
                'faces': [],
                'face_embeddings': [],
                'image_embedding': None,
                'embedding_norm': 0.0,
                'orig_shape': (0, 0),
            }

        # Format YOLO detections
        detections = client.format_detections(yolo_result)

        orig_h, orig_w = face_result.get('orig_shape', (0, 0))

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
        client = get_triton_client(self.settings.triton_url)
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

        # Tokenize using cached singleton
        tokenizer = get_clip_tokenizer()
        tokens = tokenizer(
            text,
            padding='max_length',
            max_length=77,
            truncation=True,
            return_tensors='np',
        )

        # Get Triton client and encode
        client = get_triton_client(self.settings.triton_url)
        embedding = client.encode_text(tokens['input_ids'])

        # Cache the result
        if use_cache:
            cache.set(cache_key, embedding)

        return embedding

    # =========================================================================
    # Full Analysis
    # =========================================================================

    def analyze_full(
        self,
        image_bytes: bytes,
        include_ocr: bool = True,
    ) -> dict[str, Any]:
        """
        Combined analysis: YOLO detection + faces + CLIP embedding + OCR.

        Single request that returns all analysis results:
        1. YOLO object detection + MobileCLIP embeddings
        2. SCRFD face detection + Umeyama alignment + ArcFace embeddings
        3. PP-OCRv5 text detection and recognition (optional)

        Args:
            image_bytes: Raw JPEG/PNG bytes
            include_ocr: Include OCR text extraction (default True)

        Returns:
            Dict with all analysis results:
            - Detection: num_detections, detections
            - Embeddings: image_embedding, box_embeddings
            - Faces: num_faces, faces, face_embeddings
            - OCR (if enabled): num_texts, texts, text_boxes
            - Metadata: orig_shape
        """
        from concurrent.futures import ThreadPoolExecutor

        client = get_triton_client(self.settings.triton_url)
        face_client = get_fast_face_client(self.settings.triton_url)

        try:
            # Run YOLO+CLIP and SCRFD face detection in parallel
            def run_yolo_clip():
                return client.infer_yolo_clip_cpu(image_bytes)

            def run_face_detection():
                return face_client.recognize(image_bytes, confidence=0.5)

            with ThreadPoolExecutor(max_workers=2) as executor:
                yolo_future = executor.submit(run_yolo_clip)
                face_future = executor.submit(run_face_detection)
                yolo_result = yolo_future.result()
                face_result = face_future.result()
        except Exception as e:
            logger.error(f'Full analysis failed: {e}')
            return {
                'status': 'error',
                'error': str(e),
            }

        # Format YOLO detections
        detections = client.format_detections(yolo_result)

        orig_h, orig_w = face_result.get('orig_shape', (0, 0))

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

        result = {
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

        # Run OCR if requested
        if include_ocr:
            try:
                ocr_result = client.infer_ocr(image_bytes)
                result['num_texts'] = ocr_result['num_texts']
                result['texts'] = ocr_result['texts']
                result['text_boxes'] = (
                    ocr_result['text_boxes'].tolist() if len(ocr_result['text_boxes']) > 0 else []
                )
                result['text_boxes_normalized'] = (
                    ocr_result['text_boxes_normalized'].tolist()
                    if len(ocr_result['text_boxes_normalized']) > 0
                    else []
                )
                result['text_scores'] = (
                    ocr_result['text_scores'].tolist() if len(ocr_result['text_scores']) > 0 else []
                )
            except Exception as e:
                logger.warning(f'OCR failed: {e}')
                result['num_texts'] = 0
                result['texts'] = []
                result['text_boxes'] = []
                result['text_boxes_normalized'] = []
                result['text_scores'] = []
                result['ocr_error'] = str(e)

        return result


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Get singleton inference service instance (cached)."""
    return InferenceService()
