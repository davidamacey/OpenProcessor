"""
Fast Face Detection + Recognition Client

Calls SCRFD and ArcFace models directly via gRPC for low-latency face
detection and embedding. Uses CPU preprocessing with Triton GPU inference.

Pipeline:
    FastAPI -> CPU: image decode + resize
           -> gRPC -> SCRFD TensorRT (face detection + 5-point landmarks)
           -> CPU: anchor decode + NMS + Umeyama affine alignment
           -> gRPC -> ArcFace TensorRT (512-dim face embedding)

Industry-standard pipeline matching InsightFace/Apple/Google.
"""

import logging
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from tritonclient.grpc import InferInput, InferRequestedOutput

from src.clients.triton_pool import TritonClientManager
from src.utils.face_align import align_faces_batch, preprocess_for_arcface
from src.utils.retry import retry_sync
from src.utils.scrfd_decode import (
    ALL_OUTPUT_NAMES,
    INPUT_SIZE,
    decode_scrfd_outputs,
    preprocess_scrfd,
)


logger = logging.getLogger(__name__)

# Constants
ARCFACE_SIZE = 112
MAX_FACES = 128


class FastFaceClient:
    """
    High-performance face detection and recognition client.

    Uses SCRFD for face detection with 5-point landmarks, enabling proper
    Umeyama affine alignment before ArcFace embedding extraction.
    """

    def __init__(self, triton_url: str = 'triton-server:8001'):
        self.client = TritonClientManager.get_sync_client(triton_url)
        self.scrfd_model = 'scrfd_10g_bnkps'
        self.arcface_model = 'arcface_w600k_r50'
        logger.info('FastFaceClient initialized (SCRFD + Umeyama alignment)')

    # =========================================================================
    # Triton gRPC calls
    # =========================================================================

    def _call_scrfd(self, blob: np.ndarray) -> dict[str, np.ndarray]:
        """Call SCRFD model via gRPC and return raw output tensors."""
        inputs = [InferInput('input.1', list(blob.shape), 'FP32')]
        inputs[0].set_data_from_numpy(blob)

        outputs = [InferRequestedOutput(name) for name in ALL_OUTPUT_NAMES]

        response = retry_sync(
            self.client.infer,
            model_name=self.scrfd_model,
            inputs=inputs,
            outputs=outputs,
        )

        return {name: response.as_numpy(name) for name in ALL_OUTPUT_NAMES}

    def _call_arcface(self, faces: np.ndarray) -> np.ndarray:
        """Call ArcFace model directly with batched faces."""
        if len(faces) == 0:
            return np.array([])

        inputs = [InferInput('input', list(faces.shape), 'FP32')]
        inputs[0].set_data_from_numpy(faces)

        outputs = [InferRequestedOutput('output')]

        response = retry_sync(
            self.client.infer,
            model_name=self.arcface_model,
            inputs=inputs,
            outputs=outputs,
        )

        embeddings = response.as_numpy('output')

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return embeddings / norms

    # =========================================================================
    # Quality scoring
    # =========================================================================

    def _compute_quality(self, boxes: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        """Compute face quality scores (vectorized)."""
        if len(boxes) == 0:
            return np.array([])

        face_w = boxes[:, 2] - boxes[:, 0]
        face_h = boxes[:, 3] - boxes[:, 1]
        face_area = face_w * face_h
        image_area = orig_h * orig_w
        size_score = np.clip((face_area / image_area) * 10, 0, 1)

        margin = 0.02
        boundary_score = np.ones(len(boxes))
        at_edge = (
            (boxes[:, 0] < orig_w * margin)
            | (boxes[:, 2] > orig_w * (1 - margin))
            | (boxes[:, 1] < orig_h * margin)
            | (boxes[:, 3] > orig_h * (1 - margin))
        )
        boundary_score[at_edge] *= 0.8

        aspect_ratio = np.minimum(face_w, face_h) / (np.maximum(face_w, face_h) + 1e-6)

        quality = np.clip(size_score * boundary_score * aspect_ratio, 0, 1)
        return quality.astype(np.float32)

    # =========================================================================
    # Main pipeline
    # =========================================================================

    def recognize(self, image_bytes: bytes, confidence: float = 0.5) -> dict[str, Any]:
        """
        Detect faces and extract ArcFace embeddings.

        Pipeline: SCRFD detect -> Umeyama align -> ArcFace embed.

        Args:
            image_bytes: JPEG/PNG image bytes
            confidence: Minimum detection confidence

        Returns:
            Dict with num_faces, face_boxes, face_scores, face_embeddings,
            face_landmarks, face_quality, orig_shape
        """
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {'status': 'error', 'error': 'Failed to decode image'}

        orig_h, orig_w = img.shape[:2]

        # Cap image size for faster processing
        MAX_DIM = 1024
        if max(orig_h, orig_w) > MAX_DIM:
            cap_scale = MAX_DIM / max(orig_h, orig_w)
            img = cv2.resize(img, (int(orig_w * cap_scale), int(orig_h * cap_scale)))
            orig_h, orig_w = img.shape[:2]

        # Preprocess for SCRFD
        blob, det_scale = preprocess_scrfd(img, INPUT_SIZE)

        # Call SCRFD via Triton gRPC
        raw_outputs = self._call_scrfd(blob)

        # Decode: anchor decode + NMS on CPU
        boxes, scores, landmarks = decode_scrfd_outputs(
            raw_outputs,
            det_scale,
            det_thresh=confidence,
            nms_thresh=0.4,
            max_faces=MAX_FACES,
        )

        num_faces = len(boxes)
        if num_faces == 0:
            return {
                'status': 'success',
                'num_faces': 0,
                'face_boxes': [],
                'face_scores': [],
                'face_embeddings': [],
                'face_landmarks': [],
                'face_quality': [],
                'orig_shape': (orig_h, orig_w),
            }

        # Align faces using Umeyama similarity transform
        aligned_faces = align_faces_batch(img, landmarks, ARCFACE_SIZE)

        # Preprocess for ArcFace: BGR->RGB, CHW, normalize
        face_batch = preprocess_for_arcface(aligned_faces)

        # Call ArcFace via Triton gRPC
        embeddings = self._call_arcface(face_batch)

        # Compute quality scores
        quality = self._compute_quality(boxes, orig_h, orig_w)

        # Normalize boxes to [0, 1]
        boxes_norm = boxes.copy()
        boxes_norm[:, [0, 2]] /= orig_w
        boxes_norm[:, [1, 3]] /= orig_h

        # Normalize landmarks to [0, 1] and flatten to [N, 10]
        lmk_norm = landmarks.copy()  # [N, 5, 2]
        lmk_norm[:, :, 0] /= orig_w
        lmk_norm[:, :, 1] /= orig_h
        lmk_flat = lmk_norm.reshape(-1, 10)  # [N, 10] flat: [x1,y1,...,x5,y5]

        return {
            'status': 'success',
            'num_faces': num_faces,
            'face_boxes': boxes_norm.tolist(),
            'face_scores': scores.tolist(),
            'face_embeddings': embeddings.tolist(),
            'face_landmarks': lmk_flat.tolist(),
            'face_quality': quality.tolist(),
            'orig_shape': (orig_h, orig_w),
        }


@lru_cache(maxsize=4)
def get_fast_face_client(triton_url: str = 'triton-server:8001') -> FastFaceClient:
    """Get a cached FastFaceClient instance."""
    return FastFaceClient(triton_url=triton_url)
