"""
Unified Triton Client for inference pipelines.

Single client class that handles all Triton inference with proper connection pooling
and shared utilities. Sync-only for backpressure safety (async causes server deadlock
at 256+ clients).

Architecture:
- Shared gRPC connection pool via TritonClientManager
- Shared affine matrix calculation and JPEG parsing from utils/affine.py
- Shared detection formatting from utils/affine.py
- Clear method names: infer_yolo_end2end(), infer_yolo_clip_cpu()

Performance:
- Sync client with thread pool: Proper backpressure, 200+ RPS
- Shared connection enables Triton dynamic batching (5-10x throughput)

Active models:
- yolov11_small_trt_end2end: Object detection (YOLO11 End2End TRT)
- scrfd_10g_bnkps: Face detection with 5-point landmarks (SCRFD)
- arcface_w600k_r50: Face embedding extraction (ArcFace)
- mobileclip2_s2_image_encoder: Image embedding (MobileCLIP)
- mobileclip2_s2_text_encoder: Text embedding (MobileCLIP)
- paddleocr_det_trt + paddleocr_rec_trt: OCR (via ocr_pipeline BLS)
"""

import io
import logging
import time
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image
from tritonclient.grpc import InferInput, InferRequestedOutput
from ultralytics.data.augment import LetterBox

from src.clients.model_adapters import DetectionAdapter, resolve_adapter
from src.clients.triton_client_manager import TritonClientManager
from src.clients.triton_ocr import infer_ocr
from src.clients.triton_pool_core import uint8_images_input
from src.clients.triton_preprocess import (
    preprocess_clip_cpu,
    preprocess_for_mobileclip,
    preprocess_yolo_cpu,
)
from src.config.settings import TritonModelConfig
from src.utils.affine import format_detections_from_triton, inverse_letterbox_coords
from src.utils.retry import retry_sync


logger = logging.getLogger(__name__)

_MODEL_INFO_TTL_S = 30.0


class TritonClient:
    """
    Unified Triton client for inference pipelines.

    CRITICAL: Uses sync gRPC only (not async) to prevent server deadlock
    at high concurrency (256+ clients). FastAPI handles async via thread pool.

    Pipelines:
    - infer_yolo_end2end: End2End TRT + GPU NMS (CPU preprocessing)
    - infer_yolo_clip_cpu: YOLO + MobileCLIP (CPU preprocessing)
    - infer_ocr: PP-OCRv5 text detection + recognition (BLS pipeline)
    """

    def __init__(
        self,
        triton_url: str = 'triton-server:8001',
        max_retries: int = 3,
        retry_base_delay: float = 0.1,
        retry_max_delay: float = 5.0,
    ):
        self.triton_url = triton_url
        self.client = TritonClientManager.get_sync_client(triton_url)
        self.input_size = 640  # YOLO input size
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        # Per-model serving facts (output adapter + input size) from Triton
        # metadata, re-read after ``_MODEL_INFO_TTL_S``: a model re-promoted or
        # replaced under the same name must not keep its old size, and the API
        # runs many worker processes, so an in-process invalidation could not
        # reach them all.
        self._model_info: dict[str, tuple[float, DetectionAdapter, int]] = {}
        logger.info(f'Unified Triton client initialized (sync, retries={max_retries})')

    def _infer_with_retry(self, model_name: str, inputs: list, outputs: list):
        """Execute Triton inference with automatic retry on transient failures."""
        return retry_sync(
            self.client.infer,
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
        )

    def _read_model_info(self, model_name: str) -> tuple[DetectionAdapter, int]:
        """Output adapter (YOLO11 end2end 4-tensor or YOLO26 fused) and the
        square input size ``model_name`` was exported at (``images`` is
        ``[N, 3, H, W]``; 640 when dynamic or absent), cached for
        ``_MODEL_INFO_TTL_S``. A model promoted at another size would otherwise
        fail with a shape error."""
        now = time.monotonic()
        cached = self._model_info.get(model_name)
        if cached is not None and cached[0] > now:
            return cached[1], cached[2]
        metadata = self.client.get_model_metadata(model_name)
        adapter = resolve_adapter(metadata)
        size = self.input_size
        for tensor in metadata.inputs:
            dims = [int(d) for d in tensor.shape]
            if tensor.name == 'images' and len(dims) == 4 and dims[2] > 0:
                size = dims[2]
        self._model_info[model_name] = (now + _MODEL_INFO_TTL_S, adapter, size)
        return adapter, size

    def _get_detection_adapter(self, model_name: str) -> DetectionAdapter:
        return self._read_model_info(model_name)[0]

    def _model_input_size(self, model_name: str) -> int:
        return self._read_model_info(model_name)[1]

    # =========================================================================
    # YOLO End2End: CPU Preprocessing + TensorRT + GPU NMS
    # =========================================================================

    def infer_yolo_end2end(self, image_array: np.ndarray, model_name: str) -> dict[str, Any]:
        """
        YOLO End2End inference: CPU preprocessing + TensorRT with GPU NMS.

        Args:
            image_array: Preprocessed image (HWC, BGR, 0-255) from cv2.imdecode
            model_name: Triton model name (e.g., "yolov11_small_trt_end2end")

        Returns:
            Dict with num_dets, boxes, scores, classes, orig_shape, scale, padding
        """
        return self.infer_yolo_end2end_batch([image_array], model_name)[0]

    def infer_yolo_end2end_batch(self, images: list, model_name: str) -> list:
        """
        Batch YOLO End2End inference: CPU preprocessing + TensorRT with GPU NMS.

        Args:
            images: List of images (HWC, BGR, 0-255) from cv2.imdecode
            model_name: Triton model name (e.g., "yolov11_small_trt_end2end")

        Returns:
            List of dicts with num_dets, boxes, scores, classes, orig_shape, scale, padding
        """
        size = self._model_input_size(model_name)
        letterbox = LetterBox(new_shape=(size, size), auto=False, scaleup=False)

        orig_shapes = []
        scales = []
        paddings = []
        preprocessed = []

        for img in images:
            orig_h, orig_w = img.shape[:2]
            orig_shapes.append((orig_h, orig_w))

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_letterbox = cast('np.ndarray', letterbox(image=img_rgb))

            scale = min(size / orig_h, size / orig_w)
            scale = min(scale, 1.0)
            scales.append(scale)

            new_unpad_w = round(orig_w * scale)
            new_unpad_h = round(orig_h * scale)
            pad_w = (size - new_unpad_w) / 2.0
            pad_h = (size - new_unpad_h) / 2.0
            paddings.append((pad_w, pad_h))

            preprocessed.append(np.transpose(img_letterbox, (2, 0, 1)))

        input_data = np.stack(preprocessed, axis=0)
        batch_size = input_data.shape[0]

        inputs = [uint8_images_input(input_data)]

        adapter = self._get_detection_adapter(model_name)
        outputs = [InferRequestedOutput(name) for name in adapter.requested_outputs]

        response = self._infer_with_retry(model_name, inputs, outputs)

        return [
            {
                **detections,
                'orig_shape': orig_shapes[i],
                'scale': scales[i],
                'padding': paddings[i],
                'input_size': size,
            }
            for i, detections in enumerate(adapter.parse(response, batch_size))
        ]

    # =========================================================================
    # YOLO + MobileCLIP: CPU Preprocessing (stable, high-throughput)
    # =========================================================================

    def infer_yolo_clip_cpu(self, image_bytes: bytes) -> dict[str, Any]:
        """
        YOLO + MobileCLIP with CPU preprocessing for stable high-throughput inference.

        Pipeline:
        1. CPU decode (PIL/cv2)
        2. CPU letterbox for YOLO (640x640)
        3. CPU resize/crop for CLIP (256x256)
        4. Direct TRT inference for YOLO and CLIP

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with detections and image embedding
        """
        # CPU decode image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        orig_w, orig_h = img.size
        img_array = np.array(img)  # HWC, RGB, uint8

        # YOLO preprocessing (CPU letterbox) at the configured detector's own size
        yolo_model = TritonModelConfig.YOLO_MODEL
        yolo_size = self._model_input_size(yolo_model)
        yolo_input, scale, padding = self._preprocess_yolo_cpu(img_array, yolo_size)

        # CLIP preprocessing (CPU resize/crop)
        clip_input = self._preprocess_clip_cpu(img_array)

        # Run YOLO TRT inference (default detector; adapter handles either
        # the end2end 4-tensor or the fused single-tensor contract)
        adapter = self._get_detection_adapter(yolo_model)
        yolo_inputs = [uint8_images_input(yolo_input)]
        yolo_outputs = [InferRequestedOutput(name) for name in adapter.requested_outputs]
        yolo_response = self._infer_with_retry(yolo_model, yolo_inputs, yolo_outputs)

        # Run CLIP TRT inference
        clip_inputs = [InferInput('images', clip_input.shape, 'FP32')]
        clip_inputs[0].set_data_from_numpy(clip_input)
        clip_outputs = [InferRequestedOutput('image_embeddings')]
        clip_response = self._infer_with_retry(
            TritonModelConfig.CLIP_IMAGE_MODEL, clip_inputs, clip_outputs
        )

        # Parse outputs
        detections = adapter.parse(yolo_response, batch_size=1)[0]
        image_embedding = clip_response.as_numpy('image_embeddings')[0]

        # DF1: per-box CLIP embeddings, cropped from the FULL-RESOLUTION
        # image (not the 640x640 letterboxed YOLO input) — same approach as
        # the /embed/boxes route. Without this, generic ingest indexed no
        # per-box vectors at all (it read keys this method never returned).
        num_dets = int(detections.get('num_dets', 0))
        normalized_boxes = np.empty((0, 4), dtype=np.float32)
        box_embeddings = np.empty((0, 512), dtype=np.float32)
        if num_dets > 0:
            normalized_boxes = inverse_letterbox_coords(
                np.asarray(detections['boxes'], dtype=np.float64),
                (orig_h, orig_w),
                scale,
                padding,
                yolo_size,
            ).astype(np.float32)
            from src.utils.cpu_preprocess import embed_boxes_from_full_res

            box_embeddings = embed_boxes_from_full_res(
                img_array, normalized_boxes, self.infer_mobileclip_batch
            )

        return {
            **detections,
            'image_embedding': image_embedding,
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
            'input_size': yolo_size,
            'normalized_boxes': normalized_boxes,
            'box_embeddings': box_embeddings,
        }

    def _preprocess_yolo_cpu(
        self, img_array: np.ndarray, target_size: int
    ) -> tuple[np.ndarray, float, tuple]:
        """CPU letterbox preprocessing for YOLO at ``target_size``."""
        return preprocess_yolo_cpu(img_array, target_size)

    def _preprocess_clip_cpu(self, img_array: np.ndarray) -> np.ndarray:
        """CPU preprocessing for MobileCLIP (256x256, center crop)."""
        return preprocess_clip_cpu(img_array)

    # =========================================================================
    # Individual Model Inference (MobileCLIP Components)
    # =========================================================================

    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        """Encode image to 512-dim embedding via MobileCLIP."""
        img_array = self._preprocess_for_mobileclip(image_bytes)

        input_tensor = InferInput('images', [1, 3, 256, 256], 'FP32')
        input_tensor.set_data_from_numpy(img_array)

        output = InferRequestedOutput('image_embeddings')
        clip_image_model = TritonModelConfig.CLIP_IMAGE_MODEL
        response = self._infer_with_retry(clip_image_model, [input_tensor], [output])

        return response.as_numpy('image_embeddings')[0]

    def encode_text(self, tokens: np.ndarray) -> np.ndarray:
        """Encode tokenized text to 512-dim embedding via MobileCLIP."""
        input_tensor = InferInput('text_tokens', [1, 77], 'INT64')
        input_tensor.set_data_from_numpy(tokens.astype(np.int64))

        output = InferRequestedOutput('text_embeddings')
        clip_text_model = TritonModelConfig.CLIP_TEXT_MODEL
        response = self._infer_with_retry(clip_text_model, [input_tensor], [output])

        return response.as_numpy('text_embeddings')[0]

    def _preprocess_for_mobileclip(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for MobileCLIP (256x256, BILINEAR, center crop)."""
        return preprocess_for_mobileclip(image_bytes)

    # =========================================================================
    # OCR: PP-OCRv5 Text Detection and Recognition
    # =========================================================================

    def infer_ocr(self, image_bytes: bytes) -> dict[str, Any]:
        """
        OCR inference: PP-OCRv5 text detection and recognition via BLS pipeline.

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with num_texts, texts, text_boxes, text_boxes_normalized,
            text_scores, rec_scores
        """
        return infer_ocr(self._infer_with_retry, image_bytes)

    # =========================================================================
    # Formatting Utilities
    # =========================================================================

    @staticmethod
    def format_detections(result: dict[str, Any], model_name: str | None = None) -> list:
        """Format detections with coordinates normalized to original image dimensions.

        ``model_name``
        resolves ``class_name`` from that model's own labels instead of
        always assuming the stock COCO vocabulary -- pass the same
        ``model_name`` the detection came from (see
        :func:`src.utils.affine.format_detections_from_triton`).
        """
        return format_detections_from_triton(
            result, input_size=result.get('input_size', 640), model_name=model_name
        )

    # =========================================================================
    # Optimized Batched Inference Methods (Bypass Python BLS)
    # =========================================================================

    def infer_mobileclip_batch(
        self,
        crops: np.ndarray,
        max_batch_size: int = 64,
    ) -> np.ndarray:
        """
        Batched MobileCLIP inference for multiple image crops.

        Args:
            crops: [N, 3, 256, 256] FP32 normalized tensor
            max_batch_size: Maximum batch size per Triton request

        Returns:
            [N, 512] FP32 L2-normalized embeddings
        """
        if crops.ndim != 4 or crops.shape[1:] != (3, 256, 256):
            raise ValueError(f'Expected crops shape [N, 3, 256, 256], got {crops.shape}')
        if crops.dtype != np.float32:
            crops = crops.astype(np.float32)

        if crops.shape[0] == 0:
            return np.empty((0, 512), dtype=np.float32)

        n_crops = crops.shape[0]
        all_embeddings = []

        for i in range(0, n_crops, max_batch_size):
            batch = crops[i : i + max_batch_size]
            batch_size = batch.shape[0]

            input_tensor = InferInput('images', [batch_size, 3, 256, 256], 'FP32')
            input_tensor.set_data_from_numpy(batch.astype(np.float32))

            output = InferRequestedOutput('image_embeddings')

            response = self._infer_with_retry(
                TritonModelConfig.CLIP_IMAGE_MODEL, [input_tensor], [output]
            )

            embeddings = response.as_numpy('image_embeddings')
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 512), dtype=np.float32)


# =============================================================================
# Singleton Instance
# =============================================================================
_client_instance: TritonClient | None = None


def get_sync_triton_client(triton_url: str = 'triton-server:8001') -> TritonClient:
    """Get singleton Triton client instance."""
    global _client_instance  # noqa: PLW0603  # module-level cache set lazily
    if _client_instance is None:
        _client_instance = TritonClient(triton_url)
    return _client_instance
