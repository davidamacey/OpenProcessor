"""
Unified Triton Client for inference pipelines.

Single client class that handles all Triton inference with proper connection pooling
and shared utilities. Sync-only for backpressure safety (async causes server deadlock
at 256+ clients).

Architecture:
- Shared gRPC connection pool via TritonClientManager
- Shared affine matrix calculation and JPEG parsing from utils/affine.py
- Shared detection formatting from utils/affine.py
- Clear method names: infer_yolo_end2end(), infer_yolo_clip(), infer_yolo_clip_cpu()

Performance:
- Sync client with thread pool: Proper backpressure, 200+ RPS
- Shared connection enables Triton dynamic batching (5-10x throughput)
"""

import io
import logging
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image
from tritonclient.grpc import InferInput, InferRequestedOutput
from ultralytics.data.augment import LetterBox

from src.clients.triton_pool import TritonClientManager
from src.utils.affine import (
    calculate_affine_matrix,
    format_detections_from_triton,
    get_jpeg_dimensions_fast,
)
from src.utils.retry import retry_sync


if TYPE_CHECKING:
    from src.services.cpu_preprocess import PreprocessResult


logger = logging.getLogger(__name__)


class TritonClient:
    """
    Unified Triton client for inference pipelines.

    CRITICAL: Uses sync gRPC only (not async) to prevent server deadlock
    at high concurrency (256+ clients). FastAPI handles async via thread pool.

    Pipelines:
    - infer_yolo_end2end: End2End TRT + GPU NMS (CPU preprocessing)
    - infer_yolo_dali: DALI + TRT (100% GPU pipeline) [deprecated]
    - infer_yolo_clip: YOLO + MobileCLIP ensemble (visual search, DALI)
    - infer_yolo_clip_cpu: YOLO + MobileCLIP (CPU preprocessing)
    """

    def __init__(
        self,
        triton_url: str = 'triton-api:8001',
        max_retries: int = 3,
        retry_base_delay: float = 0.1,
        retry_max_delay: float = 5.0,
    ):
        """
        Initialize unified Triton client with retry support.

        Args:
            triton_url: Triton gRPC endpoint
            max_retries: Maximum retry attempts for failed requests
            retry_base_delay: Initial retry delay in seconds
            retry_max_delay: Maximum retry delay in seconds
        """
        self.triton_url = triton_url
        self.client = TritonClientManager.get_sync_client(triton_url)
        self.input_size = 640  # YOLO input size
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        logger.info(f'Unified Triton client initialized (sync, retries={max_retries})')

    def _infer_with_retry(self, model_name: str, inputs: list, outputs: list):
        """
        Execute Triton inference with automatic retry on transient failures.

        Retries on: queue full, resource exhausted, timeout, unavailable.
        Does NOT retry on: invalid input, model not found, etc.

        Args:
            model_name: Triton model name
            inputs: List of InferInput objects
            outputs: List of InferRequestedOutput objects

        Returns:
            InferResult from Triton

        Raises:
            RetryExhaustedError: If all retries exhausted
            Exception: For non-retryable errors
        """
        return retry_sync(
            self.client.infer,
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
        )

    # =========================================================================
    # YOLO End2End: CPU Preprocessing + TensorRT + GPU NMS
    # =========================================================================
    def infer_yolo_end2end(self, image_array: np.ndarray, model_name: str) -> dict[str, Any]:
        """
        YOLO End2End inference: CPU preprocessing + TensorRT with GPU NMS.

        Uses CPU-based letterbox preprocessing then runs TensorRT End2End model
        which includes GPU-accelerated NMS.

        Args:
            image_array: Preprocessed image (HWC, BGR, 0-255) from cv2.imdecode
            model_name: Triton model name (e.g., "yolov11_small_trt_end2end")

        Returns:
            Dict with num_dets, boxes, scores, classes, orig_shape, scale, padding
        """
        # Store original dimensions
        orig_h, orig_w = image_array.shape[:2]

        # Convert BGR (from cv2.imdecode) to RGB (YOLO trained on RGB)
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # Use Ultralytics LetterBox for exact preprocessing match
        letterbox = LetterBox(
            new_shape=(self.input_size, self.input_size), auto=False, scaleup=False
        )
        img_letterbox = letterbox(image=image_rgb)

        # Calculate transformation parameters (for inverse transform)
        scale = min(self.input_size / orig_h, self.input_size / orig_w)
        scale = min(scale, 1.0)  # scaleup=False

        # Calculate padding
        new_unpad_w = round(orig_w * scale)
        new_unpad_h = round(orig_h * scale)
        pad_w = (self.input_size - new_unpad_w) / 2.0
        pad_h = (self.input_size - new_unpad_h) / 2.0
        padding = (pad_w, pad_h)

        # Normalize to 0-1
        img_norm = img_letterbox.astype(np.float32) / 255.0

        # HWC to CHW
        img_chw = np.transpose(img_norm, (2, 0, 1))

        # Add batch dimension
        input_data = np.expand_dims(img_chw, axis=0)

        # Create Triton inputs
        inputs = [InferInput('images', input_data.shape, 'FP32')]
        inputs[0].set_data_from_numpy(input_data)

        # Create Triton outputs
        outputs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
        ]

        # Run inference with retry
        response = self._infer_with_retry(model_name, inputs, outputs)

        # Parse outputs
        num_dets = int(response.as_numpy('num_dets')[0][0])
        boxes = response.as_numpy('det_boxes')[0][:num_dets]
        scores = response.as_numpy('det_scores')[0][:num_dets]
        classes = response.as_numpy('det_classes')[0][:num_dets]

        return {
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

    def infer_yolo_end2end_batch(self, images: list, model_name: str) -> list:
        """
        Batch YOLO End2End inference: CPU preprocessing + TensorRT with GPU NMS.

        Processes multiple images in a single batched request for higher throughput.

        Args:
            images: List of images (HWC, BGR, 0-255) from cv2.imdecode
            model_name: Triton model name (e.g., "yolov11_small_trt_end2end")

        Returns:
            List of dicts with num_dets, boxes, scores, classes, orig_shape, scale, padding
        """
        letterbox = LetterBox(
            new_shape=(self.input_size, self.input_size), auto=False, scaleup=False
        )

        # Store original dimensions and preprocess
        orig_shapes = []
        scales = []
        paddings = []
        preprocessed = []

        for img in images:
            orig_h, orig_w = img.shape[:2]
            orig_shapes.append((orig_h, orig_w))

            # Convert BGR to RGB (YOLO trained on RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Letterbox transform
            img_letterbox = letterbox(image=img_rgb)

            # Calculate scale and padding
            scale = min(self.input_size / orig_h, self.input_size / orig_w)
            scale = min(scale, 1.0)
            scales.append(scale)

            new_unpad_w = round(orig_w * scale)
            new_unpad_h = round(orig_h * scale)
            pad_w = (self.input_size - new_unpad_w) / 2.0
            pad_h = (self.input_size - new_unpad_h) / 2.0
            paddings.append((pad_w, pad_h))

            # Normalize and transpose
            img_norm = img_letterbox.astype(np.float32) / 255.0
            img_chw = np.transpose(img_norm, (2, 0, 1))
            preprocessed.append(img_chw)

        # Stack into batch
        input_data = np.stack(preprocessed, axis=0)
        batch_size = input_data.shape[0]

        # Create Triton inputs/outputs
        inputs = [InferInput('images', input_data.shape, 'FP32')]
        inputs[0].set_data_from_numpy(input_data)

        outputs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
        ]

        # Run inference with retry
        response = self._infer_with_retry(model_name, inputs, outputs)

        # Parse batch outputs
        num_dets_batch = response.as_numpy('num_dets')
        boxes_batch = response.as_numpy('det_boxes')
        scores_batch = response.as_numpy('det_scores')
        classes_batch = response.as_numpy('det_classes')

        results = []
        for i in range(batch_size):
            num_dets = int(num_dets_batch[i][0])
            results.append(
                {
                    'num_dets': num_dets,
                    'boxes': boxes_batch[i][:num_dets],
                    'scores': scores_batch[i][:num_dets],
                    'classes': classes_batch[i][:num_dets],
                    'orig_shape': orig_shapes[i],
                    'scale': scales[i],
                    'padding': paddings[i],
                }
            )

        return results

    # =========================================================================
    # YOLO DALI: Full GPU Pipeline (DEPRECATED - use infer_yolo_clip_cpu instead)
    # =========================================================================
    def infer_yolo_dali(
        self, image_bytes: bytes, model_name: str, auto_affine: bool = False
    ) -> dict[str, Any]:
        """
        DEPRECATED: Use infer_yolo_clip_cpu() for stable high-throughput inference.

        Full GPU pipeline with DALI preprocessing. While theoretically faster,
        DALI ensembles can have stability issues at high concurrency. CPU
        preprocessing with direct TRT calls provides more reliable throughput.

        GPU pipeline:
        1. nvJPEG decode (GPU)
        2. warp_affine letterbox (GPU)
        3. normalize + CHW transpose (GPU)
        4. TensorRT inference + NMS (GPU)

        CPU overhead: Only JPEG header parse (~0.1ms) + affine calc (~0.001ms)

        Args:
            image_bytes: Raw JPEG/PNG bytes
            model_name: Triton ensemble name (e.g., "yolov11_small_gpu_e2e")
            auto_affine: If True, use auto-affine (100% GPU, no CPU affine calc)

        Returns:
            Dict with num_dets, boxes, scores, classes, orig_shape, scale, padding
        """
        # Fast JPEG header parse for dimensions (~0.1ms)
        orig_w, orig_h = get_jpeg_dimensions_fast(image_bytes)

        # Prepare encoded image input
        input_data = np.frombuffer(image_bytes, dtype=np.uint8)
        input_data = np.expand_dims(input_data, axis=0)

        inputs = [InferInput('encoded_images', input_data.shape, 'UINT8')]
        inputs[0].set_data_from_numpy(input_data)

        if not auto_affine:
            # CPU affine calculation (cached, ~0.001ms for cache hit)
            affine_matrix, scale, padding = calculate_affine_matrix(orig_w, orig_h, self.input_size)
            input_affine = InferInput('affine_matrices', [1, 2, 3], 'FP32')
            input_affine.set_data_from_numpy(np.expand_dims(affine_matrix, axis=0))
            inputs.append(input_affine)
        else:
            # Auto-affine: GPU calculates affine matrix
            scale = min(self.input_size / orig_w, self.input_size / orig_h)
            padding = (0.0, 0.0)

        # Create Triton outputs
        outputs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
        ]

        # Run inference with retry
        response = self._infer_with_retry(model_name, inputs, outputs)

        # Parse outputs
        num_dets = int(response.as_numpy('num_dets')[0][0])
        boxes = response.as_numpy('det_boxes')[0][:num_dets]
        scores = response.as_numpy('det_scores')[0][:num_dets]
        classes = response.as_numpy('det_classes')[0][:num_dets]

        return {
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

    # =========================================================================
    # YOLO + MobileCLIP: Visual Search Ensemble (DALI preprocessing)
    # =========================================================================
    def infer_yolo_clip(self, image_bytes: bytes, full_pipeline: bool = False) -> dict[str, Any]:
        """
        YOLO + MobileCLIP ensemble inference for visual search.

        Uses DALI for GPU preprocessing. Combines object detection with
        image embedding extraction in a single request.

        Ensembles:
        - yolo_clip_ensemble: YOLO detections + global image embedding
        - yolo_mobileclip_ensemble: + per-box embeddings (full_pipeline=True)

        Pipeline (all in single Triton call):
        1. GPU JPEG decode (nvJPEG)
        2. GPU letterbox for YOLO (warp_affine)
        3. GPU resize/crop for CLIP
        4. GPU inference + NMS (TensorRT End2End)
        5. GPU image encoding (MobileCLIP)
        6. [Full only] Per-box embeddings (ROI align + MobileCLIP)

        Args:
            image_bytes: Raw JPEG/PNG bytes
            full_pipeline: If True, include per-box embeddings

        Returns:
            Dict with detections, embeddings, and transformation params
        """
        # Fast JPEG header parse for dimensions (~0.1ms)
        orig_w, orig_h = get_jpeg_dimensions_fast(image_bytes)

        # Get cached affine matrix (~0.001ms for cache hit)
        affine_matrix, scale, padding = calculate_affine_matrix(orig_w, orig_h, self.input_size)

        # Prepare inputs
        input_data = np.frombuffer(image_bytes, dtype=np.uint8)
        input_data = np.expand_dims(input_data, axis=0)

        inputs = [
            InferInput('encoded_images', input_data.shape, 'UINT8'),
            InferInput('affine_matrices', [1, 2, 3], 'FP32'),
        ]
        inputs[0].set_data_from_numpy(input_data)
        inputs[1].set_data_from_numpy(np.expand_dims(affine_matrix, axis=0))

        # Choose ensemble and outputs
        # NOTE: yolo_mobileclip_ensemble (with box_embeddings) requires models that aren't built.
        # Using yolo_clip_ensemble for both cases - box embeddings computed separately if needed.
        ensemble_name = 'yolo_clip_ensemble'
        outputs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
            InferRequestedOutput('global_embeddings'),
        ]

        # Sync inference with retry - proper backpressure handling
        response = self._infer_with_retry(ensemble_name, inputs, outputs)

        # Parse outputs
        num_dets = int(response.as_numpy('num_dets')[0][0])
        boxes = response.as_numpy('det_boxes')[0][:num_dets]
        scores = response.as_numpy('det_scores')[0][:num_dets]
        classes = response.as_numpy('det_classes')[0][:num_dets]
        image_embedding = response.as_numpy('global_embeddings')[0]

        result = {
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'image_embedding': image_embedding,
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

        # Note: Box embeddings not available from yolo_clip_ensemble
        # They would need to be computed separately if needed
        if full_pipeline:
            result['box_embeddings'] = np.array([])
            result['normalized_boxes'] = boxes.copy() if num_dets > 0 else np.array([])

        return result

    def infer_yolo_clip_batch(
        self, images_bytes: list[bytes], max_workers: int = 32
    ) -> list[dict[str, Any]]:
        """
        Batch YOLO + MobileCLIP inference: Process multiple images in parallel.

        Uses ThreadPoolExecutor to send parallel requests to Triton,
        which batches them via dynamic batching for optimal GPU utilization.

        For large photo libraries (50K+ images), batch sizes of 16-64
        significantly improve throughput by:
        - Reducing HTTP overhead
        - Ensuring full DALI/TRT batch utilization
        - Maximizing GPU parallelism

        Args:
            images_bytes: List of raw JPEG/PNG bytes (up to 64 images)
            max_workers: Max parallel threads (default 32)

        Returns:
            List of result dicts with detections and embeddings
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        batch_size = len(images_bytes)
        if batch_size == 0:
            return []

        # Limit batch size to max_batch_size
        if batch_size > 64:
            logger.warning(f'Batch size {batch_size} exceeds max 64, truncating')
            images_bytes = images_bytes[:64]
            batch_size = 64

        results = [None] * batch_size

        def process_single(idx: int, img_bytes: bytes) -> tuple[int, dict]:
            """Process a single image and return (index, result)."""
            result = self.infer_yolo_clip(img_bytes, full_pipeline=False)
            return idx, result

        # Process in parallel
        workers = min(max_workers, batch_size)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single, i, img): i for i, img in enumerate(images_bytes)
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def infer_yolo_clip_batch_full(
        self,
        images_bytes: list[bytes],
        max_workers: int = 64,
    ) -> list[dict[str, Any]]:
        """
        Batch YOLO + MobileCLIP with full pipeline (detections + per-box embeddings).

        Optimized for large photo library ingestion:
        - Submits ALL images to Triton simultaneously
        - Triton's dynamic batcher groups them (16-48 avg batch size)
        - Returns full pipeline results including box embeddings

        Performance: 3-5x faster than sequential /ingest calls.
        Target: 300+ RPS with batch sizes of 32-64.

        Args:
            images_bytes: List of raw JPEG/PNG bytes (max 64 images)
            max_workers: Max parallel threads for Triton submission

        Returns:
            List of result dicts with detections, embeddings, and box data
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        batch_size = len(images_bytes)
        if batch_size == 0:
            return []

        # Limit batch size
        if batch_size > 64:
            logger.warning(f'Batch size {batch_size} exceeds max 64, truncating')
            images_bytes = images_bytes[:64]
            batch_size = 64

        results = [None] * batch_size

        def process_single(idx: int, img_bytes: bytes) -> tuple[int, dict]:
            """Process a single image with full pipeline."""
            try:
                result = self.infer_yolo_clip(img_bytes, full_pipeline=True)
                return idx, result
            except Exception as e:
                logger.error(f'Batch inference failed for image {idx}: {e}')
                return idx, {'error': str(e), 'num_dets': 0}

        # Submit ALL images in parallel - Triton will batch them
        workers = min(max_workers, batch_size)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single, i, img): i for i, img in enumerate(images_bytes)
            }

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    # =========================================================================
    # Individual Model Inference (MobileCLIP Components)
    # =========================================================================
    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Encode image to 512-dim embedding via MobileCLIP.

        Args:
            image_bytes: JPEG/PNG bytes

        Returns:
            512-dim L2-normalized embedding
        """
        # Preprocess for MobileCLIP (256x256, normalized)
        img_array = self._preprocess_for_mobileclip(image_bytes)

        # Prepare input
        input_tensor = InferInput('images', [1, 3, 256, 256], 'FP32')
        input_tensor.set_data_from_numpy(img_array)

        # Output
        output = InferRequestedOutput('image_embeddings')

        # Sync inference with retry
        response = self._infer_with_retry('mobileclip2_s2_image_encoder', [input_tensor], [output])

        return response.as_numpy('image_embeddings')[0]

    def encode_text(self, tokens: np.ndarray) -> np.ndarray:
        """
        Encode tokenized text to 512-dim embedding via MobileCLIP.

        Args:
            tokens: Tokenized text [1, 77] INT64

        Returns:
            512-dim L2-normalized embedding
        """
        # Prepare input
        input_tensor = InferInput('text_tokens', [1, 77], 'INT64')
        input_tensor.set_data_from_numpy(tokens.astype(np.int64))

        # Output
        output = InferRequestedOutput('text_embeddings')

        # Sync inference with retry
        response = self._infer_with_retry('mobileclip2_s2_text_encoder', [input_tensor], [output])

        return response.as_numpy('text_embeddings')[0]

    # =========================================================================
    # YOLO + MobileCLIP: CPU Preprocessing (stable, high-throughput)
    # =========================================================================
    def infer_yolo_clip_cpu(self, image_bytes: bytes) -> dict[str, Any]:
        """
        YOLO + MobileCLIP with CPU preprocessing for stable high-throughput inference.

        Recommended for production ingestion pipelines. Uses CPU preprocessing
        with direct TRT model calls:
        1. CPU decode (PIL/cv2)
        2. CPU letterbox for YOLO (640x640)
        3. CPU resize/crop for CLIP (256x256)
        4. Direct TRT inference for YOLO and CLIP

        Advantages over DALI pipeline:
        - More stable at high concurrency (100% success rate)
        - Enables more TRT instances since DALI doesn't reserve VRAM
        - Simpler debugging and monitoring

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with detections and image embedding
        """
        # CPU decode image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        orig_w, orig_h = img.size
        img_array = np.array(img)  # HWC, RGB, uint8

        # ---- YOLO Preprocessing (CPU letterbox) ----
        yolo_input, scale, padding = self._preprocess_yolo_cpu(img_array)

        # ---- CLIP Preprocessing (CPU resize/crop) ----
        clip_input = self._preprocess_clip_cpu(img_array)

        # ---- Run YOLO TRT Inference ----
        yolo_inputs = [InferInput('images', yolo_input.shape, 'FP32')]
        yolo_inputs[0].set_data_from_numpy(yolo_input)
        yolo_outputs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
        ]
        yolo_response = self._infer_with_retry(
            'yolov11_small_trt_end2end', yolo_inputs, yolo_outputs
        )

        # ---- Run CLIP TRT Inference ----
        clip_inputs = [InferInput('images', clip_input.shape, 'FP32')]
        clip_inputs[0].set_data_from_numpy(clip_input)
        clip_outputs = [InferRequestedOutput('image_embeddings')]
        clip_response = self._infer_with_retry(
            'mobileclip2_s2_image_encoder', clip_inputs, clip_outputs
        )

        # Parse YOLO outputs
        num_dets = int(yolo_response.as_numpy('num_dets')[0][0])
        boxes = yolo_response.as_numpy('det_boxes')[0][:num_dets]
        scores = yolo_response.as_numpy('det_scores')[0][:num_dets]
        classes = yolo_response.as_numpy('det_classes')[0][:num_dets]

        # Parse CLIP output
        image_embedding = clip_response.as_numpy('image_embeddings')[0]

        return {
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'image_embedding': image_embedding,
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

    def _preprocess_yolo_cpu(self, img_array: np.ndarray) -> tuple[np.ndarray, float, tuple]:
        """
        CPU letterbox preprocessing for YOLO (matches DALI output exactly).

        Args:
            img_array: HWC, RGB, uint8 numpy array

        Returns:
            Tuple of:
            - preprocessed: [1, 3, 640, 640] FP32 normalized
            - scale: float for inverse transform
            - padding: (pad_x, pad_y) tuple
        """
        orig_h, orig_w = img_array.shape[:2]
        target_size = self.input_size  # 640

        # Calculate scale (don't upscale)
        scale = min(target_size / orig_h, target_size / orig_w)
        scale = min(scale, 1.0)

        # New dimensions after scaling
        new_w = round(orig_w * scale)
        new_h = round(orig_h * scale)

        # Resize using cv2 (faster than PIL for numpy arrays)
        if scale < 1.0:
            resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = img_array

        # Calculate padding
        pad_w = (target_size - new_w) / 2.0
        pad_h = (target_size - new_h) / 2.0
        top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
        left, right = round(pad_w - 0.1), round(pad_w + 0.1)

        # Add padding (gray = 114)
        letterboxed = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Ensure exact size (rounding may cause 1px difference)
        if letterboxed.shape[:2] != (target_size, target_size):
            letterboxed = cv2.resize(letterboxed, (target_size, target_size))

        # Normalize to [0, 1]
        normalized = letterboxed.astype(np.float32) / 255.0

        # HWC -> CHW
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched, scale, (pad_w, pad_h)

    def _preprocess_clip_cpu(self, img_array: np.ndarray) -> np.ndarray:
        """
        CPU preprocessing for MobileCLIP (256x256, center crop).

        Args:
            img_array: HWC, RGB, uint8 numpy array

        Returns:
            Preprocessed array [1, 3, 256, 256] FP32
        """
        orig_h, orig_w = img_array.shape[:2]
        target_size = 256

        # Resize shortest edge to 256
        scale = target_size / min(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center crop to 256x256
        start_x = (new_w - target_size) // 2
        start_y = (new_h - target_size) // 2
        cropped = resized[start_y : start_y + target_size, start_x : start_x + target_size]

        # Normalize to [0, 1]
        normalized = cropped.astype(np.float32) / 255.0

        # HWC -> CHW
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        return np.expand_dims(transposed, axis=0)

    # =========================================================================
    # Preprocessing Utilities
    # =========================================================================
    def _preprocess_for_mobileclip(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image for MobileCLIP inference.

        Uses BILINEAR interpolation to match Apple's MobileCLIP training config
        (OpenCLIP _mccfg sets interpolation='bilinear').

        Args:
            image_bytes: JPEG/PNG bytes

        Returns:
            Preprocessed image array [1, 3, 256, 256] FP32
        """
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Resize shortest edge to 256, then center crop (matches OpenCLIP MobileCLIP config)
        width, height = img.size
        scale = 256 / min(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

        # Center crop to 256x256
        left = (new_width - 256) // 2
        top = (new_height - 256) // 2
        img = img.crop((left, top, left + 256, top + 256))

        # Convert to numpy and normalize
        img_array = np.array(img).astype(np.float32)
        img_array = img_array / 255.0  # [0, 1]

        # Transpose HWC -> CHW
        img_array = np.transpose(img_array, (2, 0, 1))

        return img_array[np.newaxis, ...]  # Add batch dimension

    # =========================================================================
    # Full Face Pipeline: YOLO + SCRFD + MobileCLIP + ArcFace Unified Ensemble
    # =========================================================================
    def infer_faces_full(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Full face recognition pipeline: YOLO + SCRFD + MobileCLIP + ArcFace.

        All processing happens in Triton - single request, no round-trips:
        1. GPU JPEG decode (nvJPEG via DALI)
        2. Quad-branch preprocessing (YOLO 640, CLIP 256, SCRFD 640, HD original)
        3. YOLO object detection (parallel with 4, 5)
        4. MobileCLIP global embedding (parallel with 3, 5)
        5. SCRFD face detection + NMS (parallel with 3, 4)
        6. Per-box MobileCLIP embeddings (depends on 3)
        7. Per-face ArcFace embeddings (depends on 5)

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with:
                - YOLO detections (num_dets, boxes, scores, classes)
                - Global MobileCLIP embedding (512-dim)
                - Per-box MobileCLIP embeddings (300 x 512-dim)
                - Face detections (num_faces, face_boxes, face_landmarks, face_scores)
                - Face ArcFace embeddings (128 x 512-dim)
                - Face quality scores
        """
        # Fast JPEG header parse for dimensions (~0.1ms)
        orig_w, orig_h = get_jpeg_dimensions_fast(image_bytes)

        # Get cached affine matrix (~0.001ms for cache hit)
        affine_matrix, scale, padding = calculate_affine_matrix(orig_w, orig_h, self.input_size)

        # Prepare inputs
        input_data = np.frombuffer(image_bytes, dtype=np.uint8)
        input_data = np.expand_dims(input_data, axis=0)

        orig_shape = np.array([orig_h, orig_w], dtype=np.int32)
        orig_shape = np.expand_dims(orig_shape, axis=0)

        inputs = [
            InferInput('encoded_images', input_data.shape, 'UINT8'),
            InferInput('affine_matrices', [1, 2, 3], 'FP32'),
            InferInput('orig_shape', [1, 2], 'INT32'),
        ]
        inputs[0].set_data_from_numpy(input_data)
        inputs[1].set_data_from_numpy(np.expand_dims(affine_matrix, axis=0))
        inputs[2].set_data_from_numpy(orig_shape)

        # Request all outputs from unified ensemble
        outputs = [
            # YOLO detections
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
            # MobileCLIP global embedding
            InferRequestedOutput('global_embeddings'),
            # Face detections and embeddings
            InferRequestedOutput('num_faces'),
            InferRequestedOutput('face_boxes'),
            InferRequestedOutput('face_landmarks'),
            InferRequestedOutput('face_scores'),
            InferRequestedOutput('face_embeddings'),
            InferRequestedOutput('face_quality'),
        ]

        # Sync inference with retry
        response = self._infer_with_retry('yolo_face_clip_ensemble', inputs, outputs)

        # Parse YOLO outputs
        num_dets = int(response.as_numpy('num_dets')[0][0])
        boxes = response.as_numpy('det_boxes')[0][:num_dets]
        scores = response.as_numpy('det_scores')[0][:num_dets]
        classes = response.as_numpy('det_classes')[0][:num_dets]
        image_embedding = response.as_numpy('global_embeddings')[0]

        # Parse face outputs
        num_faces_arr = response.as_numpy('num_faces')
        num_faces = int(num_faces_arr.flatten()[0])
        face_boxes_raw = response.as_numpy('face_boxes')
        face_landmarks_raw = response.as_numpy('face_landmarks')
        face_scores_raw = response.as_numpy('face_scores')

        # Handle batch dimension
        if face_boxes_raw.ndim == 3:
            face_boxes = face_boxes_raw[0][:num_faces]
            face_landmarks = face_landmarks_raw[0][:num_faces]
            face_scores_arr = face_scores_raw[0][:num_faces]
        else:
            face_boxes = face_boxes_raw[:num_faces]
            face_landmarks = face_landmarks_raw[:num_faces]
            face_scores_arr = face_scores_raw[:num_faces]

        # Parse face embeddings
        if num_faces > 0:
            face_emb_raw = response.as_numpy('face_embeddings')
            face_quality_raw = response.as_numpy('face_quality')

            # Handle batch dimension
            if face_emb_raw.ndim == 3:
                face_emb = face_emb_raw[0].reshape(-1, 512)[:num_faces]
                face_quality_arr = face_quality_raw[0][:num_faces]
            else:
                face_emb = face_emb_raw.reshape(-1, 512)[:num_faces]
                face_quality_arr = face_quality_raw[:num_faces]
        else:
            face_emb = np.array([])
            face_quality_arr = np.array([])

        return {
            # YOLO detections
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            # MobileCLIP global embedding
            'image_embedding': image_embedding,
            # Face detections and embeddings
            'num_faces': num_faces,
            'face_boxes': face_boxes,
            'face_landmarks': face_landmarks,
            'face_scores': face_scores_arr,
            'face_embeddings': face_emb,
            'face_quality': face_quality_arr,
            # Transform params
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

    def infer_faces_only(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Face detection and recognition only (no YOLO).

        Simpler pipeline for face-focused applications:
        1. GPU JPEG decode + preprocess for SCRFD (640x640)
        2. SCRFD face detection + NMS
        3. Face alignment + ArcFace embedding extraction

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with face detections and ArcFace embeddings
        """
        # Use the full pipeline and extract face parts only
        # This ensures we use the optimized DALI quad-branch
        result = self.infer_faces_full(image_bytes)

        return {
            'num_faces': result['num_faces'],
            'face_boxes': result['face_boxes'],
            'face_landmarks': result['face_landmarks'],
            'face_scores': result['face_scores'],
            'face_embeddings': result['face_embeddings'],
            'face_quality': result['face_quality'],
            'orig_shape': result['orig_shape'],
        }

    # =========================================================================
    # Formatting Utilities (Shared across all tracks)
    # =========================================================================
    # Unified Pipeline: YOLO + MobileCLIP + Person-only Face Detection
    # =========================================================================
    def infer_unified(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Unified pipeline: YOLO + MobileCLIP + person-only face detection.

        More efficient than full-image face detection - runs SCRFD only on
        person bounding box crops. Combines all embeddings in one request.

        Pipeline:
        1. GPU JPEG decode (nvJPEG via DALI)
        2. Triple preprocessing (YOLO 640, CLIP 256, HD original)
        3. YOLO object detection (parallel with 4)
        4. MobileCLIP global embedding (parallel with 3)
        5. Unified extraction:
           - MobileCLIP per-box embeddings (all boxes)
           - SCRFD face detection (person boxes only)
           - ArcFace face embeddings

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with:
                - YOLO detections (num_dets, boxes, scores, classes)
                - Global MobileCLIP embedding (512-dim)
                - Per-box MobileCLIP embeddings (300 x 512-dim)
                - Face detections (num_faces, face_boxes, face_landmarks, face_scores)
                - Face ArcFace embeddings (64 x 512-dim)
                - Face person index (which person box each face belongs to)
        """
        # Fast JPEG header parse for dimensions
        orig_w, orig_h = get_jpeg_dimensions_fast(image_bytes)

        # Get cached affine matrix
        affine_matrix, scale, padding = calculate_affine_matrix(orig_w, orig_h, self.input_size)

        # Prepare inputs
        input_data = np.frombuffer(image_bytes, dtype=np.uint8)
        input_data = np.expand_dims(input_data, axis=0)

        inputs = [
            InferInput('encoded_images', input_data.shape, 'UINT8'),
            InferInput('affine_matrices', [1, 2, 3], 'FP32'),
        ]
        inputs[0].set_data_from_numpy(input_data)
        inputs[1].set_data_from_numpy(np.expand_dims(affine_matrix, axis=0))

        # Request all outputs from unified ensemble
        outputs = [
            # YOLO detections
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
            # MobileCLIP embeddings
            InferRequestedOutput('global_embeddings'),
            InferRequestedOutput('box_embeddings'),
            InferRequestedOutput('normalized_boxes'),
            # Face detections and embeddings
            InferRequestedOutput('num_faces'),
            InferRequestedOutput('face_embeddings'),
            InferRequestedOutput('face_boxes'),
            InferRequestedOutput('face_landmarks'),
            InferRequestedOutput('face_scores'),
            InferRequestedOutput('face_person_idx'),
        ]

        # Sync inference with retry
        response = self._infer_with_retry('yolo_unified_ensemble', inputs, outputs)

        # Parse YOLO outputs
        num_dets = int(response.as_numpy('num_dets')[0][0])
        boxes = response.as_numpy('det_boxes')[0][:num_dets]
        scores = response.as_numpy('det_scores')[0][:num_dets]
        classes = response.as_numpy('det_classes')[0][:num_dets]

        # Parse MobileCLIP outputs
        global_embedding = response.as_numpy('global_embeddings')[0]
        box_embeddings_raw = response.as_numpy('box_embeddings')
        normalized_boxes_raw = response.as_numpy('normalized_boxes')

        # Handle batch dimension
        if box_embeddings_raw.ndim == 3:
            box_embeddings = box_embeddings_raw[0][:num_dets]
            normalized_boxes = normalized_boxes_raw[0][:num_dets]
        else:
            box_embeddings = box_embeddings_raw[:num_dets]
            normalized_boxes = normalized_boxes_raw[:num_dets]

        # Parse face outputs
        num_faces_arr = response.as_numpy('num_faces')
        num_faces = int(num_faces_arr.flatten()[0])

        face_boxes_raw = response.as_numpy('face_boxes')
        face_landmarks_raw = response.as_numpy('face_landmarks')
        face_scores_raw = response.as_numpy('face_scores')
        face_person_idx_raw = response.as_numpy('face_person_idx')
        face_emb_raw = response.as_numpy('face_embeddings')

        # Handle batch dimension for faces
        if face_boxes_raw.ndim == 3:
            face_boxes = face_boxes_raw[0][:num_faces]
            face_landmarks = face_landmarks_raw[0][:num_faces]
            face_scores_arr = face_scores_raw[0][:num_faces]
            face_person_idx = face_person_idx_raw[0][:num_faces]
            face_embeddings = face_emb_raw[0][:num_faces] if num_faces > 0 else np.array([])
        else:
            face_boxes = face_boxes_raw[:num_faces]
            face_landmarks = face_landmarks_raw[:num_faces]
            face_scores_arr = face_scores_raw[:num_faces]
            face_person_idx = face_person_idx_raw[:num_faces]
            face_embeddings = face_emb_raw[:num_faces] if num_faces > 0 else np.array([])

        return {
            # YOLO detections
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            # MobileCLIP embeddings
            'global_embedding': global_embedding,
            'box_embeddings': box_embeddings,
            'normalized_boxes': normalized_boxes,
            # Face detections and embeddings (from person crops only)
            'num_faces': num_faces,
            'face_boxes': face_boxes,
            'face_landmarks': face_landmarks,
            'face_scores': face_scores_arr,
            'face_embeddings': face_embeddings,
            'face_person_idx': face_person_idx,
            # Transform params
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

    # =========================================================================
    # OCR: PP-OCRv5 Text Detection and Recognition
    # =========================================================================

    def infer_ocr(self, image_bytes: bytes) -> dict[str, Any]:
        """
        OCR inference: PP-OCRv5 text detection and recognition.

        Pipeline:
        1. Preprocess image (resize, normalize)
        2. Call OCR pipeline BLS (detection + recognition)
        3. Return text, boxes, and confidence scores

        Args:
            image_bytes: Raw JPEG/PNG bytes

        Returns:
            Dict with OCR results:
            - num_texts: Number of text regions detected
            - texts: List of detected text strings
            - text_boxes: [N, 8] Quadrilateral boxes
            - text_boxes_normalized: [N, 4] Axis-aligned boxes normalized
            - text_scores: Detection confidence scores
            - rec_scores: Recognition confidence scores
        """
        import cv2

        # Decode image (handle RGBA by reading with alpha then converting)
        img_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if img_array is None:
            return {
                'num_texts': 0,
                'texts': [],
                'text_boxes': np.array([]),
                'text_boxes_normalized': np.array([]),
                'text_scores': np.array([]),
                'rec_scores': np.array([]),
            }

        # Handle different channel formats
        if len(img_array.shape) == 2:
            # Grayscale -> BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            # RGBA/BGRA -> BGR (composite alpha onto white background)
            alpha = img_array[:, :, 3:4] / 255.0
            rgb = img_array[:, :, :3]
            white_bg = np.ones_like(rgb) * 255
            img_array = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)

        orig_h, orig_w = img_array.shape[:2]

        # Preprocess for OCR detection
        # PP-OCR approach: scale to fit within limit, pad to 32-boundary
        # For small images (< 640px), upscale to ensure text is detectable
        ocr_limit_side = 960
        ocr_min_side = 640  # Minimum dimension for reliable text detection

        # Calculate scaling ratio
        max_side = max(orig_h, orig_w)
        if max_side > ocr_limit_side:
            # Downscale large images
            ratio = ocr_limit_side / max_side
        elif max_side < ocr_min_side:
            # Upscale small images for better detection
            # Use conservative upscaling (up to min_side, not limit_side)
            ratio = ocr_min_side / max_side
        else:
            # Keep original size
            ratio = 1.0

        resize_h = int(orig_h * ratio)
        resize_w = int(orig_w * ratio)

        # Ensure minimum size
        resize_h = max(32, resize_h)
        resize_w = max(32, resize_w)

        # Resize
        resized = cv2.resize(img_array, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        # Pad to 32-boundary
        pad_h = (32 - resize_h % 32) % 32
        pad_w = (32 - resize_w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            padded = np.zeros((resize_h + pad_h, resize_w + pad_w, 3), dtype=np.uint8)
            padded[:resize_h, :resize_w, :] = resized
            resized = padded

        # Normalize: (x / 127.5) - 1 for PP-OCR
        ocr_input = resized.astype(np.float32) / 127.5 - 1.0

        # HWC -> CHW (OpenCV reads as BGR, PP-OCR expects BGR)
        ocr_input = ocr_input.transpose(2, 0, 1)

        # Original image for cropping (normalize to [0, 1])
        orig_normalized = img_array.astype(np.float32) / 255.0
        orig_normalized = orig_normalized.transpose(2, 0, 1)

        # Prepare inputs for ocr_pipeline (max_batch_size: 0, no batch dim)
        inputs = [
            InferInput('ocr_images', list(ocr_input.shape), 'FP32'),
            InferInput('original_image', list(orig_normalized.shape), 'FP32'),
            InferInput('orig_shape', [2], 'INT32'),
        ]

        inputs[0].set_data_from_numpy(ocr_input)
        inputs[1].set_data_from_numpy(orig_normalized)
        inputs[2].set_data_from_numpy(np.array([orig_h, orig_w], dtype=np.int32))

        outputs = [
            InferRequestedOutput('num_texts'),
            InferRequestedOutput('text_boxes'),
            InferRequestedOutput('text_boxes_normalized'),
            InferRequestedOutput('texts'),
            InferRequestedOutput('text_scores'),
            InferRequestedOutput('rec_scores'),
        ]

        try:
            response = self._infer_with_retry('ocr_pipeline', inputs, outputs)
        except Exception as e:
            logger.error(f'OCR inference failed: {e}')
            return {
                'num_texts': 0,
                'texts': [],
                'text_boxes': np.array([]),
                'text_boxes_normalized': np.array([]),
                'text_scores': np.array([]),
                'rec_scores': np.array([]),
            }

        # Parse outputs
        num_texts_raw = response.as_numpy('num_texts')
        logger.info(
            f'OCR response: num_texts_raw shape={num_texts_raw.shape}, value={num_texts_raw}'
        )
        num_texts = int(num_texts_raw[0])
        text_boxes = response.as_numpy('text_boxes')[:num_texts]
        text_boxes_norm = response.as_numpy('text_boxes_normalized')[:num_texts]
        text_scores = response.as_numpy('text_scores')[:num_texts]
        rec_scores = response.as_numpy('rec_scores')[:num_texts]

        # Decode text strings (Triton returns bytes)
        texts_raw = response.as_numpy('texts')[:num_texts]
        texts = []
        for t in texts_raw:
            if isinstance(t, bytes):
                texts.append(t.decode('utf-8', errors='ignore'))
            elif isinstance(t, np.bytes_):
                texts.append(str(t, 'utf-8', errors='ignore'))
            else:
                texts.append(str(t))

        return {
            'num_texts': num_texts,
            'texts': texts,
            'text_boxes': text_boxes,
            'text_boxes_normalized': text_boxes_norm,
            'text_scores': text_scores,
            'rec_scores': rec_scores,
            'image_width': orig_w,
            'image_height': orig_h,
        }

    # =========================================================================
    # Unified Complete Pipeline: Detection + Embeddings + Faces + OCR
    # =========================================================================

    def infer_unified_complete(
        self, image_bytes: bytes, face_model: str = 'scrfd'
    ) -> dict[str, Any]:
        """
        Unified complete analysis pipeline: YOLO + CLIP + Faces + OCR.

        Single request that returns all analysis results from an image:
        1. YOLO object detection
        2. MobileCLIP global and per-box embeddings
        3. Face detection + ArcFace embeddings (selectable model)
        4. PP-OCRv5 text detection and recognition

        Face Model Selection:
        - "scrfd" (default): SCRFD on person crops only (more efficient)
        - "yolo11": YOLO11-face on full image (may detect more faces)

        Args:
            image_bytes: Raw JPEG/PNG bytes
            face_model: "scrfd" or "yolo11"

        Returns:
            Dict with all analysis results:
            - Detection: num_dets, boxes, scores, classes
            - Embeddings: global_embedding, box_embeddings, normalized_boxes
            - Faces: num_faces, face_boxes, face_landmarks, face_scores, face_embeddings, face_person_idx
            - OCR: num_texts, texts, text_boxes, text_det_scores, text_rec_scores
            - Metadata: face_model_used, orig_shape
        """
        # Fast JPEG header parse for dimensions
        orig_w, orig_h = get_jpeg_dimensions_fast(image_bytes)

        # Get cached affine matrix
        affine_matrix, scale, padding = calculate_affine_matrix(orig_w, orig_h, self.input_size)

        # Prepare inputs
        input_data = np.frombuffer(image_bytes, dtype=np.uint8)
        input_data = np.expand_dims(input_data, axis=0)

        inputs = [
            InferInput('encoded_images', input_data.shape, 'UINT8'),
            InferInput('affine_matrices', [1, 2, 3], 'FP32'),
            InferInput('face_model', [1, 1], 'BYTES'),
        ]
        inputs[0].set_data_from_numpy(input_data)
        inputs[1].set_data_from_numpy(np.expand_dims(affine_matrix, axis=0))
        inputs[2].set_data_from_numpy(np.array([[face_model.encode('utf-8')]], dtype=object))

        # Request all outputs
        outputs = [
            # Detection
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
            # Embeddings
            InferRequestedOutput('global_embeddings'),
            InferRequestedOutput('box_embeddings'),
            InferRequestedOutput('normalized_boxes'),
            # Faces
            InferRequestedOutput('num_faces'),
            InferRequestedOutput('face_embeddings'),
            InferRequestedOutput('face_boxes'),
            InferRequestedOutput('face_landmarks'),
            InferRequestedOutput('face_scores'),
            InferRequestedOutput('face_person_idx'),
            # OCR
            InferRequestedOutput('num_texts'),
            InferRequestedOutput('text_boxes'),
            InferRequestedOutput('text_boxes_normalized'),
            InferRequestedOutput('texts'),
            InferRequestedOutput('text_det_scores'),
            InferRequestedOutput('text_rec_scores'),
            # Metadata
            InferRequestedOutput('face_model_used'),
        ]

        # Sync inference with retry
        response = self._infer_with_retry('unified_complete_pipeline', inputs, outputs)

        # Parse detection outputs (handle batch dimension variations)
        num_dets_raw = response.as_numpy('num_dets')
        num_dets = int(num_dets_raw.flatten()[0])

        det_boxes_raw = response.as_numpy('det_boxes')
        det_scores_raw = response.as_numpy('det_scores')
        det_classes_raw = response.as_numpy('det_classes')

        if det_boxes_raw.ndim == 3:
            boxes = det_boxes_raw[0][:num_dets]
            scores = det_scores_raw[0][:num_dets]
            classes = det_classes_raw[0][:num_dets]
        else:
            boxes = det_boxes_raw[:num_dets]
            scores = det_scores_raw[:num_dets]
            classes = det_classes_raw[:num_dets]

        # Parse embedding outputs
        global_embedding_raw = response.as_numpy('global_embeddings')
        if global_embedding_raw.ndim == 2:
            global_embedding = global_embedding_raw[0]
        else:
            global_embedding = global_embedding_raw

        box_embeddings_raw = response.as_numpy('box_embeddings')
        normalized_boxes_raw = response.as_numpy('normalized_boxes')

        if box_embeddings_raw.ndim == 3:
            box_embeddings = box_embeddings_raw[0][:num_dets]
            normalized_boxes = normalized_boxes_raw[0][:num_dets]
        else:
            box_embeddings = box_embeddings_raw[:num_dets]
            normalized_boxes = normalized_boxes_raw[:num_dets]

        # Parse face outputs
        num_faces_arr = response.as_numpy('num_faces')
        num_faces = int(num_faces_arr.flatten()[0])

        face_boxes_raw = response.as_numpy('face_boxes')
        face_landmarks_raw = response.as_numpy('face_landmarks')
        face_scores_raw = response.as_numpy('face_scores')
        face_person_idx_raw = response.as_numpy('face_person_idx')
        face_emb_raw = response.as_numpy('face_embeddings')

        if face_boxes_raw.ndim == 3:
            face_boxes = face_boxes_raw[0][:num_faces]
            face_landmarks = face_landmarks_raw[0][:num_faces]
            face_scores_arr = face_scores_raw[0][:num_faces]
            face_person_idx = face_person_idx_raw[0][:num_faces]
            face_embeddings = face_emb_raw[0][:num_faces] if num_faces > 0 else np.array([])
        else:
            face_boxes = face_boxes_raw[:num_faces]
            face_landmarks = face_landmarks_raw[:num_faces]
            face_scores_arr = face_scores_raw[:num_faces]
            face_person_idx = face_person_idx_raw[:num_faces]
            face_embeddings = face_emb_raw[:num_faces] if num_faces > 0 else np.array([])

        # Parse OCR outputs
        num_texts_raw = response.as_numpy('num_texts')
        num_texts = int(num_texts_raw.flatten()[0])
        text_boxes_raw = response.as_numpy('text_boxes')
        text_boxes_norm_raw = response.as_numpy('text_boxes_normalized')
        texts_raw = response.as_numpy('texts')
        text_det_scores_raw = response.as_numpy('text_det_scores')
        text_rec_scores_raw = response.as_numpy('text_rec_scores')

        if text_boxes_raw.ndim == 3:
            text_boxes = text_boxes_raw[0][:num_texts]
            text_boxes_norm = text_boxes_norm_raw[0][:num_texts]
            text_det_scores = text_det_scores_raw[0][:num_texts]
            text_rec_scores = text_rec_scores_raw[0][:num_texts]
        else:
            text_boxes = text_boxes_raw[:num_texts]
            text_boxes_norm = text_boxes_norm_raw[:num_texts]
            text_det_scores = text_det_scores_raw[:num_texts]
            text_rec_scores = text_rec_scores_raw[:num_texts]

        # Decode text strings
        texts = []
        if texts_raw.size > 0:
            texts_flat = texts_raw.flatten()[:num_texts]
            for t in texts_flat:
                if isinstance(t, bytes):
                    texts.append(t.decode('utf-8', errors='replace'))
                else:
                    texts.append(str(t))

        # Parse metadata
        face_model_used_raw = response.as_numpy('face_model_used')
        if face_model_used_raw.size > 0:
            fm = face_model_used_raw.flatten()[0]
            if isinstance(fm, bytes):
                face_model_used = fm.decode('utf-8')
            else:
                face_model_used = str(fm)
        else:
            face_model_used = face_model

        return {
            # Detection
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            # Embeddings
            'global_embedding': global_embedding,
            'box_embeddings': box_embeddings,
            'normalized_boxes': normalized_boxes,
            # Faces
            'num_faces': num_faces,
            'face_boxes': face_boxes,
            'face_landmarks': face_landmarks,
            'face_scores': face_scores_arr,
            'face_embeddings': face_embeddings,
            'face_person_idx': face_person_idx,
            # OCR
            'num_texts': num_texts,
            'texts': texts,
            'text_boxes': text_boxes,
            'text_boxes_normalized': text_boxes_norm,
            'text_det_scores': text_det_scores,
            'text_rec_scores': text_rec_scores,
            # Metadata
            'face_model_used': face_model_used,
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

    # =========================================================================
    # YOLO11-Face: Alternative Face Detection Pipeline
    # =========================================================================

    def infer_faces_yolo11(self, image_bytes: bytes, confidence: float = 0.5) -> dict[str, Any]:
        """
        Face detection and recognition using YOLO11-face + ArcFace.

        HIGH-PERFORMANCE: Direct gRPC calls to TensorRT models, bypassing Python BLS.
        2-3x faster than BLS pipeline by eliminating:
        - BLS call serialization overhead
        - Python backend context switching
        - Extra memory copies

        Pipeline:
        1. Decode and letterbox preprocess (CPU - 8ms)
        2. Call YOLO11-face TensorRT directly via gRPC (~7ms)
        3. Crop faces on CPU with MTCNN-style margins (~2ms)
        4. Call ArcFace TensorRT directly via gRPC (~5ms)
        5. L2 normalize embeddings

        Args:
            image_bytes: Raw JPEG/PNG bytes
            confidence: Minimum detection confidence

        Returns:
            Dict with:
                - num_faces: Number of faces detected
                - face_boxes: [N, 4] normalized boxes [x1, y1, x2, y2]
                - face_landmarks: [N, 10] zeros (not extracted in fast mode)
                - face_scores: [N] detection confidence
                - face_embeddings: [N, 512] L2-normalized ArcFace embeddings
                - face_quality: [N] quality scores
                - orig_shape: (height, width)
        """
        # Constants
        YOLO_SIZE = 640
        ARCFACE_SIZE = 112
        FACE_MARGIN = 0.4

        # Decode image - KEEP ORIGINAL HD for quality face cropping
        img_hd = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_hd is None:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img = pil_img.convert('RGB')
            img_hd = np.array(pil_img)[:, :, ::-1]

        orig_h, orig_w = img_hd.shape[:2]

        # For very large images, cap HD size but keep reasonable resolution for cropping
        # This preserves face detail while managing memory
        MAX_HD_DIM = 2048  # Keep HD up to 2048px for quality crops
        if max(orig_h, orig_w) > MAX_HD_DIM:
            hd_scale = MAX_HD_DIM / max(orig_h, orig_w)
            img_hd = cv2.resize(img_hd, (int(orig_w * hd_scale), int(orig_h * hd_scale)))
            orig_h, orig_w = img_hd.shape[:2]

        # Create 640x640 letterbox for YOLO detection (separate from HD image)
        scale = min(YOLO_SIZE / orig_h, YOLO_SIZE / orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (YOLO_SIZE - new_w) / 2
        pad_y = (YOLO_SIZE - new_h) / 2

        resized = cv2.resize(img_hd, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(pad_y), YOLO_SIZE - int(pad_y) - new_h
        left, right = int(pad_x), YOLO_SIZE - int(pad_x) - new_w
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        face_input = padded.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Call YOLO11-face directly (no BLS)
        yolo_inputs = [InferInput('images', [1, 3, YOLO_SIZE, YOLO_SIZE], 'FP32')]
        yolo_inputs[0].set_data_from_numpy(face_input[np.newaxis, ...])
        yolo_outputs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
        ]
        yolo_response = self._infer_with_retry(
            'yolo11_face_small_trt_end2end', yolo_inputs, yolo_outputs
        )

        num_dets = int(yolo_response.as_numpy('num_dets').flatten()[0])
        if num_dets == 0:
            return {
                'num_faces': 0,
                'face_boxes': np.array([]),
                'face_landmarks': np.array([]),
                'face_scores': np.array([]),
                'face_embeddings': np.array([]),
                'face_quality': np.array([]),
                'orig_shape': (orig_h, orig_w),
            }

        det_boxes = yolo_response.as_numpy('det_boxes')[0, :num_dets]  # [N, 4] normalized [0,1]
        det_scores = yolo_response.as_numpy('det_scores')[0, :num_dets]

        # Inverse letterbox: convert to original image coordinates
        boxes_px = det_boxes.copy()
        boxes_px[:, [0, 2]] *= YOLO_SIZE
        boxes_px[:, [1, 3]] *= YOLO_SIZE
        boxes_px[:, [0, 2]] = (boxes_px[:, [0, 2]] - pad_x) / scale
        boxes_px[:, [1, 3]] = (boxes_px[:, [1, 3]] - pad_y) / scale
        boxes_px[:, [0, 2]] = np.clip(boxes_px[:, [0, 2]], 0, orig_w)
        boxes_px[:, [1, 3]] = np.clip(boxes_px[:, [1, 3]], 0, orig_h)

        # Filter by confidence
        mask = det_scores >= confidence
        boxes_px = boxes_px[mask]
        det_scores = det_scores[mask]
        num_faces = len(det_scores)

        if num_faces == 0:
            return {
                'num_faces': 0,
                'face_boxes': np.array([]),
                'face_landmarks': np.array([]),
                'face_scores': np.array([]),
                'face_embeddings': np.array([]),
                'face_quality': np.array([]),
                'orig_shape': (orig_h, orig_w),
            }

        # Crop faces with MTCNN-style margin (CPU - fast numpy)
        crops = []
        for box in boxes_px:
            x1, y1, x2, y2 = box
            face_w, face_h = x2 - x1, y2 - y1
            margin_w, margin_h = face_w * FACE_MARGIN, face_h * FACE_MARGIN

            # Expand and make square
            x1_exp, y1_exp = x1 - margin_w, y1 - margin_h
            x2_exp, y2_exp = x2 + margin_w, y2 + margin_h
            box_w, box_h = x2_exp - x1_exp, y2_exp - y1_exp
            max_dim = max(box_w, box_h)
            cx, cy = (x1_exp + x2_exp) / 2, (y1_exp + y2_exp) / 2
            x1_sq = max(0, int(cx - max_dim / 2))
            y1_sq = max(0, int(cy - max_dim / 2))
            x2_sq = min(orig_w, int(cx + max_dim / 2))
            y2_sq = min(orig_h, int(cy + max_dim / 2))

            # CRITICAL: Crop from HD image, not the 640x640 detection input
            # This preserves facial details for accurate recognition
            face_crop = img_hd[y1_sq:y2_sq, x1_sq:x2_sq]
            if face_crop.size == 0:
                continue
            face_resized = cv2.resize(face_crop, (ARCFACE_SIZE, ARCFACE_SIZE))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_chw = face_rgb.transpose(2, 0, 1).astype(np.float32)
            face_norm = (face_chw - 127.5) / 128.0
            crops.append(face_norm)

        if not crops:
            return {
                'num_faces': 0,
                'face_boxes': np.array([]),
                'face_landmarks': np.array([]),
                'face_scores': np.array([]),
                'face_embeddings': np.array([]),
                'face_quality': np.array([]),
                'orig_shape': (orig_h, orig_w),
            }

        face_crops = np.stack(crops, axis=0)

        # Call ArcFace directly (no BLS)
        arcface_inputs = [InferInput('input', list(face_crops.shape), 'FP32')]
        arcface_inputs[0].set_data_from_numpy(face_crops)
        arcface_outputs = [InferRequestedOutput('output')]
        arcface_response = self._infer_with_retry(
            'arcface_w600k_r50', arcface_inputs, arcface_outputs
        )
        embeddings = arcface_response.as_numpy('output')

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)

        # Compute quality scores
        face_w = boxes_px[:, 2] - boxes_px[:, 0]
        face_h = boxes_px[:, 3] - boxes_px[:, 1]
        face_area = face_w * face_h
        size_score = np.clip((face_area / (orig_h * orig_w)) * 10, 0, 1)
        aspect_ratio = np.minimum(face_w, face_h) / (np.maximum(face_w, face_h) + 1e-6)
        quality = (size_score * aspect_ratio).astype(np.float32)

        # Normalize boxes to [0, 1]
        boxes_norm = boxes_px.copy()
        boxes_norm[:, [0, 2]] /= orig_w
        boxes_norm[:, [1, 3]] /= orig_h

        return {
            'num_faces': num_faces,
            'face_boxes': boxes_norm,
            'face_landmarks': np.zeros((num_faces, 10), dtype=np.float32),
            'face_scores': det_scores,
            'face_embeddings': embeddings,
            'face_quality': quality,
            'orig_shape': (orig_h, orig_w),
        }

    def infer_faces_full_yolo11(
        self, image_bytes: bytes, confidence: float = 0.5
    ) -> dict[str, Any]:
        """
        Full pipeline with YOLO11-face: YOLO detection + MobileCLIP + YOLO11-face + ArcFace.

        Combines visual search capabilities with YOLO11-face detection:
        1. YOLO object detection (parallel with 2, 3)
        2. MobileCLIP global embedding (parallel with 1, 3)
        3. YOLO11-face detection + ArcFace embeddings (parallel with 1, 2)

        This is an alternative to infer_faces_full() which uses SCRFD for face detection.
        YOLO11-face may detect more faces in challenging conditions (occlusion, pose).

        Args:
            image_bytes: Raw JPEG/PNG bytes
            confidence: Minimum face detection confidence (default 0.5)

        Returns:
            Dict with:
                - YOLO detections (num_dets, boxes, scores, classes)
                - MobileCLIP global embedding (512-dim)
                - Face detections (num_faces, face_boxes, face_landmarks, face_scores)
                - Face ArcFace embeddings (128 x 512-dim)
                - Face quality scores
                - Transform params (orig_shape, scale, padding)
        """
        from concurrent.futures import ThreadPoolExecutor

        # Fast JPEG header parse for dimensions
        orig_w, orig_h = get_jpeg_dimensions_fast(image_bytes)

        # Get cached affine matrix
        _affine_matrix, scale, padding = calculate_affine_matrix(orig_w, orig_h, self.input_size)

        # Run YOLO + MobileCLIP and YOLO11-face in parallel
        def run_yolo_clip():
            return self.infer_yolo_clip(image_bytes, full_pipeline=False)

        def run_yolo11_face():
            return self.infer_faces_yolo11(image_bytes, confidence=confidence)

        with ThreadPoolExecutor(max_workers=2) as executor:
            yolo_clip_future = executor.submit(run_yolo_clip)
            face_future = executor.submit(run_yolo11_face)

            yolo_clip_result = yolo_clip_future.result()
            face_result = face_future.result()

        # Combine results
        return {
            # YOLO detections
            'num_dets': yolo_clip_result['num_dets'],
            'boxes': yolo_clip_result['boxes'],
            'scores': yolo_clip_result['scores'],
            'classes': yolo_clip_result['classes'],
            # MobileCLIP global embedding
            'image_embedding': yolo_clip_result['image_embedding'],
            # Face detections from YOLO11-face pipeline
            'num_faces': face_result['num_faces'],
            'face_boxes': face_result['face_boxes'],
            'face_landmarks': face_result['face_landmarks'],
            'face_scores': face_result['face_scores'],
            'face_embeddings': face_result['face_embeddings'],
            'face_quality': face_result['face_quality'],
            # Transform params
            'orig_shape': (orig_h, orig_w),
            'scale': scale,
            'padding': padding,
        }

    # =========================================================================
    # Unified Complete Pipeline: CPU Preprocessing Variants
    # =========================================================================

    def infer_unified_complete_cpu(
        self, image_bytes: bytes, face_model: str = 'scrfd'
    ) -> dict[str, Any]:
        """
        Unified complete analysis pipeline with CPU preprocessing.

        Same as infer_unified_complete but uses CPU preprocessing instead of DALI.
        Calls 'unified_complete_pipeline_direct' which expects pre-processed tensors.

        Pipeline:
        1. CPU JPEG decode (PIL/cv2)
        2. CPU preprocessing for YOLO (640x640 letterbox)
        3. CPU preprocessing for CLIP (256x256 center crop)
        4. CPU preprocessing for HD original (variable size)
        5. CPU affine matrix calculation
        6. Direct TRT inference via unified_complete_pipeline_direct

        Args:
            image_bytes: Raw JPEG/PNG bytes
            face_model: "scrfd" or "yolo11"

        Returns:
            Dict with all analysis results (same format as infer_unified_complete)
        """
        from src.services.cpu_preprocess import preprocess_single

        # CPU preprocessing
        prep = preprocess_single(image_bytes)

        # Use the tensor-based method
        return self.infer_unified_complete_from_tensors(prep, face_model=face_model)

    def infer_unified_complete_from_tensors(
        self, prep: 'PreprocessResult', face_model: str = 'scrfd'
    ) -> dict[str, Any]:
        """
        Unified complete analysis pipeline from pre-processed tensors.

        For batch processing where preprocessing is done separately (e.g., in parallel).
        Calls 'unified_complete_pipeline_direct' model directly.

        Args:
            prep: PreprocessResult with yolo_image, clip_image, original_image, affine_matrix, etc.
            face_model: "scrfd" or "yolo11"

        Returns:
            Dict with all analysis results (same format as infer_unified_complete)
        """
        import time as _time

        _t_start = _time.perf_counter()

        # Build inputs from preprocessed tensors
        # yolo_images: [1, 3, 640, 640], FP32
        yolo_input = InferInput('yolo_images', [1, 3, 640, 640], 'FP32')
        yolo_input.set_data_from_numpy(prep.yolo_image)

        # clip_images: [1, 3, 256, 256], FP32
        clip_input = InferInput('clip_images', [1, 3, 256, 256], 'FP32')
        clip_input.set_data_from_numpy(prep.clip_image)

        # original_images: [1, 3, H, W], FP32 (variable size)
        orig_shape = prep.original_image.shape  # [1, 3, H, W]
        original_input = InferInput('original_images', list(orig_shape), 'FP32')
        original_input.set_data_from_numpy(prep.original_image)

        # affine_matrices: [1, 2, 3], FP32
        affine_input = InferInput('affine_matrices', [1, 2, 3], 'FP32')
        affine_input.set_data_from_numpy(prep.affine)

        # face_model: [1, 1], BYTES
        face_model_input = InferInput('face_model', [1, 1], 'BYTES')
        face_model_input.set_data_from_numpy(np.array([[face_model.encode('utf-8')]], dtype=object))

        inputs = [yolo_input, clip_input, original_input, affine_input, face_model_input]

        _t_input_prep = _time.perf_counter()

        # Request all outputs (same as infer_unified_complete)
        outputs = [
            # Detection
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
            # Embeddings
            InferRequestedOutput('global_embeddings'),
            InferRequestedOutput('box_embeddings'),
            InferRequestedOutput('normalized_boxes'),
            # Faces
            InferRequestedOutput('num_faces'),
            InferRequestedOutput('face_embeddings'),
            InferRequestedOutput('face_boxes'),
            InferRequestedOutput('face_landmarks'),
            InferRequestedOutput('face_scores'),
            InferRequestedOutput('face_person_idx'),
            # OCR
            InferRequestedOutput('num_texts'),
            InferRequestedOutput('text_boxes'),
            InferRequestedOutput('text_boxes_normalized'),
            InferRequestedOutput('texts'),
            InferRequestedOutput('text_det_scores'),
            InferRequestedOutput('text_rec_scores'),
            # Metadata
            InferRequestedOutput('face_model_used'),
        ]

        # Sync inference with retry
        response = self._infer_with_retry('unified_complete_pipeline', inputs, outputs)

        _t_triton_done = _time.perf_counter()
        _input_prep_ms = (_t_input_prep - _t_start) * 1000
        _triton_ms = (_t_triton_done - _t_input_prep) * 1000
        logger.debug(
            f'[PROFILE] Input prep: {_input_prep_ms:.1f}ms, Triton call: {_triton_ms:.1f}ms'
        )

        # Parse detection outputs (handle batch dimension variations)
        num_dets_raw = response.as_numpy('num_dets')
        num_dets = int(num_dets_raw.flatten()[0])

        det_boxes_raw = response.as_numpy('det_boxes')
        det_scores_raw = response.as_numpy('det_scores')
        det_classes_raw = response.as_numpy('det_classes')

        if det_boxes_raw.ndim == 3:
            boxes = det_boxes_raw[0][:num_dets]
            scores = det_scores_raw[0][:num_dets]
            classes = det_classes_raw[0][:num_dets]
        else:
            boxes = det_boxes_raw[:num_dets]
            scores = det_scores_raw[:num_dets]
            classes = det_classes_raw[:num_dets]

        # Parse embedding outputs
        global_embedding_raw = response.as_numpy('global_embeddings')
        if global_embedding_raw.ndim == 2:
            global_embedding = global_embedding_raw[0]
        else:
            global_embedding = global_embedding_raw

        box_embeddings_raw = response.as_numpy('box_embeddings')
        normalized_boxes_raw = response.as_numpy('normalized_boxes')

        if box_embeddings_raw.ndim == 3:
            box_embeddings = box_embeddings_raw[0][:num_dets]
            normalized_boxes = normalized_boxes_raw[0][:num_dets]
        else:
            box_embeddings = box_embeddings_raw[:num_dets]
            normalized_boxes = normalized_boxes_raw[:num_dets]

        # Parse face outputs
        num_faces_arr = response.as_numpy('num_faces')
        num_faces = int(num_faces_arr.flatten()[0])

        face_boxes_raw = response.as_numpy('face_boxes')
        face_landmarks_raw = response.as_numpy('face_landmarks')
        face_scores_raw = response.as_numpy('face_scores')
        face_person_idx_raw = response.as_numpy('face_person_idx')
        face_emb_raw = response.as_numpy('face_embeddings')

        if face_boxes_raw.ndim == 3:
            face_boxes = face_boxes_raw[0][:num_faces]
            face_landmarks = face_landmarks_raw[0][:num_faces]
            face_scores_arr = face_scores_raw[0][:num_faces]
            face_person_idx = face_person_idx_raw[0][:num_faces]
            face_embeddings = face_emb_raw[0][:num_faces] if num_faces > 0 else np.array([])
        else:
            face_boxes = face_boxes_raw[:num_faces]
            face_landmarks = face_landmarks_raw[:num_faces]
            face_scores_arr = face_scores_raw[:num_faces]
            face_person_idx = face_person_idx_raw[:num_faces]
            face_embeddings = face_emb_raw[:num_faces] if num_faces > 0 else np.array([])

        # Parse OCR outputs
        num_texts_raw = response.as_numpy('num_texts')
        num_texts = int(num_texts_raw.flatten()[0])
        text_boxes_raw = response.as_numpy('text_boxes')
        text_boxes_norm_raw = response.as_numpy('text_boxes_normalized')
        texts_raw = response.as_numpy('texts')
        text_det_scores_raw = response.as_numpy('text_det_scores')
        text_rec_scores_raw = response.as_numpy('text_rec_scores')

        if text_boxes_raw.ndim == 3:
            text_boxes = text_boxes_raw[0][:num_texts]
            text_boxes_norm = text_boxes_norm_raw[0][:num_texts]
            text_det_scores = text_det_scores_raw[0][:num_texts]
            text_rec_scores = text_rec_scores_raw[0][:num_texts]
        else:
            text_boxes = text_boxes_raw[:num_texts]
            text_boxes_norm = text_boxes_norm_raw[:num_texts]
            text_det_scores = text_det_scores_raw[:num_texts]
            text_rec_scores = text_rec_scores_raw[:num_texts]

        # Decode text strings
        texts = []
        if texts_raw.size > 0:
            texts_flat = texts_raw.flatten()[:num_texts]
            for t in texts_flat:
                if isinstance(t, bytes):
                    texts.append(t.decode('utf-8', errors='replace'))
                else:
                    texts.append(str(t))

        # Parse metadata
        face_model_used_raw = response.as_numpy('face_model_used')
        if face_model_used_raw.size > 0:
            fm = face_model_used_raw.flatten()[0]
            if isinstance(fm, bytes):
                face_model_used = fm.decode('utf-8')
            else:
                face_model_used = str(fm)
        else:
            face_model_used = face_model

        # Get original shape from prep object
        orig_h, orig_w = prep.orig_h, prep.orig_w

        return {
            # Detection
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            # Embeddings
            'global_embedding': global_embedding,
            'box_embeddings': box_embeddings,
            'normalized_boxes': normalized_boxes,
            # Faces
            'num_faces': num_faces,
            'face_boxes': face_boxes,
            'face_landmarks': face_landmarks,
            'face_scores': face_scores_arr,
            'face_embeddings': face_embeddings,
            'face_person_idx': face_person_idx,
            # OCR
            'num_texts': num_texts,
            'texts': texts,
            'text_boxes': text_boxes,
            'text_boxes_normalized': text_boxes_norm,
            'text_det_scores': text_det_scores,
            'text_rec_scores': text_rec_scores,
            # Metadata
            'face_model_used': face_model_used,
            'orig_shape': (orig_h, orig_w),
            'scale': prep.scale,
            'padding': prep.padding,
        }

    def infer_unified_direct_ensemble(self, prep: 'PreprocessResult') -> dict[str, Any]:
        """
        Call unified_direct_ensemble with preprocessed tensors.

        This uses Triton's native ensemble scheduling for parallel YOLO+CLIP execution,
        which is faster than sequential BLS calls in unified_complete_pipeline_direct.

        NOTE: This ensemble does NOT include OCR. Use infer_unified_complete_from_tensors
        if OCR is needed.

        Args:
            prep: PreprocessResult with yolo_image, clip_image, original_image, affine

        Returns:
            Dict with detection, embedding, and face results (no OCR)
        """
        # Build inputs from preprocessed tensors
        yolo_input = InferInput('yolo_images', [1, 3, 640, 640], 'FP32')
        yolo_input.set_data_from_numpy(prep.yolo_image)

        clip_input = InferInput('clip_images', [1, 3, 256, 256], 'FP32')
        clip_input.set_data_from_numpy(prep.clip_image)

        orig_shape = prep.original_image.shape
        original_input = InferInput('original_images', list(orig_shape), 'FP32')
        original_input.set_data_from_numpy(prep.original_image)

        affine_input = InferInput('affine_matrices', [1, 2, 3], 'FP32')
        affine_input.set_data_from_numpy(prep.affine)

        inputs = [yolo_input, clip_input, original_input, affine_input]

        outputs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
            InferRequestedOutput('global_embeddings'),
            InferRequestedOutput('box_embeddings'),
            InferRequestedOutput('normalized_boxes'),
            InferRequestedOutput('num_faces'),
            InferRequestedOutput('face_embeddings'),
            InferRequestedOutput('face_boxes'),
            InferRequestedOutput('face_landmarks'),
            InferRequestedOutput('face_scores'),
            InferRequestedOutput('face_person_idx'),
        ]

        response = self._infer_with_retry('unified_direct_ensemble', inputs, outputs)

        # Parse detection outputs
        num_dets_raw = response.as_numpy('num_dets')
        num_dets = int(num_dets_raw.flatten()[0])

        det_boxes_raw = response.as_numpy('det_boxes')
        det_scores_raw = response.as_numpy('det_scores')
        det_classes_raw = response.as_numpy('det_classes')

        if det_boxes_raw.ndim == 3:
            boxes = det_boxes_raw[0][:num_dets]
            scores = det_scores_raw[0][:num_dets]
            classes = det_classes_raw[0][:num_dets]
        else:
            boxes = det_boxes_raw[:num_dets]
            scores = det_scores_raw[:num_dets]
            classes = det_classes_raw[:num_dets]

        # Parse embedding outputs
        global_embedding_raw = response.as_numpy('global_embeddings')
        global_embedding = (
            global_embedding_raw[0] if global_embedding_raw.ndim == 2 else global_embedding_raw
        )

        box_embeddings_raw = response.as_numpy('box_embeddings')
        normalized_boxes_raw = response.as_numpy('normalized_boxes')

        if box_embeddings_raw.ndim == 3:
            box_embeddings = box_embeddings_raw[0][:num_dets]
            normalized_boxes = normalized_boxes_raw[0][:num_dets]
        else:
            box_embeddings = box_embeddings_raw[:num_dets]
            normalized_boxes = normalized_boxes_raw[:num_dets]

        # Parse face outputs
        num_faces_arr = response.as_numpy('num_faces')
        num_faces = int(num_faces_arr.flatten()[0])

        face_boxes_raw = response.as_numpy('face_boxes')
        face_landmarks_raw = response.as_numpy('face_landmarks')
        face_scores_raw = response.as_numpy('face_scores')
        face_person_idx_raw = response.as_numpy('face_person_idx')
        face_emb_raw = response.as_numpy('face_embeddings')

        if face_boxes_raw.ndim == 3:
            face_boxes = face_boxes_raw[0][:num_faces]
            face_landmarks = face_landmarks_raw[0][:num_faces]
            face_scores_arr = face_scores_raw[0][:num_faces]
            face_person_idx = face_person_idx_raw[0][:num_faces]
            face_embeddings = face_emb_raw[0][:num_faces] if num_faces > 0 else np.array([])
        else:
            face_boxes = face_boxes_raw[:num_faces]
            face_landmarks = face_landmarks_raw[:num_faces]
            face_scores_arr = face_scores_raw[:num_faces]
            face_person_idx = face_person_idx_raw[:num_faces]
            face_embeddings = face_emb_raw[:num_faces] if num_faces > 0 else np.array([])

        orig_h, orig_w = prep.orig_h, prep.orig_w

        return {
            'num_dets': num_dets,
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'global_embedding': global_embedding,
            'box_embeddings': box_embeddings,
            'normalized_boxes': normalized_boxes,
            'num_faces': num_faces,
            'face_boxes': face_boxes,
            'face_landmarks': face_landmarks,
            'face_scores': face_scores_arr,
            'face_embeddings': face_embeddings,
            'face_person_idx': face_person_idx,
            'orig_shape': (orig_h, orig_w),
            'scale': prep.scale,
            'padding': prep.padding,
        }

    # =========================================================================
    @staticmethod
    def format_detections(result: dict[str, Any]) -> list:
        """
        Format detections with coordinates normalized to original image dimensions.

        Uses shared utility from affine.py to apply inverse letterbox transformation.
        This matches PyTorch boxes.xyxyn coordinate output.

        Args:
            result: Inference result with boxes, scores, classes,
                   and optionally orig_shape, scale, padding for inverse transform

        Returns:
            List of detection dicts with x1, y1, x2, y2 normalized to original image
        """
        return format_detections_from_triton(result, input_size=640)

    # =========================================================================
    # Optimized Batched Inference Methods (Bypass Python BLS)
    # =========================================================================
    # These methods enable cross-image batching for maximum throughput.
    # Instead of processing crops per-image through Python BLS, we:
    # 1. Collect ALL crops from ALL images in a batch
    # 2. Send ONE batched request to each TRT model
    # 3. Unpack results back to per-image structure
    #
    # This follows Google/Meta best practices:
    # - Maximize GPU utilization with large batches
    # - Minimize Triton BLS call overhead
    # - Enable parallel CPU preprocessing

    def infer_mobileclip_batch(
        self,
        crops: np.ndarray,
        max_batch_size: int = 64,
    ) -> np.ndarray:
        """
        Batched MobileCLIP inference for multiple image crops.

        Optimized for high throughput:
        - Accepts pre-cropped, pre-normalized tensors
        - Handles batches larger than model max by chunking
        - Returns embeddings in same order as input

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

        # Process in chunks to respect model max_batch_size
        for i in range(0, n_crops, max_batch_size):
            batch = crops[i : i + max_batch_size]
            batch_size = batch.shape[0]

            input_tensor = InferInput('images', [batch_size, 3, 256, 256], 'FP32')
            input_tensor.set_data_from_numpy(batch.astype(np.float32))

            output = InferRequestedOutput('image_embeddings')

            response = self._infer_with_retry(
                'mobileclip2_s2_image_encoder', [input_tensor], [output]
            )

            embeddings = response.as_numpy('image_embeddings')
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 512), dtype=np.float32)

    def infer_yolo_face_batch(
        self,
        crops: np.ndarray,
        max_batch_size: int = 64,
        conf_threshold: float = 0.5,
    ) -> list[dict]:
        """
        Batched YOLO11-Face inference for person crop images.

        Optimized for high throughput:
        - Accepts pre-cropped person regions (640x640)
        - Returns face detections per input crop
        - Uses TRT End2End model with GPU NMS

        Args:
            crops: [N, 3, 640, 640] FP32 normalized tensor
            max_batch_size: Maximum batch size per Triton request
            conf_threshold: Minimum confidence for face detections

        Returns:
            List of N dicts, each with:
            - num_faces: int
            - boxes: [K, 4] normalized face boxes
            - scores: [K] confidence scores
        """
        if crops.shape[0] == 0:
            return []

        n_crops = crops.shape[0]
        all_results = []

        # Process in chunks
        for i in range(0, n_crops, max_batch_size):
            batch = crops[i : i + max_batch_size]
            batch_size = batch.shape[0]

            input_tensor = InferInput('images', [batch_size, 3, 640, 640], 'FP32')
            input_tensor.set_data_from_numpy(batch.astype(np.float32))

            outputs = [
                InferRequestedOutput('num_dets'),
                InferRequestedOutput('det_boxes'),
                InferRequestedOutput('det_scores'),
                InferRequestedOutput('det_classes'),
            ]

            response = self._infer_with_retry(
                'yolo11_face_small_trt_end2end', [input_tensor], outputs
            )

            num_dets = response.as_numpy('num_dets')  # [B, 1]
            det_boxes = response.as_numpy('det_boxes')  # [B, 100, 4]
            det_scores = response.as_numpy('det_scores')  # [B, 100]

            # Parse per-crop results
            for j in range(batch_size):
                n_det = int(num_dets[j, 0])
                boxes = det_boxes[j, :n_det]
                scores = det_scores[j, :n_det]

                # Filter by confidence
                mask = scores >= conf_threshold
                all_results.append(
                    {
                        'num_faces': int(mask.sum()),
                        'boxes': boxes[mask],
                        'scores': scores[mask],
                    }
                )

        return all_results

    def infer_arcface_batch(
        self,
        faces: np.ndarray,
        max_batch_size: int = 128,
    ) -> np.ndarray:
        """
        Batched ArcFace inference for face embeddings.

        Optimized for high throughput:
        - Accepts pre-cropped, pre-normalized face tensors (112x112)
        - Handles batches larger than model max by chunking
        - Returns 512-dim L2-normalized embeddings

        Args:
            faces: [N, 3, 112, 112] FP32 tensor, range [-1, 1]
            max_batch_size: Maximum batch size per Triton request

        Returns:
            [N, 512] FP32 L2-normalized embeddings
        """
        if faces.shape[0] == 0:
            return np.empty((0, 512), dtype=np.float32)

        n_faces = faces.shape[0]
        all_embeddings = []

        # Process in chunks
        for i in range(0, n_faces, max_batch_size):
            batch = faces[i : i + max_batch_size]
            batch_size = batch.shape[0]

            input_tensor = InferInput('input', [batch_size, 3, 112, 112], 'FP32')
            input_tensor.set_data_from_numpy(batch.astype(np.float32))

            output = InferRequestedOutput('output')

            response = self._infer_with_retry('arcface_w600k_r50', [input_tensor], [output])

            embeddings = response.as_numpy('output')
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 512), dtype=np.float32)

    def infer_yolo_batch(
        self,
        images: np.ndarray,
        max_batch_size: int = 64,
    ) -> list[dict]:
        """
        Batched YOLO inference for object detection.

        Args:
            images: [N, 3, 640, 640] FP32 normalized tensor
            max_batch_size: Maximum batch size per Triton request

        Returns:
            List of N dicts with detections per image
        """
        if images.shape[0] == 0:
            return []

        n_images = images.shape[0]
        all_results = []

        for i in range(0, n_images, max_batch_size):
            batch = images[i : i + max_batch_size]
            batch_size = batch.shape[0]

            input_tensor = InferInput('images', [batch_size, 3, 640, 640], 'FP32')
            input_tensor.set_data_from_numpy(batch.astype(np.float32))

            outputs = [
                InferRequestedOutput('num_dets'),
                InferRequestedOutput('det_boxes'),
                InferRequestedOutput('det_scores'),
                InferRequestedOutput('det_classes'),
            ]

            response = self._infer_with_retry('yolov11_small_trt_end2end', [input_tensor], outputs)

            num_dets = response.as_numpy('num_dets')
            det_boxes = response.as_numpy('det_boxes')
            det_scores = response.as_numpy('det_scores')
            det_classes = response.as_numpy('det_classes')

            for j in range(batch_size):
                n_det = int(num_dets[j, 0])
                all_results.append(
                    {
                        'num_dets': n_det,
                        'boxes': det_boxes[j, :n_det],
                        'scores': det_scores[j, :n_det],
                        'classes': det_classes[j, :n_det],
                    }
                )

        return all_results


# =============================================================================
# Singleton Instance (for shared client across requests)
# =============================================================================
# Global singleton for shared Triton client instance
# Justification: Required for connection pooling and efficient resource usage
# - Single gRPC connection shared across all requests enables Triton's dynamic batching
# - Thread-safe sync client prevents deadlock at high concurrency (256+ clients)
# - Alternative (per-request clients) would exhaust connections and disable batching
_client_instance: TritonClient | None = None


def get_triton_client(triton_url: str = 'triton-api:8001') -> TritonClient:
    """
    Get singleton Triton client instance.

    Uses shared sync gRPC connection for proper backpressure handling.
    Thread-safe because gRPC sync client handles concurrent requests internally.

    Returns:
        Shared TritonClient instance
    """
    global _client_instance  # noqa: PLW0603 - Singleton pattern documented above
    if _client_instance is None:
        _client_instance = TritonClient(triton_url)
    return _client_instance
