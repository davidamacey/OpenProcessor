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
from typing import Any

import cv2
import numpy as np
from PIL import Image
from tritonclient.grpc import InferInput, InferRequestedOutput
from ultralytics.data.augment import LetterBox

from src.clients.triton_pool import TritonClientManager
from src.utils.affine import format_detections_from_triton
from src.utils.retry import retry_sync


logger = logging.getLogger(__name__)


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
        orig_h, orig_w = image_array.shape[:2]

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # Ultralytics LetterBox for exact preprocessing match
        letterbox = LetterBox(
            new_shape=(self.input_size, self.input_size), auto=False, scaleup=False
        )
        img_letterbox = letterbox(image=image_rgb)

        # Calculate transformation parameters
        scale = min(self.input_size / orig_h, self.input_size / orig_w)
        scale = min(scale, 1.0)

        new_unpad_w = round(orig_w * scale)
        new_unpad_h = round(orig_h * scale)
        pad_w = (self.input_size - new_unpad_w) / 2.0
        pad_h = (self.input_size - new_unpad_h) / 2.0
        padding = (pad_w, pad_h)

        # Normalize, HWC->CHW, add batch dim
        img_norm = img_letterbox.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))
        input_data = np.expand_dims(img_chw, axis=0)

        inputs = [InferInput('images', input_data.shape, 'FP32')]
        inputs[0].set_data_from_numpy(input_data)

        outputs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
        ]

        response = self._infer_with_retry(model_name, inputs, outputs)

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

        Args:
            images: List of images (HWC, BGR, 0-255) from cv2.imdecode
            model_name: Triton model name (e.g., "yolov11_small_trt_end2end")

        Returns:
            List of dicts with num_dets, boxes, scores, classes, orig_shape, scale, padding
        """
        letterbox = LetterBox(
            new_shape=(self.input_size, self.input_size), auto=False, scaleup=False
        )

        orig_shapes = []
        scales = []
        paddings = []
        preprocessed = []

        for img in images:
            orig_h, orig_w = img.shape[:2]
            orig_shapes.append((orig_h, orig_w))

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_letterbox = letterbox(image=img_rgb)

            scale = min(self.input_size / orig_h, self.input_size / orig_w)
            scale = min(scale, 1.0)
            scales.append(scale)

            new_unpad_w = round(orig_w * scale)
            new_unpad_h = round(orig_h * scale)
            pad_w = (self.input_size - new_unpad_w) / 2.0
            pad_h = (self.input_size - new_unpad_h) / 2.0
            paddings.append((pad_w, pad_h))

            img_norm = img_letterbox.astype(np.float32) / 255.0
            img_chw = np.transpose(img_norm, (2, 0, 1))
            preprocessed.append(img_chw)

        input_data = np.stack(preprocessed, axis=0)
        batch_size = input_data.shape[0]

        inputs = [InferInput('images', input_data.shape, 'FP32')]
        inputs[0].set_data_from_numpy(input_data)

        outputs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
        ]

        response = self._infer_with_retry(model_name, inputs, outputs)

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

        # YOLO preprocessing (CPU letterbox)
        yolo_input, scale, padding = self._preprocess_yolo_cpu(img_array)

        # CLIP preprocessing (CPU resize/crop)
        clip_input = self._preprocess_clip_cpu(img_array)

        # Run YOLO TRT inference
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

        # Run CLIP TRT inference
        clip_inputs = [InferInput('images', clip_input.shape, 'FP32')]
        clip_inputs[0].set_data_from_numpy(clip_input)
        clip_outputs = [InferRequestedOutput('image_embeddings')]
        clip_response = self._infer_with_retry(
            'mobileclip2_s2_image_encoder', clip_inputs, clip_outputs
        )

        # Parse outputs
        num_dets = int(yolo_response.as_numpy('num_dets')[0][0])
        boxes = yolo_response.as_numpy('det_boxes')[0][:num_dets]
        scores = yolo_response.as_numpy('det_scores')[0][:num_dets]
        classes = yolo_response.as_numpy('det_classes')[0][:num_dets]
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
        """CPU letterbox preprocessing for YOLO."""
        orig_h, orig_w = img_array.shape[:2]
        target_size = self.input_size  # 640

        scale = min(target_size / orig_h, target_size / orig_w)
        scale = min(scale, 1.0)

        new_w = round(orig_w * scale)
        new_h = round(orig_h * scale)

        if scale < 1.0:
            resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = img_array

        pad_w = (target_size - new_w) / 2.0
        pad_h = (target_size - new_h) / 2.0
        top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
        left, right = round(pad_w - 0.1), round(pad_w + 0.1)

        letterboxed = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        if letterboxed.shape[:2] != (target_size, target_size):
            letterboxed = cv2.resize(letterboxed, (target_size, target_size))

        normalized = letterboxed.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)

        return batched, scale, (pad_w, pad_h)

    def _preprocess_clip_cpu(self, img_array: np.ndarray) -> np.ndarray:
        """CPU preprocessing for MobileCLIP (256x256, center crop)."""
        orig_h, orig_w = img_array.shape[:2]
        target_size = 256

        scale = target_size / min(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        start_x = (new_w - target_size) // 2
        start_y = (new_h - target_size) // 2
        cropped = resized[start_y : start_y + target_size, start_x : start_x + target_size]

        normalized = cropped.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0)

    # =========================================================================
    # Individual Model Inference (MobileCLIP Components)
    # =========================================================================

    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        """Encode image to 512-dim embedding via MobileCLIP."""
        img_array = self._preprocess_for_mobileclip(image_bytes)

        input_tensor = InferInput('images', [1, 3, 256, 256], 'FP32')
        input_tensor.set_data_from_numpy(img_array)

        output = InferRequestedOutput('image_embeddings')
        response = self._infer_with_retry('mobileclip2_s2_image_encoder', [input_tensor], [output])

        return response.as_numpy('image_embeddings')[0]

    def encode_text(self, tokens: np.ndarray) -> np.ndarray:
        """Encode tokenized text to 512-dim embedding via MobileCLIP."""
        input_tensor = InferInput('text_tokens', [1, 77], 'INT64')
        input_tensor.set_data_from_numpy(tokens.astype(np.int64))

        output = InferRequestedOutput('text_embeddings')
        response = self._infer_with_retry('mobileclip2_s2_text_encoder', [input_tensor], [output])

        return response.as_numpy('text_embeddings')[0]

    def _preprocess_for_mobileclip(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for MobileCLIP (256x256, BILINEAR, center crop)."""
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        width, height = img.size
        scale = 256 / min(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

        left = (new_width - 256) // 2
        top = (new_height - 256) // 2
        img = img.crop((left, top, left + 256, top + 256))

        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))

        return img_array[np.newaxis, ...]

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
        import cv2

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
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            alpha = img_array[:, :, 3:4] / 255.0
            rgb = img_array[:, :, :3]
            white_bg = np.ones_like(rgb) * 255
            img_array = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)

        orig_h, orig_w = img_array.shape[:2]

        # PP-OCR approach: scale to fit within limit, pad to 32-boundary
        ocr_limit_side = 960
        ocr_min_side = 640

        max_side = max(orig_h, orig_w)
        if max_side > ocr_limit_side:
            ratio = ocr_limit_side / max_side
        elif max_side < ocr_min_side:
            ratio = ocr_min_side / max_side
        else:
            ratio = 1.0

        resize_h = max(32, int(orig_h * ratio))
        resize_w = max(32, int(orig_w * ratio))

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
        ocr_input = ocr_input.transpose(2, 0, 1)

        # Original image for cropping
        orig_normalized = img_array.astype(np.float32) / 255.0
        orig_normalized = orig_normalized.transpose(2, 0, 1)

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

        num_texts_raw = response.as_numpy('num_texts')
        logger.info(
            f'OCR response: num_texts_raw shape={num_texts_raw.shape}, value={num_texts_raw}'
        )
        num_texts = int(num_texts_raw[0])
        text_boxes = response.as_numpy('text_boxes')[:num_texts]
        text_boxes_norm = response.as_numpy('text_boxes_normalized')[:num_texts]
        text_scores = response.as_numpy('text_scores')[:num_texts]
        rec_scores = response.as_numpy('rec_scores')[:num_texts]

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
    # Formatting Utilities
    # =========================================================================

    @staticmethod
    def format_detections(result: dict[str, Any]) -> list:
        """Format detections with coordinates normalized to original image dimensions."""
        return format_detections_from_triton(result, input_size=640)

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
                'mobileclip2_s2_image_encoder', [input_tensor], [output]
            )

            embeddings = response.as_numpy('image_embeddings')
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 512), dtype=np.float32)

    def infer_arcface_batch(
        self,
        faces: np.ndarray,
        max_batch_size: int = 128,
    ) -> np.ndarray:
        """
        Batched ArcFace inference for face embeddings.

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
# Singleton Instance
# =============================================================================
_client_instance: TritonClient | None = None


def get_triton_client(triton_url: str = 'triton-server:8001') -> TritonClient:
    """Get singleton Triton client instance."""
    global _client_instance  # noqa: PLW0603
    if _client_instance is None:
        _client_instance = TritonClient(triton_url)
    return _client_instance
