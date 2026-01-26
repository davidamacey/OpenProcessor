"""
Parallel Ingest Pipeline - High-Performance Async Implementation.

Architecture:
    Request with N images
        |
        v
    asyncio.gather() for concurrent processing
        |
        v (parallel per image)
    +---------------------------------------------+
    | Worker 1: Image 1                           |
    |   - CPU preprocess (via ThreadPoolExecutor) |
    |   - Async Triton YOLO -> detections         |
    |   - Async Triton CLIP -> global embedding   |
    |   - CPU crop boxes                          |
    |   - Async Triton CLIP -> box embeddings     |
    |   - CPU crop persons                        |
    |   - Async Triton YOLO-Face -> face boxes    |
    |   - CPU crop faces                          |
    |   - Async Triton ArcFace -> face embeddings |
    |   - Async Triton OCR -> text                |
    |   - Return results                          |
    +---------------------------------------------+
        |
        v
    AsyncTritonPool handles all GPU calls with:
      - 4 gRPC channels (true parallelism)
      - Semaphore backpressure (max 64 concurrent)
      - Round-robin client selection
        |
        v
    Collect results -> Index to OpenSearch -> Return

Performance:
- True async Triton calls (no thread wrapping)
- Connection pooling with separate TCP connections
- Buffer pooling for zero-allocation hot path
- Backpressure prevents server overload
- OCR integration for text extraction
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from tritonclient.grpc.aio import InferInput, InferRequestedOutput

from src.clients.triton_pool import AsyncTritonPool
from src.services.buffer_pool import (
    ARCFACE_BATCH_POOL,
    CLIP_BATCH_POOL,
    CLIP_BUFFER_POOL,
    YOLO_BATCH_POOL,
    YOLO_BUFFER_POOL,
)


logger = logging.getLogger(__name__)

# Model input sizes
YOLO_SIZE = 640
CLIP_SIZE = 256
ARCFACE_SIZE = 112
OCR_SIZE = 960
PERSON_CLASS_ID = 0


@dataclass
class IngestResult:
    """Result for a single ingested image."""

    image_id: str
    image_path: str | None

    # Original image info
    orig_width: int
    orig_height: int

    # Detections
    num_dets: int = 0
    boxes: np.ndarray | None = None  # [N, 4] normalized
    scores: np.ndarray | None = None  # [N]
    classes: np.ndarray | None = None  # [N]

    # Embeddings
    global_embedding: np.ndarray | None = None  # [512]
    box_embeddings: np.ndarray | None = None  # [N, 512]

    # Faces
    num_faces: int = 0
    face_boxes: np.ndarray | None = None  # [M, 4] normalized
    face_scores: np.ndarray | None = None  # [M]
    face_embeddings: np.ndarray | None = None  # [M, 512]
    face_person_idx: np.ndarray | None = None  # [M]

    # OCR
    num_ocr: int = 0
    ocr_texts: list[str] | None = None
    ocr_boxes: np.ndarray | None = None  # [K, 4] normalized
    ocr_scores: np.ndarray | None = None  # [K]

    # Timing
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0

    # Error
    error: str | None = None


def _cpu_preprocess(
    image_bytes: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """
    CPU preprocessing: decode + resize for YOLO and CLIP.

    Returns:
        yolo_tensor: [1, 3, 640, 640] FP32 normalized
        clip_tensor: [1, 3, 256, 256] FP32 normalized
        original_bgr: [H, W, 3] uint8 BGR for cropping
        orig_shape: (height, width)
    """
    # Decode JPEG/PNG
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError('Failed to decode image')

    orig_h, orig_w = img.shape[:2]

    # YOLO letterbox (640x640)
    scale = min(YOLO_SIZE / orig_h, YOLO_SIZE / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = (YOLO_SIZE - new_h) // 2
    pad_w = (YOLO_SIZE - new_w) // 2
    yolo_img = np.full((YOLO_SIZE, YOLO_SIZE, 3), 114, dtype=np.uint8)
    yolo_img[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    # Convert to tensor: HWC BGR -> CHW RGB, normalize to [0, 1]
    yolo_rgb = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)
    yolo_tensor = yolo_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    yolo_tensor = np.expand_dims(yolo_tensor, axis=0)  # [1, 3, 640, 640]

    # CLIP center crop (256x256)
    crop_size = min(orig_h, orig_w)
    start_h = (orig_h - crop_size) // 2
    start_w = (orig_w - crop_size) // 2
    clip_crop = img[start_h : start_h + crop_size, start_w : start_w + crop_size]
    clip_img = cv2.resize(clip_crop, (CLIP_SIZE, CLIP_SIZE), interpolation=cv2.INTER_LINEAR)

    clip_rgb = cv2.cvtColor(clip_img, cv2.COLOR_BGR2RGB)
    clip_tensor = clip_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    clip_tensor = np.expand_dims(clip_tensor, axis=0)  # [1, 3, 256, 256]

    return yolo_tensor, clip_tensor, img, (orig_h, orig_w)


def _crop_box_for_clip(
    original_bgr: np.ndarray, box: np.ndarray, orig_shape: tuple[int, int]
) -> np.ndarray:
    """Crop a single box from original image for CLIP embedding."""
    orig_h, orig_w = orig_shape
    x1, y1, x2, y2 = box

    px1 = max(0, int(x1 * orig_w))
    py1 = max(0, int(y1 * orig_h))
    px2 = min(orig_w, int(x2 * orig_w))
    py2 = min(orig_h, int(y2 * orig_h))

    crop = original_bgr[py1:py2, px1:px2]
    if crop.size == 0:
        crop = np.full((CLIP_SIZE, CLIP_SIZE, 3), 128, dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (CLIP_SIZE, CLIP_SIZE), interpolation=cv2.INTER_LINEAR)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = crop_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)  # [1, 3, 256, 256]


def _crop_person_for_face(
    original_bgr: np.ndarray, box: np.ndarray, orig_shape: tuple[int, int]
) -> np.ndarray:
    """Crop person box and letterbox to 640x640 for face detection."""
    orig_h, orig_w = orig_shape
    x1, y1, x2, y2 = box

    px1 = max(0, int(x1 * orig_w))
    py1 = max(0, int(y1 * orig_h))
    px2 = min(orig_w, int(x2 * orig_w))
    py2 = min(orig_h, int(y2 * orig_h))

    crop = original_bgr[py1:py2, px1:px2]
    if crop.size == 0:
        return np.full((1, 3, YOLO_SIZE, YOLO_SIZE), 114 / 255, dtype=np.float32)

    # Letterbox to 640x640
    h, w = crop.shape[:2]
    scale = min(YOLO_SIZE / h, YOLO_SIZE / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = (YOLO_SIZE - new_h) // 2
    pad_w = (YOLO_SIZE - new_w) // 2
    letterboxed = np.full((YOLO_SIZE, YOLO_SIZE, 3), 114, dtype=np.uint8)
    letterboxed[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)  # [1, 3, 640, 640]


def _crop_face_for_arcface(
    original_bgr: np.ndarray,
    person_box: np.ndarray,
    face_box: np.ndarray,
    orig_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop face from original image for ArcFace.

    Returns:
        face_tensor: [1, 3, 112, 112] FP32 range [-1, 1]
        face_box_normalized: [4] normalized to original image
    """
    orig_h, orig_w = orig_shape

    # Person box in pixels
    px1 = int(person_box[0] * orig_w)
    py1 = int(person_box[1] * orig_h)
    person_w = int((person_box[2] - person_box[0]) * orig_w)
    person_h = int((person_box[3] - person_box[1]) * orig_h)

    # Face box is normalized within person crop
    fx1, fy1, fx2, fy2 = face_box
    face_x1 = px1 + int(fx1 * person_w)
    face_y1 = py1 + int(fy1 * person_h)
    face_x2 = px1 + int(fx2 * person_w)
    face_y2 = py1 + int(fy2 * person_h)

    # Add 40% margin
    face_w = face_x2 - face_x1
    face_h = face_y2 - face_y1
    margin_w = int(face_w * 0.4)
    margin_h = int(face_h * 0.4)

    face_x1 = max(0, face_x1 - margin_w)
    face_y1 = max(0, face_y1 - margin_h)
    face_x2 = min(orig_w, face_x2 + margin_w)
    face_y2 = min(orig_h, face_y2 + margin_h)

    # Crop and resize
    face_crop = original_bgr[face_y1:face_y2, face_x1:face_x2]
    if face_crop.size == 0:
        face_crop = np.full((ARCFACE_SIZE, ARCFACE_SIZE, 3), 128, dtype=np.uint8)
    else:
        face_crop = cv2.resize(
            face_crop, (ARCFACE_SIZE, ARCFACE_SIZE), interpolation=cv2.INTER_LINEAR
        )

    # Convert to RGB and normalize to [-1, 1]
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_tensor = (face_rgb.astype(np.float32) - 127.5) / 128.0
    face_tensor = face_tensor.transpose(2, 0, 1)
    face_tensor = np.expand_dims(face_tensor, axis=0)  # [1, 3, 112, 112]

    # Normalized box in original image
    face_box_norm = np.array(
        [face_x1 / orig_w, face_y1 / orig_h, face_x2 / orig_w, face_y2 / orig_h]
    )

    return face_tensor, face_box_norm


async def process_single_image_async(
    image_bytes: bytes,
    image_id: str,
    image_path: str | None,
    triton_pool: AsyncTritonPool,
    executor: ThreadPoolExecutor,
    conf_threshold: float = 0.25,
    face_conf_threshold: float = 0.5,
    enable_ocr: bool = False,
) -> IngestResult:
    """
    Process a single image through the full pipeline (async version).

    Uses AsyncTritonPool for true async gRPC calls with connection pooling.
    CPU preprocessing runs in ThreadPoolExecutor to avoid blocking.
    """
    t_start = time.perf_counter()

    # Initialize result
    result = IngestResult(
        image_id=image_id,
        image_path=image_path,
        orig_width=0,
        orig_height=0,
    )

    try:
        # === CPU Preprocessing (run in executor to avoid blocking) ===
        t_preprocess = time.perf_counter()
        loop = asyncio.get_event_loop()
        yolo_tensor, clip_tensor, original_bgr, orig_shape = await loop.run_in_executor(
            executor, _cpu_preprocess, image_bytes
        )
        result.orig_height, result.orig_width = orig_shape
        preprocess_ms = (time.perf_counter() - t_preprocess) * 1000

        t_inference = time.perf_counter()

        # === Parallel: YOLO + CLIP (run concurrently) ===
        yolo_input = InferInput('images', [1, 3, 640, 640], 'FP32')
        yolo_input.set_data_from_numpy(yolo_tensor)

        yolo_outputs = [
            InferRequestedOutput('num_dets'),
            InferRequestedOutput('det_boxes'),
            InferRequestedOutput('det_scores'),
            InferRequestedOutput('det_classes'),
        ]

        clip_input = InferInput('images', [1, 3, 256, 256], 'FP32')
        clip_input.set_data_from_numpy(clip_tensor)

        clip_output = InferRequestedOutput('image_embeddings')

        # Run YOLO and CLIP in parallel
        yolo_task = triton_pool.infer('yolov11_small_trt_end2end', [yolo_input], yolo_outputs)
        clip_task = triton_pool.infer(
            'mobileclip2_s2_image_encoder', [clip_input], [clip_output]
        )

        yolo_response, clip_response = await asyncio.gather(yolo_task, clip_task)

        # Parse YOLO results
        num_dets = int(yolo_response.as_numpy('num_dets')[0, 0])
        result.num_dets = num_dets

        if num_dets > 0:
            result.boxes = yolo_response.as_numpy('det_boxes')[0, :num_dets]
            result.scores = yolo_response.as_numpy('det_scores')[0, :num_dets]
            result.classes = yolo_response.as_numpy('det_classes')[0, :num_dets]

        # Parse CLIP results
        result.global_embedding = clip_response.as_numpy('image_embeddings')[0]

        # === Per-Box Embeddings (Batched) ===
        if num_dets > 0:
            # Prepare all box crops at once (CPU)
            box_crops = []
            for i in range(num_dets):
                box_crop = await loop.run_in_executor(
                    executor, _crop_box_for_clip, original_bgr, result.boxes[i], orig_shape
                )
                box_crops.append(box_crop[0])  # Remove batch dim

            # Stack into single batch tensor
            box_batch = np.stack(box_crops, axis=0)  # [N, 3, 256, 256]

            # Single batched Triton call
            box_input = InferInput('images', list(box_batch.shape), 'FP32')
            box_input.set_data_from_numpy(box_batch)

            box_response = await triton_pool.infer(
                'mobileclip2_s2_image_encoder', [box_input], [clip_output]
            )
            result.box_embeddings = box_response.as_numpy('image_embeddings')  # [N, 512]

        # === Face Detection on Person Boxes (Batched) ===
        person_indices = []
        if num_dets > 0:
            person_mask = result.classes.astype(int) == PERSON_CLASS_ID
            person_indices = np.where(person_mask)[0]

        if len(person_indices) > 0:
            # Step 1: Batch all person crops for face detection
            person_crops = []
            person_boxes_list = []
            for person_idx in person_indices:
                person_box = result.boxes[person_idx]
                person_crop = await loop.run_in_executor(
                    executor, _crop_person_for_face, original_bgr, person_box, orig_shape
                )
                person_crops.append(person_crop[0])  # Remove batch dim
                person_boxes_list.append(person_box)

            # Single batched YOLO-Face call
            person_batch = np.stack(person_crops, axis=0)  # [N_persons, 3, 640, 640]
            face_input = InferInput('images', list(person_batch.shape), 'FP32')
            face_input.set_data_from_numpy(person_batch)

            face_outputs = [
                InferRequestedOutput('num_dets'),
                InferRequestedOutput('det_boxes'),
                InferRequestedOutput('det_scores'),
            ]

            face_response = await triton_pool.infer(
                'yolo11_face_small_trt_end2end', [face_input], face_outputs
            )

            # Parse batch results and collect face crops
            all_face_tensors = []
            all_face_boxes_norm = []
            all_face_scores = []
            all_face_person_idx = []

            num_dets_batch = face_response.as_numpy('num_dets')  # [N_persons, 1]
            det_boxes_batch = face_response.as_numpy('det_boxes')  # [N_persons, max_dets, 4]
            det_scores_batch = face_response.as_numpy('det_scores')  # [N_persons, max_dets]

            for p_idx, (person_idx, person_box) in enumerate(
                zip(person_indices, person_boxes_list)
            ):
                n_faces = int(num_dets_batch[p_idx, 0])
                if n_faces > 0:
                    face_boxes_raw = det_boxes_batch[p_idx, :n_faces]
                    face_scores_raw = det_scores_batch[p_idx, :n_faces]

                    # Filter by confidence
                    mask = face_scores_raw >= face_conf_threshold
                    face_boxes_filtered = face_boxes_raw[mask]
                    face_scores_filtered = face_scores_raw[mask]

                    # Collect face crops for batched ArcFace
                    for face_box, face_score in zip(face_boxes_filtered, face_scores_filtered):
                        face_tensor, face_box_norm = await loop.run_in_executor(
                            executor,
                            _crop_face_for_arcface,
                            original_bgr,
                            person_box,
                            face_box,
                            orig_shape,
                        )
                        all_face_tensors.append(face_tensor[0])  # Remove batch dim
                        all_face_boxes_norm.append(face_box_norm)
                        all_face_scores.append(face_score)
                        all_face_person_idx.append(person_idx)

            # Step 2: Batch all face crops for ArcFace
            if all_face_tensors:
                face_batch = np.stack(all_face_tensors, axis=0)  # [N_faces, 3, 112, 112]

                arcface_input = InferInput('input.1', list(face_batch.shape), 'FP32')
                arcface_input.set_data_from_numpy(face_batch)

                arcface_output = InferRequestedOutput('683')

                arcface_response = await triton_pool.infer(
                    'arcface_w600k_r50', [arcface_input], [arcface_output]
                )

                result.num_faces = len(all_face_tensors)
                result.face_boxes = np.stack(all_face_boxes_norm, axis=0)
                result.face_scores = np.array(all_face_scores, dtype=np.float32)
                result.face_embeddings = arcface_response.as_numpy('683')  # [N_faces, 512]
                result.face_person_idx = np.array(all_face_person_idx, dtype=np.int32)

        # === OCR (optional) ===
        if enable_ocr:
            try:
                # Import OCR service lazily
                from src.services.ocr_service import get_ocr_service

                ocr_service = await get_ocr_service()
                ocr_result = await ocr_service.process_image_async(image_bytes)

                if ocr_result and ocr_result.get('texts'):
                    result.num_ocr = len(ocr_result['texts'])
                    result.ocr_texts = ocr_result['texts']
                    if 'boxes' in ocr_result:
                        result.ocr_boxes = np.array(ocr_result['boxes'], dtype=np.float32)
                    if 'scores' in ocr_result:
                        result.ocr_scores = np.array(ocr_result['scores'], dtype=np.float32)
            except Exception as e:
                logger.debug(f'OCR failed (non-fatal): {e}')

        inference_ms = (time.perf_counter() - t_inference) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000

        result.preprocess_ms = preprocess_ms
        result.inference_ms = inference_ms
        result.total_ms = total_ms

    except Exception as e:
        result.error = str(e)
        result.total_ms = (time.perf_counter() - t_start) * 1000
        logger.error(f'Failed to process image {image_id}: {e}')

    return result


async def ingest_parallel_async(
    images_data: list[tuple[bytes, str, str | None]],
    triton_pool: AsyncTritonPool,
    executor: ThreadPoolExecutor,
    max_concurrent: int = 64,
    enable_ocr: bool = False,
) -> list[IngestResult]:
    """
    Parallel async ingestion with AsyncTritonPool.

    Uses asyncio.gather for concurrent image processing with semaphore
    for backpressure control.

    Args:
        images_data: List of (image_bytes, image_id, image_path) tuples
        triton_pool: AsyncTritonPool instance
        executor: ThreadPoolExecutor for CPU preprocessing
        max_concurrent: Max concurrent images to process
        enable_ocr: Enable OCR text extraction

    Returns:
        List of IngestResult, one per image
    """
    if not images_data:
        return []

    n_images = len(images_data)
    t_start = time.perf_counter()

    # Semaphore for backpressure
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(
        img_bytes: bytes, img_id: str, img_path: str | None
    ) -> IngestResult:
        async with semaphore:
            return await process_single_image_async(
                img_bytes,
                img_id,
                img_path,
                triton_pool,
                executor,
                enable_ocr=enable_ocr,
            )

    # Process all images concurrently
    tasks = [
        process_with_semaphore(img_bytes, img_id, img_path)
        for img_bytes, img_id, img_path in images_data
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to error results
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            final_results.append(
                IngestResult(
                    image_id=images_data[i][1],
                    image_path=images_data[i][2],
                    orig_width=0,
                    orig_height=0,
                    error=str(result),
                )
            )
        else:
            final_results.append(result)

    total_ms = (time.perf_counter() - t_start) * 1000
    successful = sum(1 for r in final_results if r.error is None)

    logger.info(
        f'[PARALLEL_ASYNC] {n_images} images in {total_ms:.1f}ms '
        f'({total_ms / n_images:.1f}ms/img, {n_images * 1000 / total_ms:.1f} RPS) '
        f'[{successful} success, {n_images - successful} errors]'
    )

    return final_results


# =============================================================================
# Backward Compatible Sync API
# =============================================================================


def process_single_image(
    image_bytes: bytes,
    image_id: str,
    image_path: str | None,
    triton_client: Any,
    conf_threshold: float = 0.25,
    face_conf_threshold: float = 0.5,
) -> IngestResult:
    """
    Process a single image through the full pipeline (sync version).

    This function is designed to be called in parallel by ThreadPoolExecutor.
    Each worker processes one image, and Triton auto-batches concurrent requests.

    Kept for backward compatibility - prefer async version for new code.
    """
    from tritonclient.grpc import InferInput as SyncInferInput
    from tritonclient.grpc import InferRequestedOutput as SyncInferRequestedOutput

    t_start = time.perf_counter()

    # Initialize result
    result = IngestResult(
        image_id=image_id,
        image_path=image_path,
        orig_width=0,
        orig_height=0,
    )

    try:
        # === CPU Preprocessing ===
        t_preprocess = time.perf_counter()
        yolo_tensor, clip_tensor, original_bgr, orig_shape = _cpu_preprocess(image_bytes)
        result.orig_height, result.orig_width = orig_shape
        preprocess_ms = (time.perf_counter() - t_preprocess) * 1000

        t_inference = time.perf_counter()

        # === YOLO Detection ===
        yolo_input = SyncInferInput('images', [1, 3, 640, 640], 'FP32')
        yolo_input.set_data_from_numpy(yolo_tensor)

        yolo_outputs = [
            SyncInferRequestedOutput('num_dets'),
            SyncInferRequestedOutput('det_boxes'),
            SyncInferRequestedOutput('det_scores'),
            SyncInferRequestedOutput('det_classes'),
        ]

        yolo_response = triton_client._infer_with_retry(
            'yolov11_small_trt_end2end', [yolo_input], yolo_outputs
        )

        num_dets = int(yolo_response.as_numpy('num_dets')[0, 0])
        result.num_dets = num_dets

        if num_dets > 0:
            result.boxes = yolo_response.as_numpy('det_boxes')[0, :num_dets]
            result.scores = yolo_response.as_numpy('det_scores')[0, :num_dets]
            result.classes = yolo_response.as_numpy('det_classes')[0, :num_dets]

        # === MobileCLIP Global Embedding ===
        clip_input = SyncInferInput('images', [1, 3, 256, 256], 'FP32')
        clip_input.set_data_from_numpy(clip_tensor)

        clip_output = SyncInferRequestedOutput('image_embeddings')

        clip_response = triton_client._infer_with_retry(
            'mobileclip2_s2_image_encoder', [clip_input], [clip_output]
        )

        result.global_embedding = clip_response.as_numpy('image_embeddings')[0]

        # === Per-Box Embeddings (Batched) ===
        if num_dets > 0:
            # Prepare all box crops at once (CPU)
            box_crops = []
            for i in range(num_dets):
                box_crop = _crop_box_for_clip(original_bgr, result.boxes[i], orig_shape)
                box_crops.append(box_crop[0])  # Remove batch dim

            # Stack into single batch tensor
            box_batch = np.stack(box_crops, axis=0)  # [N, 3, 256, 256]

            # Single batched Triton call
            box_input = SyncInferInput('images', list(box_batch.shape), 'FP32')
            box_input.set_data_from_numpy(box_batch)

            box_response = triton_client._infer_with_retry(
                'mobileclip2_s2_image_encoder', [box_input], [clip_output]
            )
            result.box_embeddings = box_response.as_numpy('image_embeddings')  # [N, 512]

        # === Face Detection on Person Boxes (Batched) ===
        person_indices = []
        if num_dets > 0:
            person_mask = result.classes.astype(int) == PERSON_CLASS_ID
            person_indices = np.where(person_mask)[0]

        if len(person_indices) > 0:
            # Step 1: Batch all person crops for face detection
            person_crops = []
            person_boxes_list = []
            for person_idx in person_indices:
                person_box = result.boxes[person_idx]
                person_crop = _crop_person_for_face(original_bgr, person_box, orig_shape)
                person_crops.append(person_crop[0])  # Remove batch dim
                person_boxes_list.append(person_box)

            # Single batched YOLO-Face call
            person_batch = np.stack(person_crops, axis=0)  # [N_persons, 3, 640, 640]
            face_input = SyncInferInput('images', list(person_batch.shape), 'FP32')
            face_input.set_data_from_numpy(person_batch)

            face_outputs = [
                SyncInferRequestedOutput('num_dets'),
                SyncInferRequestedOutput('det_boxes'),
                SyncInferRequestedOutput('det_scores'),
            ]

            face_response = triton_client._infer_with_retry(
                'yolo11_face_small_trt_end2end', [face_input], face_outputs
            )

            # Parse batch results and collect face crops
            all_face_tensors = []
            all_face_boxes_norm = []
            all_face_scores = []
            all_face_person_idx = []

            num_dets_batch = face_response.as_numpy('num_dets')  # [N_persons, 1]
            det_boxes_batch = face_response.as_numpy('det_boxes')  # [N_persons, max_dets, 4]
            det_scores_batch = face_response.as_numpy('det_scores')  # [N_persons, max_dets]

            for p_idx, (person_idx, person_box) in enumerate(
                zip(person_indices, person_boxes_list)
            ):
                n_faces = int(num_dets_batch[p_idx, 0])
                if n_faces > 0:
                    face_boxes_raw = det_boxes_batch[p_idx, :n_faces]
                    face_scores_raw = det_scores_batch[p_idx, :n_faces]

                    # Filter by confidence
                    mask = face_scores_raw >= face_conf_threshold
                    face_boxes_filtered = face_boxes_raw[mask]
                    face_scores_filtered = face_scores_raw[mask]

                    # Collect face crops for batched ArcFace
                    for face_box, face_score in zip(face_boxes_filtered, face_scores_filtered):
                        face_tensor, face_box_norm = _crop_face_for_arcface(
                            original_bgr, person_box, face_box, orig_shape
                        )
                        all_face_tensors.append(face_tensor[0])  # Remove batch dim
                        all_face_boxes_norm.append(face_box_norm)
                        all_face_scores.append(face_score)
                        all_face_person_idx.append(person_idx)

            # Step 2: Batch all face crops for ArcFace
            if all_face_tensors:
                face_batch = np.stack(all_face_tensors, axis=0)  # [N_faces, 3, 112, 112]

                arcface_input = SyncInferInput('input.1', list(face_batch.shape), 'FP32')
                arcface_input.set_data_from_numpy(face_batch)

                arcface_output = SyncInferRequestedOutput('683')

                arcface_response = triton_client._infer_with_retry(
                    'arcface_w600k_r50', [arcface_input], [arcface_output]
                )

                result.num_faces = len(all_face_tensors)
                result.face_boxes = np.stack(all_face_boxes_norm, axis=0)
                result.face_scores = np.array(all_face_scores, dtype=np.float32)
                result.face_embeddings = arcface_response.as_numpy('683')  # [N_faces, 512]
                result.face_person_idx = np.array(all_face_person_idx, dtype=np.int32)

        inference_ms = (time.perf_counter() - t_inference) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000

        result.preprocess_ms = preprocess_ms
        result.inference_ms = inference_ms
        result.total_ms = total_ms

    except Exception as e:
        result.error = str(e)
        result.total_ms = (time.perf_counter() - t_start) * 1000
        logger.error(f'Failed to process image {image_id}: {e}')

    return result


def ingest_parallel(
    images_data: list[tuple[bytes, str, str | None]],
    triton_client: Any,
    max_workers: int = 64,
) -> list[IngestResult]:
    """
    Parallel ingestion with per-image workers (sync version).

    Each worker processes one image independently:
    1. CPU preprocessing (decode, resize)
    2. Triton inference calls (YOLO, CLIP, Face, ArcFace)
    3. Return results

    Triton automatically batches concurrent requests from all workers.

    Kept for backward compatibility - prefer async version for new code.

    Args:
        images_data: List of (image_bytes, image_id, image_path) tuples
        triton_client: TritonClient instance
        max_workers: Max parallel workers (default: 64)

    Returns:
        List of IngestResult, one per image
    """
    from concurrent.futures import as_completed

    if not images_data:
        return []

    n_images = len(images_data)
    results = [None] * n_images

    t_start = time.perf_counter()

    workers = min(max_workers, n_images)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_single_image, img_bytes, img_id, img_path, triton_client
            ): i
            for i, (img_bytes, img_id, img_path) in enumerate(images_data)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f'Worker failed for image {idx}: {e}')
                results[idx] = IngestResult(
                    image_id=images_data[idx][1],
                    image_path=images_data[idx][2],
                    orig_width=0,
                    orig_height=0,
                    error=str(e),
                )

    total_ms = (time.perf_counter() - t_start) * 1000
    successful = sum(1 for r in results if r and r.error is None)

    logger.info(
        f'[PARALLEL] {n_images} images in {total_ms:.1f}ms '
        f'({total_ms / n_images:.1f}ms/img, {n_images * 1000 / total_ms:.1f} RPS) '
        f'[{successful} success, {n_images - successful} errors]'
    )

    return results
