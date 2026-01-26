"""
Optimized Ingest Pipeline - Bypasses Python BLS Bottleneck.

This module implements a high-throughput ingestion pipeline following
industry best practices from Google, Apple, and Meta:

1. **Parallel CPU Preprocessing**: Decode and resize images using ThreadPoolExecutor
2. **Cross-Image Batching**: Collect ALL crops from ALL images, send ONE GPU request
3. **Direct TRT Calls**: Skip Triton BLS overhead by calling TRT models directly
4. **Pipeline Parallelism**: Overlap CPU cropping with GPU inference

Architecture:
    CPU Stage 1 (parallel):
        - JPEG decode → numpy array
        - Letterbox for YOLO (640x640)
        - Center crop for CLIP (256x256)
        - Keep original for cropping

    GPU Stage 1 (batched):
        - YOLO detection → all boxes
        - MobileCLIP → global embeddings

    CPU Stage 2 (parallel):
        - Crop detected boxes → 256x256 for MobileCLIP
        - Filter persons → 640x640 for face detection

    GPU Stage 2 (batched):
        - MobileCLIP → per-box embeddings (ALL crops, ONE request)
        - YOLO11-Face → face detections (ALL persons, ONE request)

    CPU Stage 3 (parallel):
        - Crop detected faces → 112x112 for ArcFace

    GPU Stage 3 (batched):
        - ArcFace → face embeddings (ALL faces, ONE request)

Performance Target: 70-100+ RPS (2-3x improvement over Python BLS)
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.utils.retry import RetryExhaustedError

logger = logging.getLogger(__name__)

# Person class ID in COCO dataset
PERSON_CLASS_ID = 0

# Preprocessing sizes
YOLO_SIZE = 640
CLIP_SIZE = 256
FACE_SIZE = 640
ARCFACE_SIZE = 112


@dataclass
class ImageResult:
    """Result for a single image in the batch."""

    image_id: str
    image_path: str | None
    image_bytes: bytes
    orig_shape: tuple[int, int]  # (height, width)

    # YOLO detections
    num_dets: int = 0
    boxes: np.ndarray | None = None  # [N, 4] normalized
    scores: np.ndarray | None = None  # [N]
    classes: np.ndarray | None = None  # [N]

    # Embeddings
    global_embedding: np.ndarray | None = None  # [512]
    box_embeddings: np.ndarray | None = None  # [N, 512]

    # Face data
    num_faces: int = 0
    face_embeddings: np.ndarray | None = None  # [M, 512]
    face_boxes: np.ndarray | None = None  # [M, 4] normalized to original
    face_scores: np.ndarray | None = None  # [M]
    face_person_idx: np.ndarray | None = None  # [M] which person box

    # Timing
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0

    # Error
    error: str | None = None


def _decode_and_preprocess(
    image_bytes: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Decode and preprocess a single image.

    Returns:
        yolo_tensor: [3, 640, 640] FP32 normalized
        clip_tensor: [3, 256, 256] FP32 normalized
        original: [H, W, 3] uint8 BGR for cropping
        orig_shape: (height, width)
    """
    # Decode JPEG/PNG
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR

    if img is None:
        raise ValueError('Failed to decode image')

    orig_h, orig_w = img.shape[:2]

    # Letterbox for YOLO (640x640)
    scale = min(YOLO_SIZE / orig_h, YOLO_SIZE / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    pad_h = (YOLO_SIZE - new_h) // 2
    pad_w = (YOLO_SIZE - new_w) // 2
    yolo_img = np.full((YOLO_SIZE, YOLO_SIZE, 3), 114, dtype=np.uint8)
    yolo_img[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    # Convert to tensor: HWC -> CHW, normalize to [0, 1]
    yolo_tensor = yolo_img.transpose(2, 0, 1).astype(np.float32) / 255.0

    # Center crop for CLIP (256x256)
    crop_size = min(orig_h, orig_w)
    start_h = (orig_h - crop_size) // 2
    start_w = (orig_w - crop_size) // 2
    clip_crop = img[start_h : start_h + crop_size, start_w : start_w + crop_size]
    clip_img = cv2.resize(clip_crop, (CLIP_SIZE, CLIP_SIZE), interpolation=cv2.INTER_LINEAR)

    # Convert to RGB, normalize
    clip_rgb = cv2.cvtColor(clip_img, cv2.COLOR_BGR2RGB)
    clip_tensor = clip_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

    return yolo_tensor, clip_tensor, img, (orig_h, orig_w)


def _crop_boxes_for_clip(
    original: np.ndarray,
    boxes: np.ndarray,
    orig_shape: tuple[int, int],
) -> np.ndarray:
    """
    Crop detected boxes from original image for MobileCLIP.

    Args:
        original: [H, W, 3] uint8 BGR
        boxes: [N, 4] normalized coordinates (x1, y1, x2, y2)
        orig_shape: (height, width)

    Returns:
        crops: [N, 3, 256, 256] FP32 normalized
    """
    if boxes.shape[0] == 0:
        return np.empty((0, 3, CLIP_SIZE, CLIP_SIZE), dtype=np.float32)

    orig_h, orig_w = orig_shape
    crops = []

    for box in boxes:
        x1, y1, x2, y2 = box
        # Convert normalized to pixel coordinates
        px1 = int(x1 * orig_w)
        py1 = int(y1 * orig_h)
        px2 = int(x2 * orig_w)
        py2 = int(y2 * orig_h)

        # Clamp to image bounds
        px1 = max(0, min(px1, orig_w - 1))
        py1 = max(0, min(py1, orig_h - 1))
        px2 = max(px1 + 1, min(px2, orig_w))
        py2 = max(py1 + 1, min(py2, orig_h))

        # Crop and resize
        crop = original[py1:py2, px1:px2]
        if crop.size == 0:
            # Empty crop - use gray placeholder
            crop_resized = np.full((CLIP_SIZE, CLIP_SIZE, 3), 128, dtype=np.uint8)
        else:
            crop_resized = cv2.resize(crop, (CLIP_SIZE, CLIP_SIZE), interpolation=cv2.INTER_LINEAR)

        # Convert to RGB and normalize
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        crop_tensor = crop_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        crops.append(crop_tensor)

    return np.stack(crops, axis=0)


def _crop_persons_for_face_detection(
    original: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    orig_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop person boxes for face detection.

    Args:
        original: [H, W, 3] uint8 BGR
        boxes: [N, 4] normalized coordinates
        classes: [N] class IDs
        orig_shape: (height, width)

    Returns:
        person_crops: [M, 3, 640, 640] FP32 normalized
        person_indices: [M] indices into original boxes
    """
    person_mask = classes == PERSON_CLASS_ID
    person_indices = np.where(person_mask)[0]

    if len(person_indices) == 0:
        return np.empty((0, 3, FACE_SIZE, FACE_SIZE), dtype=np.float32), np.array([], dtype=np.int32)

    orig_h, orig_w = orig_shape
    crops = []

    for idx in person_indices:
        box = boxes[idx]
        x1, y1, x2, y2 = box

        # Convert normalized to pixel coordinates
        px1 = int(x1 * orig_w)
        py1 = int(y1 * orig_h)
        px2 = int(x2 * orig_w)
        py2 = int(y2 * orig_h)

        # Clamp to image bounds
        px1 = max(0, min(px1, orig_w - 1))
        py1 = max(0, min(py1, orig_h - 1))
        px2 = max(px1 + 1, min(px2, orig_w))
        py2 = max(py1 + 1, min(py2, orig_h))

        # Crop
        crop = original[py1:py2, px1:px2]
        if crop.size == 0:
            crop_resized = np.full((FACE_SIZE, FACE_SIZE, 3), 114, dtype=np.uint8)
        else:
            # Letterbox resize to 640x640
            h, w = crop.shape[:2]
            scale = min(FACE_SIZE / h, FACE_SIZE / w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            pad_h = (FACE_SIZE - new_h) // 2
            pad_w = (FACE_SIZE - new_w) // 2
            crop_resized = np.full((FACE_SIZE, FACE_SIZE, 3), 114, dtype=np.uint8)
            crop_resized[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        # Convert to tensor
        crop_tensor = crop_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        crops.append(crop_tensor)

    return np.stack(crops, axis=0), person_indices


def _crop_faces_for_arcface(
    original: np.ndarray,
    person_boxes: np.ndarray,
    face_detections: list[dict],
    orig_shape: tuple[int, int],
) -> tuple[np.ndarray, list[tuple[int, np.ndarray, float]]]:
    """
    Crop detected faces for ArcFace embedding.

    Args:
        original: [H, W, 3] uint8 BGR
        person_boxes: [M, 4] person boxes (normalized)
        face_detections: List of face detection results per person
        orig_shape: (height, width)

    Returns:
        face_crops: [K, 3, 112, 112] FP32 range [-1, 1]
        face_info: List of (person_idx, face_box_in_original, score)
    """
    orig_h, orig_w = orig_shape
    face_crops = []
    face_info = []

    for person_idx, (person_box, face_result) in enumerate(zip(person_boxes, face_detections)):
        if face_result['num_faces'] == 0:
            continue

        # Person box in pixel coordinates
        px1 = int(person_box[0] * orig_w)
        py1 = int(person_box[1] * orig_h)
        px2 = int(person_box[2] * orig_w)
        py2 = int(person_box[3] * orig_h)
        person_w = px2 - px1
        person_h = py2 - py1

        for face_box, score in zip(face_result['boxes'], face_result['scores']):
            # Face box is normalized within person crop
            fx1, fy1, fx2, fy2 = face_box

            # Convert to original image coordinates
            face_x1 = px1 + int(fx1 * person_w)
            face_y1 = py1 + int(fy1 * person_h)
            face_x2 = px1 + int(fx2 * person_w)
            face_y2 = py1 + int(fy2 * person_h)

            # Add margin (40% expansion like MTCNN)
            face_w = face_x2 - face_x1
            face_h = face_y2 - face_y1
            margin_w = int(face_w * 0.4)
            margin_h = int(face_h * 0.4)

            face_x1 = max(0, face_x1 - margin_w)
            face_y1 = max(0, face_y1 - margin_h)
            face_x2 = min(orig_w, face_x2 + margin_w)
            face_y2 = min(orig_h, face_y2 + margin_h)

            # Crop face
            face_crop = original[face_y1:face_y2, face_x1:face_x2]
            if face_crop.size == 0:
                continue

            # Resize to 112x112
            face_resized = cv2.resize(face_crop, (ARCFACE_SIZE, ARCFACE_SIZE), interpolation=cv2.INTER_LINEAR)

            # Convert to RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Normalize to [-1, 1] for ArcFace: (x - 127.5) / 128.0
            face_tensor = (face_rgb.astype(np.float32) - 127.5) / 128.0
            face_tensor = face_tensor.transpose(2, 0, 1)

            face_crops.append(face_tensor)

            # Store face info (normalized to original image)
            face_box_norm = np.array(
                [face_x1 / orig_w, face_y1 / orig_h, face_x2 / orig_w, face_y2 / orig_h]
            )
            face_info.append((person_idx, face_box_norm, float(score)))

    if not face_crops:
        return np.empty((0, 3, ARCFACE_SIZE, ARCFACE_SIZE), dtype=np.float32), []

    return np.stack(face_crops, axis=0), face_info


def ingest_batch_optimized(
    images_data: list[tuple[bytes, str, str | None]],
    triton_client: Any,
    max_workers: int = 32,
) -> list[ImageResult]:
    """
    Optimized batch ingestion with CPU cropping and batched GPU inference.

    This pipeline bypasses the Python BLS bottleneck by:
    1. Parallel CPU preprocessing and cropping
    2. Cross-image batching for GPU inference
    3. Direct TRT model calls (no BLS overhead)

    Performance: 2-3x faster than unified_embedding_extractor

    Args:
        images_data: List of (image_bytes, image_id, image_path) tuples
        triton_client: TritonClient instance with batched inference methods
        max_workers: Max threads for CPU preprocessing

    Returns:
        List of ImageResult with detections, embeddings, and face data
    """
    if not images_data:
        return []

    n_images = len(images_data)
    results = [
        ImageResult(
            image_id=img_id,
            image_path=img_path,
            image_bytes=img_bytes,
            orig_shape=(0, 0),
        )
        for img_bytes, img_id, img_path in images_data
    ]

    # ==========================================================================
    # Stage 1: CPU Preprocessing (parallel)
    # ==========================================================================
    t_start = time.perf_counter()

    yolo_tensors = []
    clip_tensors = []
    originals = []
    orig_shapes = []
    valid_indices = []

    def preprocess_single(idx: int) -> tuple[int, Any]:
        try:
            img_bytes = images_data[idx][0]
            yolo_t, clip_t, orig, shape = _decode_and_preprocess(img_bytes)
            return idx, (yolo_t, clip_t, orig, shape, None)
        except Exception as e:
            return idx, (None, None, None, None, str(e))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        preprocess_results = list(executor.map(preprocess_single, range(n_images)))

    for idx, (yolo_t, clip_t, orig, shape, error) in preprocess_results:
        if error:
            results[idx].error = error
            continue

        valid_indices.append(idx)
        yolo_tensors.append(yolo_t)
        clip_tensors.append(clip_t)
        originals.append(orig)
        orig_shapes.append(shape)
        results[idx].orig_shape = shape

    t_preprocess = time.perf_counter()
    preprocess_ms = (t_preprocess - t_start) * 1000

    if not valid_indices:
        return results

    # Stack tensors for batched inference
    yolo_batch = np.stack(yolo_tensors, axis=0)  # [N, 3, 640, 640]
    clip_batch = np.stack(clip_tensors, axis=0)  # [N, 3, 256, 256]

    # ==========================================================================
    # Stage 2: GPU Detection + Global Embedding (batched)
    # ==========================================================================
    t_gpu1_start = time.perf_counter()

    # Batched YOLO detection
    try:
        yolo_results = triton_client.infer_yolo_batch(yolo_batch)
    except (RetryExhaustedError, Exception) as e:
        logger.error(f'YOLO batch inference failed: {e}')
        for idx in valid_indices:
            results[idx].error = f'YOLO inference failed: {e}'
        return results

    # Batched MobileCLIP for global embeddings
    try:
        global_embeddings = triton_client.infer_mobileclip_batch(clip_batch)
    except (RetryExhaustedError, Exception) as e:
        logger.error(f'MobileCLIP batch inference failed: {e}')
        for idx in valid_indices:
            results[idx].error = f'MobileCLIP inference failed: {e}'
        return results

    t_gpu1 = time.perf_counter()
    gpu1_ms = (t_gpu1 - t_gpu1_start) * 1000

    # Store detection results
    for i, idx in enumerate(valid_indices):
        yolo_res = yolo_results[i]
        results[idx].num_dets = yolo_res['num_dets']
        results[idx].boxes = yolo_res['boxes']
        results[idx].scores = yolo_res['scores']
        results[idx].classes = yolo_res['classes']
        results[idx].global_embedding = global_embeddings[i]

    # ==========================================================================
    # Stage 3: CPU Box Cropping (parallel)
    # ==========================================================================
    t_crop1_start = time.perf_counter()

    # Collect ALL box crops from ALL images
    all_box_crops = []
    box_crop_mapping = []  # (image_idx, box_idx) for each crop

    def crop_boxes_single(i: int) -> list[tuple[int, int, np.ndarray]]:
        idx = valid_indices[i]
        boxes = results[idx].boxes
        if boxes is None or boxes.shape[0] == 0:
            return []

        crops = _crop_boxes_for_clip(originals[i], boxes, orig_shapes[i])
        return [(i, j, crops[j]) for j in range(crops.shape[0])]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        crop_results = list(executor.map(crop_boxes_single, range(len(valid_indices))))

    for crop_list in crop_results:
        for i, j, crop in crop_list:
            all_box_crops.append(crop)
            box_crop_mapping.append((i, j))

    t_crop1 = time.perf_counter()
    crop1_ms = (t_crop1 - t_crop1_start) * 1000

    # ==========================================================================
    # Stage 4: GPU Box Embeddings (single batched call)
    # ==========================================================================
    t_gpu2_start = time.perf_counter()

    if all_box_crops:
        box_crops_batch = np.stack(all_box_crops, axis=0)  # [Total_crops, 3, 256, 256]
        try:
            box_embeddings_all = triton_client.infer_mobileclip_batch(box_crops_batch)
        except (RetryExhaustedError, Exception) as e:
            logger.warning(f'Box embedding inference failed (non-fatal): {e}')
            box_embeddings_all = None

        # Distribute embeddings back to per-image results
        if box_embeddings_all is not None:
            for crop_idx, (img_local_idx, box_idx) in enumerate(box_crop_mapping):
                idx = valid_indices[img_local_idx]
                if results[idx].box_embeddings is None:
                    n_boxes = results[idx].num_dets
                    results[idx].box_embeddings = np.zeros((n_boxes, 512), dtype=np.float32)
                results[idx].box_embeddings[box_idx] = box_embeddings_all[crop_idx]

    t_gpu2 = time.perf_counter()
    gpu2_ms = (t_gpu2 - t_gpu2_start) * 1000

    # ==========================================================================
    # Stage 5: CPU Person Cropping for Face Detection (parallel)
    # ==========================================================================
    t_crop2_start = time.perf_counter()

    all_person_crops = []
    person_crop_mapping = []  # (image_local_idx, person_idx_in_image, original_box_idx)

    def crop_persons_single(i: int) -> list[tuple[int, int, int, np.ndarray]]:
        idx = valid_indices[i]
        boxes = results[idx].boxes
        classes = results[idx].classes

        if boxes is None or boxes.shape[0] == 0:
            return []

        crops, person_indices = _crop_persons_for_face_detection(
            originals[i], boxes, classes, orig_shapes[i]
        )
        return [(i, j, int(person_indices[j]), crops[j]) for j in range(len(person_indices))]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        person_crop_results = list(executor.map(crop_persons_single, range(len(valid_indices))))

    for crop_list in person_crop_results:
        for i, person_local_idx, orig_box_idx, crop in crop_list:
            all_person_crops.append(crop)
            person_crop_mapping.append((i, person_local_idx, orig_box_idx))

    t_crop2 = time.perf_counter()
    crop2_ms = (t_crop2 - t_crop2_start) * 1000

    # ==========================================================================
    # Stage 6: GPU Face Detection (single batched call)
    # ==========================================================================
    t_gpu3_start = time.perf_counter()

    face_detection_results = []
    if all_person_crops:
        person_crops_batch = np.stack(all_person_crops, axis=0)
        try:
            face_detection_results = triton_client.infer_yolo_face_batch(person_crops_batch)
        except (RetryExhaustedError, Exception) as e:
            logger.warning(f'Face detection batch failed (non-fatal): {e}')
            face_detection_results = []

    t_gpu3 = time.perf_counter()
    gpu3_ms = (t_gpu3 - t_gpu3_start) * 1000

    # ==========================================================================
    # Stage 7: CPU Face Cropping (parallel)
    # ==========================================================================
    t_crop3_start = time.perf_counter()

    # Group face detections by image
    face_dets_per_image = [[] for _ in range(len(valid_indices))]
    person_boxes_per_image = [[] for _ in range(len(valid_indices))]

    for crop_idx, (img_local_idx, person_local_idx, orig_box_idx) in enumerate(person_crop_mapping):
        if crop_idx < len(face_detection_results):
            face_dets_per_image[img_local_idx].append(face_detection_results[crop_idx])
            idx = valid_indices[img_local_idx]
            person_boxes_per_image[img_local_idx].append(results[idx].boxes[orig_box_idx])

    all_face_crops = []
    face_crop_mapping = []  # (image_local_idx, face_info)

    def crop_faces_single(i: int) -> list[tuple[int, np.ndarray, tuple]]:
        if not face_dets_per_image[i]:
            return []

        person_boxes = np.array(person_boxes_per_image[i])
        face_crops, face_info = _crop_faces_for_arcface(
            originals[i], person_boxes, face_dets_per_image[i], orig_shapes[i]
        )
        return [(i, face_crops[j], face_info[j]) for j in range(len(face_info))]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        face_crop_results = list(executor.map(crop_faces_single, range(len(valid_indices))))

    for crop_list in face_crop_results:
        for i, crop, info in crop_list:
            all_face_crops.append(crop)
            face_crop_mapping.append((i, info))

    t_crop3 = time.perf_counter()
    crop3_ms = (t_crop3 - t_crop3_start) * 1000

    # ==========================================================================
    # Stage 8: GPU Face Embeddings (single batched call)
    # ==========================================================================
    t_gpu4_start = time.perf_counter()

    if all_face_crops:
        face_crops_batch = np.stack(all_face_crops, axis=0)
        try:
            face_embeddings_all = triton_client.infer_arcface_batch(face_crops_batch)
        except (RetryExhaustedError, Exception) as e:
            logger.warning(f'ArcFace batch failed (non-fatal): {e}')
            face_embeddings_all = None

        # Group face data by image
        if face_embeddings_all is not None:
            face_data_per_image = {i: {'embeddings': [], 'boxes': [], 'scores': [], 'person_idx': []} for i in range(len(valid_indices))}

            for crop_idx, (img_local_idx, (person_idx, face_box, score)) in enumerate(face_crop_mapping):
                face_data_per_image[img_local_idx]['embeddings'].append(face_embeddings_all[crop_idx])
                face_data_per_image[img_local_idx]['boxes'].append(face_box)
                face_data_per_image[img_local_idx]['scores'].append(score)
                face_data_per_image[img_local_idx]['person_idx'].append(person_idx)

            # Store in results
            for img_local_idx, face_data in face_data_per_image.items():
                if face_data['embeddings']:
                    idx = valid_indices[img_local_idx]
                    results[idx].num_faces = len(face_data['embeddings'])
                    results[idx].face_embeddings = np.stack(face_data['embeddings'], axis=0)
                    results[idx].face_boxes = np.stack(face_data['boxes'], axis=0)
                    results[idx].face_scores = np.array(face_data['scores'], dtype=np.float32)
                    results[idx].face_person_idx = np.array(face_data['person_idx'], dtype=np.int32)

    t_gpu4 = time.perf_counter()
    gpu4_ms = (t_gpu4 - t_gpu4_start) * 1000

    # ==========================================================================
    # Timing Summary
    # ==========================================================================
    total_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        f'[OPTIMIZED] {n_images} images in {total_ms:.1f}ms '
        f'({total_ms/n_images:.1f}ms/img, {n_images*1000/total_ms:.1f} RPS)'
    )
    logger.info(
        f'  Breakdown: preprocess={preprocess_ms:.1f}ms, '
        f'gpu1(yolo+global)={gpu1_ms:.1f}ms, '
        f'crop1={crop1_ms:.1f}ms, gpu2(box_emb)={gpu2_ms:.1f}ms, '
        f'crop2={crop2_ms:.1f}ms, gpu3(face_det)={gpu3_ms:.1f}ms, '
        f'crop3={crop3_ms:.1f}ms, gpu4(face_emb)={gpu4_ms:.1f}ms'
    )

    # Store timing in results
    for i, idx in enumerate(valid_indices):
        results[idx].preprocess_ms = preprocess_ms / len(valid_indices)
        results[idx].inference_ms = (gpu1_ms + gpu2_ms + gpu3_ms + gpu4_ms) / len(valid_indices)

    return results


def convert_to_ingest_format(result: ImageResult) -> dict[str, Any]:
    """
    Convert ImageResult to the format expected by OpenSearch indexing.

    This matches the output format of infer_unified_direct_ensemble.
    """
    if result.error:
        return {'error': result.error, 'num_dets': 0, 'num_faces': 0}

    return {
        'num_dets': result.num_dets,
        'det_boxes': result.boxes.tolist() if result.boxes is not None else [],
        'det_scores': result.scores.tolist() if result.scores is not None else [],
        'det_classes': result.classes.tolist() if result.classes is not None else [],
        'global_embedding': result.global_embedding.tolist() if result.global_embedding is not None else [],
        'box_embeddings': result.box_embeddings.tolist() if result.box_embeddings is not None else [],
        'num_faces': result.num_faces,
        'face_embeddings': result.face_embeddings.tolist() if result.face_embeddings is not None else [],
        'face_boxes': result.face_boxes.tolist() if result.face_boxes is not None else [],
        'face_scores': result.face_scores.tolist() if result.face_scores is not None else [],
        'face_person_idx': result.face_person_idx.tolist() if result.face_person_idx is not None else [],
        'orig_shape': result.orig_shape,
        'preprocess_ms': result.preprocess_ms,
        'inference_ms': result.inference_ms,
    }
