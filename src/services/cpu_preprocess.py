"""
CPU Preprocessing Module.

Provides CPU-based preprocessing using cv2 and numpy for all inference pipelines.

Preprocessing function:
- center_crop_cpu: MobileCLIP resize + center crop (256x256, BILINEAR per Apple/OpenCLIP)

All tensors are CHW format, FP32, normalized to [0, 1] range.
"""

from collections.abc import Callable

import cv2
import numpy as np


def center_crop_cpu(
    img_rgb: np.ndarray,
    target_size: int = 256,
) -> np.ndarray:
    """
    CPU preprocessing for MobileCLIP (matches Apple's OpenCLIP MobileCLIP config).

    Uses BILINEAR interpolation (cv2.INTER_LINEAR) per OpenCLIP _mccfg which sets
    interpolation='bilinear' for all MobileCLIP variants. Normalization is simple
    /255 (mean=0, std=1) per Apple's MobileCLIP2-S2 config.

    Pipeline: resize shortest edge → center crop → normalize [0,1] → CHW

    Args:
        img_rgb: HWC, RGB, uint8 numpy array
        target_size: Target crop size (default 256 for MobileCLIP)

    Returns:
        tensor: [3, target_size, target_size] FP32 normalized [0,1]
    """
    orig_h, orig_w = img_rgb.shape[:2]

    # Resize shortest edge to target_size. F-28: round() (not int()/truncate)
    # and a max(target_size, ...) floor -- on a small detection crop (common
    # on ordinary COCO images: a distant person/object a few dozen px wide),
    # scale * min(orig_h, orig_w) is mathematically target_size but float
    # rounding can land a hair under it (e.g. 255.9999...), and int()
    # truncates that to target_size - 1. The undershoot then made new_h (or
    # new_w) < target_size, so start_y/start_x went negative and the
    # subsequent slice silently returned a smaller (sometimes near-empty,
    # e.g. 1x1) crop instead of erroring -- np.stack across a batch of such
    # crops then failed with "all input arrays must have the same shape"
    # once one image's boxes happened to round differently than another's.
    scale = target_size / min(orig_h, orig_w)
    new_w = max(target_size, round(orig_w * scale))
    new_h = max(target_size, round(orig_h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop to target_size x target_size. new_w/new_h >= target_size
    # is now guaranteed, so start_x/start_y are always >= 0 and the crop is
    # always exactly (target_size, target_size).
    start_x = max(0, (new_w - target_size) // 2)
    start_y = max(0, (new_h - target_size) // 2)
    cropped = resized[start_y : start_y + target_size, start_x : start_x + target_size]

    # Normalize to [0, 1]
    normalized = cropped.astype(np.float32) / 255.0

    # HWC -> CHW
    return np.transpose(normalized, (2, 0, 1))


def embed_boxes_from_full_res(
    img_array: np.ndarray,
    normalized_boxes: np.ndarray,
    embed_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Crop each detection box from the full-resolution decoded image and
    return its MobileCLIP embedding, matching the /embed/boxes crop
    pipeline (numpy crop + CPU bilinear resize/center-crop to 256x256).

    Args:
        img_array: HWC RGB uint8 full-resolution image
        normalized_boxes: [N, 4] XYXY boxes normalized [0,1] against
            img_array's own dimensions
        embed_fn: batched embedding function, e.g.
            ``TritonClient.infer_mobileclip_batch``, taking
            [M, 3, 256, 256] FP32 and returning [M, 512] FP32

    Returns:
        [N, 512] FP32 embeddings (rows for out-of-bounds / degenerate boxes
        are zero vectors, in the same order as input)
    """
    orig_h, orig_w = img_array.shape[:2]
    n_boxes = normalized_boxes.shape[0]
    if n_boxes == 0:
        return np.empty((0, 512), dtype=np.float32)

    preprocessed: list[np.ndarray] = []
    valid_idx: list[int] = []
    for i in range(n_boxes):
        x1, y1, x2, y2 = normalized_boxes[i]
        px1 = max(0, min(int(x1 * orig_w), orig_w - 1))
        py1 = max(0, min(int(y1 * orig_h), orig_h - 1))
        px2 = max(px1 + 1, min(int(x2 * orig_w), orig_w))
        py2 = max(py1 + 1, min(int(y2 * orig_h), orig_h))
        if px2 <= px1 or py2 <= py1:
            continue
        crop_rgb = img_array[py1:py2, px1:px2]
        preprocessed.append(center_crop_cpu(crop_rgb, target_size=256))
        valid_idx.append(i)

    embeddings = np.zeros((n_boxes, 512), dtype=np.float32)
    if not preprocessed:
        return embeddings

    batch_tensor = np.stack(preprocessed)  # [M, 3, 256, 256]
    valid_embeddings = embed_fn(batch_tensor)  # [M, 512]
    for out_i, box_i in enumerate(valid_idx):
        embeddings[box_i] = valid_embeddings[out_i]

    return embeddings
