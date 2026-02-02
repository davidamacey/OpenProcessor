"""
CPU Preprocessing Module.

Provides CPU-based preprocessing using cv2 and numpy for all inference pipelines.

Preprocessing functions:
- letterbox_cpu: YOLO letterbox (640x640, matches Ultralytics LetterBox)
- center_crop_cpu: MobileCLIP resize + center crop (256x256, BILINEAR per Apple/OpenCLIP)
- resize_hd_cpu: HD resize for face alignment (max 1920px longest edge)

All tensors are CHW format, FP32, normalized to [0, 1] range.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    """Result of CPU preprocessing for a single image.

    Attributes:
        yolo_tensor: [3, 640, 640] FP32 normalized [0,1] - YOLO detection input
        clip_tensor: [3, 256, 256] FP32 normalized [0,1] - MobileCLIP input
        hd_tensor: [3, H, W] FP32 normalized [0,1] - HD image for face alignment
        affine_matrix: [2, 3] FP32 - YOLO letterbox transformation matrix
        orig_shape: (height, width) - Original image dimensions
        scale: float - Scale factor applied for letterbox
        padding: (pad_w, pad_h) - Padding applied for letterbox
    """

    yolo_tensor: np.ndarray  # [3, 640, 640] FP32
    clip_tensor: np.ndarray  # [3, 256, 256] FP32
    hd_tensor: np.ndarray  # [3, H, W] FP32, max 1920px longest edge
    affine_matrix: np.ndarray  # [2, 3] FP32
    orig_shape: tuple[int, int]  # (height, width)
    scale: float
    padding: tuple[float, float]  # (pad_w, pad_h)

    @property
    def hd_h(self) -> int:
        """Height of HD tensor."""
        return int(self.hd_tensor.shape[1])

    @property
    def hd_w(self) -> int:
        """Width of HD tensor."""
        return int(self.hd_tensor.shape[2])

    @property
    def orig_h(self) -> int:
        """Original image height."""
        return self.orig_shape[0]

    @property
    def orig_w(self) -> int:
        """Original image width."""
        return self.orig_shape[1]

    @property
    def yolo_image(self) -> np.ndarray:
        """YOLO tensor with batch dimension [1, 3, 640, 640]."""
        return np.expand_dims(self.yolo_tensor, axis=0)

    @property
    def clip_image(self) -> np.ndarray:
        """CLIP tensor with batch dimension [1, 3, 256, 256]."""
        return np.expand_dims(self.clip_tensor, axis=0)

    @property
    def original_image(self) -> np.ndarray:
        """HD tensor with batch dimension [1, 3, H, W]."""
        return np.expand_dims(self.hd_tensor, axis=0)

    @property
    def affine(self) -> np.ndarray:
        """Affine matrix with batch dimension [1, 2, 3]."""
        return np.expand_dims(self.affine_matrix, axis=0)


def letterbox_cpu(
    img_rgb: np.ndarray,
    target_size: int = 640,
    scaleup: bool = False,
    pad_value: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, np.ndarray, float, tuple[float, float]]:
    """
    CPU letterbox preprocessing for YOLO (matches Ultralytics LetterBox exactly).

    Scales image to fit within target_size without changing aspect ratio,
    then pads with gray (114, 114, 114) to reach target dimensions.

    Args:
        img_rgb: HWC, RGB, uint8 numpy array
        target_size: Target dimension (both width and height)
        scaleup: If False, don't upscale images smaller than target_size
        pad_value: RGB tuple for padding color (default gray 114, 114, 114)

    Returns:
        Tuple of:
        - tensor: [3, target_size, target_size] FP32 normalized [0,1]
        - affine_matrix: [2, 3] FP32 for inverse transformation
        - scale: float scale factor applied
        - padding: (pad_w, pad_h) tuple
    """
    orig_h, orig_w = img_rgb.shape[:2]

    # Calculate scale (don't upscale if scaleup=False)
    scale = min(target_size / orig_h, target_size / orig_w)
    if not scaleup:
        scale = min(scale, 1.0)

    # New dimensions after scaling
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    # Resize using cv2 (faster than PIL for numpy arrays)
    if scale != 1.0:
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = img_rgb

    # Calculate padding (center the image)
    pad_w = (target_size - new_w) / 2.0
    pad_h = (target_size - new_h) / 2.0

    # Compute integer padding (matches Ultralytics LetterBox rounding)
    top = round(pad_h - 0.1)
    bottom = round(pad_h + 0.1)
    left = round(pad_w - 0.1)
    right = round(pad_w + 0.1)

    # Add padding with gray border
    letterboxed = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )

    # Ensure exact size (rounding may cause 1px difference)
    if letterboxed.shape[:2] != (target_size, target_size):
        letterboxed = cv2.resize(letterboxed, (target_size, target_size))

    # Normalize to [0, 1]
    normalized = letterboxed.astype(np.float32) / 255.0

    # HWC -> CHW
    tensor = np.transpose(normalized, (2, 0, 1))

    # Create affine transformation matrix for inverse transform
    # Maps from letterboxed coords to original image coords
    # Forward: x_letterbox = x_orig * scale + pad_x  # noqa: ERA001
    # Inverse: x_orig = (x_letterbox - pad_x) / scale  # noqa: ERA001
    affine_matrix = np.array(
        [
            [scale, 0, left],
            [0, scale, top],
        ],
        dtype=np.float32,
    )

    return tensor, affine_matrix, scale, (pad_w, pad_h)


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

    # Resize shortest edge to target_size
    scale = target_size / min(orig_h, orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop to target_size x target_size
    start_x = (new_w - target_size) // 2
    start_y = (new_h - target_size) // 2
    cropped = resized[start_y : start_y + target_size, start_x : start_x + target_size]

    # Normalize to [0, 1]
    normalized = cropped.astype(np.float32) / 255.0

    # HWC -> CHW
    return np.transpose(normalized, (2, 0, 1))


def resize_hd_cpu(
    img_rgb: np.ndarray,
    max_size: int = 1920,
) -> np.ndarray:
    """
    Resize image so longest edge is at most max_size (don't upscale).

    Used for HD image preservation in face alignment - crops are taken
    from the highest resolution available for best recognition accuracy.

    Args:
        img_rgb: HWC, RGB, uint8 numpy array
        max_size: Maximum dimension for longest edge (default 1920)

    Returns:
        tensor: [3, H, W] FP32 normalized [0,1], where max(H, W) <= max_size
    """
    orig_h, orig_w = img_rgb.shape[:2]

    # Calculate scale (only downscale, never upscale)
    scale = min(max_size / max(orig_h, orig_w), 1.0)

    if scale < 1.0:
        new_w = round(orig_w * scale)
        new_h = round(orig_h * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        # Don't upscale - keep original
        resized = img_rgb

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # HWC -> CHW
    return np.transpose(normalized, (2, 0, 1))


def preprocess_single(
    image_bytes: bytes,
    yolo_size: int = 640,
    clip_size: int = 256,
    hd_max_size: int = 768,
) -> PreprocessResult:
    """
    Preprocess a single image for all model inputs.

    Decodes image and runs all preprocessing functions:
    - letterbox_cpu for YOLO detection
    - center_crop_cpu for MobileCLIP embedding
    - resize_hd_cpu for face alignment

    Args:
        image_bytes: Raw JPEG/PNG bytes
        yolo_size: YOLO input size (default 640)
        clip_size: CLIP input size (default 256)
        hd_max_size: Maximum HD dimension (default 1920)

    Returns:
        PreprocessResult with all tensors and metadata

    Raises:
        ValueError: If image cannot be decoded
    """
    # Decode image with cv2
    img_array = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if img_array is None:
        raise ValueError('Failed to decode image bytes')

    # Convert BGR (cv2 default) to RGB
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = img_rgb.shape[:2]

    # Run all preprocessing functions
    yolo_tensor, affine_matrix, scale, padding = letterbox_cpu(img_rgb, target_size=yolo_size)
    clip_tensor = center_crop_cpu(img_rgb, target_size=clip_size)
    hd_tensor = resize_hd_cpu(img_rgb, max_size=hd_max_size)

    return PreprocessResult(
        yolo_tensor=yolo_tensor,
        clip_tensor=clip_tensor,
        hd_tensor=hd_tensor,
        affine_matrix=affine_matrix,
        orig_shape=(orig_h, orig_w),
        scale=scale,
        padding=padding,
    )


def preprocess_batch(
    images_bytes: list[bytes],
    max_workers: int = 64,
    yolo_size: int = 640,
    clip_size: int = 256,
    hd_max_size: int = 768,
) -> list[PreprocessResult]:
    """
    Preprocess a batch of images in parallel using ThreadPoolExecutor.

    Useful for ingestion pipelines where many images need preprocessing.
    Thread-based parallelism works well because cv2 releases the GIL.

    Args:
        images_bytes: List of raw JPEG/PNG bytes
        max_workers: Maximum parallel threads (default 64)
        yolo_size: YOLO input size (default 640)
        clip_size: CLIP input size (default 256)
        hd_max_size: Maximum HD dimension (default 1920)

    Returns:
        List of PreprocessResult, one per image (preserves order)
        Failed images will have None in their position

    Note:
        Failed images are logged but don't stop processing.
        Check for None values in the returned list.
    """
    import logging

    logger = logging.getLogger(__name__)

    batch_size = len(images_bytes)
    if batch_size == 0:
        return []

    results: list[PreprocessResult | None] = [None] * batch_size

    def process_single_indexed(idx: int, img_bytes: bytes) -> tuple[int, PreprocessResult | None]:
        """Process a single image and return (index, result)."""
        try:
            result = preprocess_single(
                img_bytes,
                yolo_size=yolo_size,
                clip_size=clip_size,
                hd_max_size=hd_max_size,
            )
            return idx, result
        except Exception as e:
            logger.warning(f'Failed to preprocess image {idx}: {e}')
            return idx, None

    # Process in parallel
    workers = min(max_workers, batch_size)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_single_indexed, i, img) for i, img in enumerate(images_bytes)
        ]

        for future in futures:
            idx, result = future.result()
            results[idx] = result

    return results  # type: ignore[return-value]


def preprocess_for_triton(
    image_bytes: bytes,
    yolo_size: int = 640,
    clip_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, float, tuple[float, float], tuple[int, int]]:
    """
    Preprocess image for direct Triton inference with CPU preprocessing.

    Returns tensors with batch dimension added, ready for Triton InferInput.

    Args:
        image_bytes: Raw JPEG/PNG bytes
        yolo_size: YOLO input size (default 640)
        clip_size: CLIP input size (default 256)

    Returns:
        Tuple of:
        - yolo_input: [1, 3, 640, 640] FP32
        - clip_input: [1, 3, 256, 256] FP32
        - scale: float
        - padding: (pad_w, pad_h)
        - orig_shape: (height, width)

    Raises:
        ValueError: If image cannot be decoded
    """
    result = preprocess_single(image_bytes, yolo_size=yolo_size, clip_size=clip_size)

    # Add batch dimension for Triton
    yolo_input = np.expand_dims(result.yolo_tensor, axis=0)
    clip_input = np.expand_dims(result.clip_tensor, axis=0)

    return yolo_input, clip_input, result.scale, result.padding, result.orig_shape


def preprocess_batch_for_triton(
    images_bytes: list[bytes],
    max_workers: int = 64,
    yolo_size: int = 640,
    clip_size: int = 256,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[float],
    list[tuple[float, float]],
    list[tuple[int, int]],
]:
    """
    Preprocess batch of images for Triton batch inference.

    Returns stacked tensors ready for Triton InferInput.
    Failed images are excluded from the batch (returns smaller batch).

    Args:
        images_bytes: List of raw JPEG/PNG bytes
        max_workers: Maximum parallel threads (default 64)
        yolo_size: YOLO input size (default 640)
        clip_size: CLIP input size (default 256)

    Returns:
        Tuple of:
        - yolo_batch: [N, 3, 640, 640] FP32
        - clip_batch: [N, 3, 256, 256] FP32
        - scales: List[float] of length N
        - paddings: List[(pad_w, pad_h)] of length N
        - orig_shapes: List[(height, width)] of length N

    Raises:
        ValueError: If all images fail to decode
    """
    results = preprocess_batch(
        images_bytes,
        max_workers=max_workers,
        yolo_size=yolo_size,
        clip_size=clip_size,
    )

    # Filter out failed images
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        raise ValueError('All images failed to decode')

    # Stack tensors
    yolo_batch = np.stack([r.yolo_tensor for r in valid_results], axis=0)
    clip_batch = np.stack([r.clip_tensor for r in valid_results], axis=0)
    scales = [r.scale for r in valid_results]
    paddings = [r.padding for r in valid_results]
    orig_shapes = [r.orig_shape for r in valid_results]

    return yolo_batch, clip_batch, scales, paddings, orig_shapes
