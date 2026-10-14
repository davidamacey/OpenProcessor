"""
Affine matrix calculation for YOLO letterbox transformation.

Performance optimizations:
- @lru_cache for affine matrix calculation (same dimensions = same matrix)
- Fast JPEG header parsing (no full decode)
"""

from io import BytesIO

import numpy as np
from PIL import Image


def get_jpeg_dimensions_fast(image_bytes: bytes) -> tuple[int, int]:
    """
    Get JPEG dimensions from header without full decode.

    This is ~100x faster than PIL.Image.open().size for large images.
    Falls back to PIL for non-JPEG or corrupted headers.

    Args:
        image_bytes: Raw JPEG/PNG bytes

    Returns:
        (width, height) tuple
    """
    # Try fast JPEG header parse first
    if len(image_bytes) > 2 and image_bytes[0:2] == b'\xff\xd8':
        try:
            # JPEG format - parse SOF markers
            pos = 2
            while pos < len(image_bytes) - 8:
                if image_bytes[pos] != 0xFF:
                    break
                marker = image_bytes[pos + 1]

                # SOF markers (Start Of Frame) contain dimensions
                if marker in (0xC0, 0xC1, 0xC2):  # SOF0, SOF1, SOF2
                    height = (image_bytes[pos + 5] << 8) | image_bytes[pos + 6]
                    width = (image_bytes[pos + 7] << 8) | image_bytes[pos + 8]
                    return width, height

                # Skip to next marker
                if marker in {0xD8, 0xD9}:  # SOI, EOI
                    pos += 2
                elif marker == 0xFF:
                    pos += 1
                else:
                    length = (image_bytes[pos + 2] << 8) | image_bytes[pos + 3]
                    pos += 2 + length
        except (IndexError, ValueError):
            pass  # Fall back to PIL

    # Fallback: Use PIL (handles all formats)
    img = Image.open(BytesIO(image_bytes))
    return img.size  # (width, height)


def inverse_letterbox_coords(
    boxes: np.ndarray,
    orig_shape: tuple[int, int],
    scale: float,
    padding: tuple[float, float],
    input_size: int = 640,
) -> np.ndarray:
    """
    Convert bounding boxes from letterboxed space to original image coordinates.

    EfficientNMS with normalize_boxes=True outputs coords normalized to the 640x640
    letterboxed input [0,1]. This function applies inverse letterbox transformation
    to convert them to [0,1] normalized coordinates relative to original image dimensions.

    This matches Ultralytics PyTorch which uses boxes.xyxyn for
    original-image-normalized output.

    Args:
        boxes: np.ndarray [N, 4] in XYXY format, normalized to letterbox [0,1]
        orig_shape: Tuple (height, width) of original image
        scale: Letterbox scale factor
        padding: Tuple (pad_x, pad_y) letterbox padding
        input_size: Letterbox target size (default 640)

    Returns:
        np.ndarray [N, 4] in XYXY format, normalized to original image [0,1]
    """
    if len(boxes) == 0:
        return boxes

    orig_h, orig_w = orig_shape
    pad_x, pad_y = padding

    # Work on copy to avoid modifying input
    boxes_out = boxes.copy().astype(np.float64)

    # Auto-detect if boxes are normalized [0,1] or pixel coordinates
    # If max value > 1.0, boxes are already in pixel coords (640x640 space)
    is_pixel_coords = float(boxes_out.max()) > 1.0

    # 1. Scale letterbox-normalized [0,1] to letterbox pixels (skip if already pixels)
    if not is_pixel_coords:
        boxes_out *= input_size

    # 2. Remove padding and scale to get original pixels
    boxes_out[:, 0] = (boxes_out[:, 0] - pad_x) / scale  # x1
    boxes_out[:, 1] = (boxes_out[:, 1] - pad_y) / scale  # y1
    boxes_out[:, 2] = (boxes_out[:, 2] - pad_x) / scale  # x2
    boxes_out[:, 3] = (boxes_out[:, 3] - pad_y) / scale  # y2

    # 3. Normalize to original image dimensions
    boxes_out[:, 0] /= orig_w
    boxes_out[:, 2] /= orig_w
    boxes_out[:, 1] /= orig_h
    boxes_out[:, 3] /= orig_h

    # 4. Clip to [0, 1]
    boxes_out = np.clip(boxes_out, 0.0, 1.0)

    return boxes_out.astype(np.float32)


# COCO Class Names (80 classes)
COCO_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush',
}


def format_detections_from_triton(
    result: dict, input_size: int = 640, model_name: str | None = None
) -> list:
    """
    Format Triton EfficientNMS detections with original-image-normalized coordinates.

    Shared utility for TensorRT End2End models with EfficientNMS.
    Applies inverse letterbox transformation to match Ultralytics boxes.xyxyn format.

    Args:
        result: Inference result dict with:
            - boxes: np.ndarray [N, 4] letterbox-normalized
            - scores: np.ndarray [N]
            - classes: np.ndarray [N]
            - orig_shape: (height, width) optional
            - scale: float optional
            - padding: (pad_x, pad_y) optional
        model_name: Which Triton model produced ``result`` -- resolves
            ``class_name`` from that model's own labels (see
            :mod:`src.utils.class_names`) instead of always assuming the
            stock 80-class COCO vocabulary (F-42, fresh-start E2E
            findings 2026-09-25). ``None`` keeps the historical
            COCO-only behavior for callers that genuinely never see
            anything but the stock detector.

    Returns:
        List of detection dicts with x1, y1, x2, y2 normalized to original image
    """
    boxes = result['boxes']
    scores = result['scores']
    classes = result['classes']

    if len(boxes) == 0:
        return []

    # Apply inverse letterbox if transformation params available
    orig_shape = result.get('orig_shape')
    scale = result.get('scale')
    padding = result.get('padding')

    if orig_shape is not None and scale is not None and padding is not None:
        boxes = inverse_letterbox_coords(boxes, orig_shape, scale, padding, input_size)

    if model_name is not None:
        from src.utils.class_names import get_class_names

        class_names = get_class_names(model_name)
    else:
        class_names = COCO_CLASSES

    return [
        {
            'x1': float(boxes[i, 0]),
            'y1': float(boxes[i, 1]),
            'x2': float(boxes[i, 2]),
            'y2': float(boxes[i, 3]),
            'confidence': float(scores[i]),
            'class': int(classes[i]),
            'class_name': class_names.get(int(classes[i]), f'class_{int(classes[i])}'),
        }
        for i in range(len(boxes))
    ]
