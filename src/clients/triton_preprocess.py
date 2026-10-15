"""CPU preprocessing for the Triton client's detector and MobileCLIP paths.

Pure array functions: they take pixels or encoded bytes and return the tensors
:class:`src.clients.triton_client.TritonClient` sends, with no Triton or network access.
"""

import io

import cv2
import numpy as np
from PIL import Image


def preprocess_yolo_cpu(img_array: np.ndarray, target_size: int) -> tuple[np.ndarray, float, tuple]:
    """CPU letterbox preprocessing for YOLO at ``target_size``."""
    orig_h, orig_w = img_array.shape[:2]

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

    transposed = np.transpose(letterboxed, (2, 0, 1))
    batched = np.expand_dims(transposed, axis=0)

    return batched, scale, (pad_w, pad_h)


def preprocess_clip_cpu(img_array: np.ndarray) -> np.ndarray:
    """CPU preprocessing for MobileCLIP (256x256, center crop)."""
    orig_h, orig_w = img_array.shape[:2]
    target_size = 256

    # round() + a max(target_size, ...) floor -- see the matching
    # fix/comment in src.utils.cpu_preprocess.center_crop_cpu. Without
    # it, float truncation via int() could undershoot target_size by one
    # px and produce a smaller-than-256 crop instead of erroring.
    scale = target_size / min(orig_h, orig_w)
    new_w = max(target_size, round(orig_w * scale))
    new_h = max(target_size, round(orig_h * scale))
    resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    start_x = max(0, (new_w - target_size) // 2)
    start_y = max(0, (new_h - target_size) // 2)
    cropped = resized[start_y : start_y + target_size, start_x : start_x + target_size]

    normalized = cropped.astype(np.float32) / 255.0
    transposed = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(transposed, axis=0)


def preprocess_for_mobileclip(image_bytes: bytes) -> np.ndarray:
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
