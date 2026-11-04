"""CPU preprocessing for the Triton client's detector and MobileCLIP paths.

Pure array functions: they take pixels or encoded bytes and return the tensors
:class:`src.clients.triton_client.TritonClient` sends, with no Triton or network access.
"""

import io

import cv2
import numpy as np
from PIL import Image

from src.utils.cpu_preprocess import center_crop_cpu


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
    """CPU preprocessing for MobileCLIP (256x256, center crop), RGB uint8 in, ``[1,3,256,256]`` out."""
    return center_crop_cpu(img_array, target_size=256)[np.newaxis, ...]


def preprocess_for_mobileclip(image_bytes: bytes) -> np.ndarray:
    """Preprocess encoded image bytes for MobileCLIP (256x256, BILINEAR, center crop)."""
    rgb = np.asarray(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    return preprocess_clip_cpu(rgb)
