"""
Face alignment for ArcFace using similarity transform (Umeyama algorithm).

Aligns detected face landmarks to the ArcFace reference template, producing
a 112x112 aligned crop suitable for face recognition. This is the standard
preprocessing used by InsightFace and production face recognition systems.

The similarity transform (4 DoF: rotation, scale, tx, ty) maps the 5 detected
landmarks to fixed reference positions, normalizing for head pose, scale, and
position.

Landmark order: left_eye, right_eye, nose, left_mouth, right_mouth

Reference: InsightFace face_align.py
"""

import cv2
import numpy as np


# ArcFace reference landmarks on 112x112 canvas
# These are the target positions that detected landmarks are warped to
ARCFACE_REF = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)

ARCFACE_SIZE = 112


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Estimate similarity transform using Umeyama algorithm.

    Finds optimal rotation, scale, and translation to map src to dst
    in least-squares sense.

    Args:
        src: [N, 2] source points (detected landmarks)
        dst: [N, 2] destination points (reference template)

    Returns:
        [2, 3] affine matrix for cv2.warpAffine
    """
    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Center points
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Covariance
    A = dst_demean.T @ src_demean / num

    # SVD
    U, S, Vt = np.linalg.svd(A)

    # Reflection correction
    d = np.ones(dim, dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)

    # Variance of source
    src_var = src_demean.var(axis=0).sum()

    # Scale
    scale = (S * d).sum() / src_var

    # Rotation
    T[:dim, :dim] = U @ np.diag(d) @ Vt
    T[:dim, :dim] *= scale

    # Translation
    T[:dim, dim] = dst_mean - T[:dim, :dim] @ src_mean

    return T[:2, :]


def align_face(
    img: np.ndarray,
    landmarks: np.ndarray,
    image_size: int = ARCFACE_SIZE,
) -> np.ndarray:
    """
    Align a single face using 5-point landmarks.

    Args:
        img: BGR image, shape [H, W, 3]
        landmarks: [5, 2] landmark coordinates in pixel space
        image_size: Output size (default 112 for ArcFace)

    Returns:
        Aligned face crop, shape [image_size, image_size, 3], BGR
    """
    # Scale reference if non-standard size
    if image_size == ARCFACE_SIZE:
        dst = ARCFACE_REF
    else:
        ratio = float(image_size) / ARCFACE_SIZE
        dst = ARCFACE_REF * ratio

    # Compute similarity transform
    M = _umeyama(landmarks.astype(np.float64), dst.astype(np.float64))

    # Warp
    return cv2.warpAffine(img, M.astype(np.float32), (image_size, image_size), borderValue=0.0)


def align_faces_batch(
    img: np.ndarray,
    landmarks_batch: np.ndarray,
    image_size: int = ARCFACE_SIZE,
) -> np.ndarray:
    """
    Align multiple faces from a single image.

    Args:
        img: BGR image, shape [H, W, 3]
        landmarks_batch: [N, 5, 2] landmark coordinates in pixel space
        image_size: Output size (default 112 for ArcFace)

    Returns:
        Aligned faces, shape [N, image_size, image_size, 3], BGR
    """
    if len(landmarks_batch) == 0:
        return np.array([])

    aligned = np.zeros((len(landmarks_batch), image_size, image_size, 3), dtype=np.uint8)
    for i, lmk in enumerate(landmarks_batch):
        aligned[i] = align_face(img, lmk, image_size)

    return aligned


def preprocess_for_arcface(aligned_faces: np.ndarray) -> np.ndarray:
    """
    Convert aligned face crops to ArcFace input format.

    Args:
        aligned_faces: [N, 112, 112, 3] BGR uint8 aligned crops

    Returns:
        [N, 3, 112, 112] FP32 normalized for ArcFace
        Normalization: (pixel - 127.5) / 128.0, BGR->RGB
    """
    if len(aligned_faces) == 0:
        return np.array([], dtype=np.float32)

    # BGR -> RGB
    faces_rgb = aligned_faces[:, :, :, ::-1].copy()

    # HWC -> CHW
    faces_chw = faces_rgb.transpose(0, 3, 1, 2).astype(np.float32)

    # Normalize for ArcFace
    return (faces_chw - 127.5) / 128.0
