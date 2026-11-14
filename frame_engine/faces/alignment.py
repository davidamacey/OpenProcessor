"""Five-point alignment geometry (numpy): the similarity transform to the ArcFace template and what
the GPU warp needs to sample it.

``estimate`` is the reference's ``face_align.estimate_norm`` for a 112-pixel output: scikit-image
``SimilarityTransform.estimate`` (Umeyama's least-squares similarity, scale estimated) from the
five landmarks to ``ARCFACE_DST``. The engine image carries no scikit-image, so the Umeyama solve
is written out here, as ``skimage.transform._geometric._umeyama`` does it.

The warp (``kernels.WarpKernel``) reads the source at ``inverse`` applied to each output pixel
centre, as ``cv2.warpAffine`` does. The reference warps with plain bilinear sampling, which
aliases when a large face is reduced to 112 pixels; the GPU warp averages ``samples`` x
``samples`` bilinear taps spread over each output pixel's footprint in the source (an area
filter, OpenCV's advice for shrinking), and is the reference's bilinear warp exactly when the
face is not reduced (``samples == 1``).
"""

from __future__ import annotations

import math

import numpy as np


ARCFACE_SIZE = 112
#: The ArcFace template (InsightFace ``utils/face_align.py`` ``arcface_dst``): left eye, right eye,
#: nose tip, left and right mouth corners, in 112 x 112 output pixels.
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)
#: Upper bound of the per-axis tap count (a face 16 x the output side or larger is sampled 16 x 16).
MAX_SAMPLES = 16


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The ``3 x 3`` similarity (rotation, uniform scale, translation) that maps ``src`` onto
    ``dst`` in the least-squares sense (Umeyama 1991, eq. 40-42); NaNs when ``src`` is degenerate."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    num, dim = src.shape
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_demean, dst_demean = src - src_mean, dst - dst_mean
    a = dst_demean.T @ src_demean / num
    d = np.ones(dim)
    if np.linalg.det(a) < 0:
        d[dim - 1] = -1
    t = np.eye(dim + 1)
    u, s, v = np.linalg.svd(a)
    rank = np.linalg.matrix_rank(a)
    if rank == 0:
        return np.full((dim + 1, dim + 1), np.nan)
    if rank == dim - 1:
        if np.linalg.det(u) * np.linalg.det(v) > 0:
            t[:dim, :dim] = u @ v
        else:
            keep = d[dim - 1]
            d[dim - 1] = -1
            t[:dim, :dim] = u @ np.diag(d) @ v
            d[dim - 1] = keep
    else:
        t[:dim, :dim] = u @ np.diag(d) @ v
    scale = 1.0 / src_demean.var(axis=0).sum() * (s @ d)
    t[:dim, dim] = dst_mean - scale * (t[:dim, :dim] @ src_mean.T)
    t[:dim, :dim] *= scale
    return t


def estimate(landmarks: np.ndarray) -> np.ndarray:
    """The ``2 x 3`` forward matrix (source pixels to the 112 output) of one face's ``[5, 2]``
    landmarks, the reference's ``estimate_norm(lmk, 112)``."""
    return umeyama(landmarks, ARCFACE_DST)[:2, :]


def inverse(forward: np.ndarray) -> np.ndarray:
    """The ``2 x 3`` map from output pixels back to source pixels."""
    full = np.vstack([forward, [0.0, 0.0, 1.0]])
    return np.linalg.inv(full)[:2, :]


def samples(inv: np.ndarray) -> int:
    """Taps per axis for one face: the source pixels one output pixel spans, rounded up."""
    span = math.sqrt(abs(float(np.linalg.det(inv[:, :2]))))
    if not math.isfinite(span):
        return 1
    return min(MAX_SAMPLES, max(1, math.ceil(span - 1e-6)))


def warp_params(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For ``[n, 5, 2]`` landmarks: the inverse maps ``[n, 6]`` (float32), the tap counts ``[n]``
    (int32) and which faces have a usable transform ``[n]`` (bool; a degenerate landmark set
    has none and is not embedded)."""
    n = len(landmarks)
    inv = np.zeros((n, 6), np.float32)
    taps = np.ones(n, np.int32)
    ok = np.zeros(n, np.bool_)
    for i, lmk in enumerate(np.asarray(landmarks, np.float64).reshape(n, 5, 2)):
        fwd = estimate(lmk)
        if not np.isfinite(fwd).all() or abs(np.linalg.det(fwd[:, :2])) < 1e-12:
            continue
        m = inverse(fwd)
        inv[i] = m.reshape(-1)
        taps[i] = samples(m)
        ok[i] = True
    return inv, taps, ok
