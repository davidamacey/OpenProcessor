"""Where each pixel of a recogniser input comes from in the full-resolution frame.

PaddleOCR cuts a text line in two steps (PaddleX release/3.3 ``crop_image_regions.py``
``get_rotate_crop_image``, then ``text_recognition/processors.py`` ``resize_norm_img``):
``cv2.warpPerspective`` of the quadrilateral to a ``cw x ch`` rectangle (``cw``, ``ch`` the longer
of each pair of opposite sides, truncated to ints), a quarter turn (``np.rot90``) when
``ch / cw >= 1.5``, then ``cv2.resize`` to the batch's height with the width the aspect ratio gives.
The kernel samples the frame once per recogniser pixel, so the three maps are composed here into
one 3 x 3 projective map from a recogniser pixel ``(u, v)`` to a frame pixel ``(x, y)``, both in
OpenCV's pixel-index coordinates (a pixel's centre is its integer index).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from frame_engine.ocr.batching import REC_HEIGHT


def crop_dims(quad: np.ndarray) -> tuple[int, int]:
    """``(cw, ch)`` of ``get_rotate_crop_image``; the quadrilateral is ``[4, 2]`` (top-left,
    top-right, bottom-right, bottom-left)."""
    q = np.asarray(quad, np.float32)
    cw = int(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3])))
    ch = int(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
    return cw, ch


def is_tall(cw: int, ch: int) -> bool:
    return ch * 1.0 / cw >= 1.5


def rec_ratio(quad: np.ndarray) -> float | None:
    """Width over height of the recogniser's crop (after the quarter turn), ``None`` for a quad too
    small to cut (a side under one pixel)."""
    cw, ch = crop_dims(quad)
    if cw < 1 or ch < 1:
        return None
    return ch / cw if is_tall(cw, ch) else cw / ch


def rec_map(quad: np.ndarray, width: int) -> np.ndarray:
    """The ``[3, 3]`` float64 map from recogniser pixel ``(u, v, 1)`` to frame ``(x, y, w)``
    (divide by ``w``) for a crop resized to ``width x 48``."""
    q = np.asarray(quad, np.float32)
    cw, ch = crop_dims(quad)
    std = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32)
    m_inv = np.linalg.inv(np.asarray(cv2.getPerspectiveTransform(q, std), np.float64))
    if is_tall(cw, ch):  # np.rot90: rotated[i, j] = crop[j, cw - 1 - i]; shape (cw, ch)
        w_rot, h_rot = ch, cw
        # crop x = cw - 1 - y', crop y = x'
        to_crop = np.array([[0.0, -1.0, cw - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    else:
        w_rot, h_rot = cw, ch
        to_crop = np.eye(3)
    sx, sy = w_rot / width, h_rot / REC_HEIGHT  # cv2.resize INTER_LINEAR: centres (i + .5) * s - .5
    resize = np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]])
    return np.asarray(m_inv @ to_crop @ resize, np.float64)


def footprint(g: np.ndarray, width: int) -> tuple[float, float]:
    """Frame pixels one recogniser pixel spans along ``u`` and along ``v`` (at the crop's centre)."""
    u, v = width / 2.0, REC_HEIGHT / 2.0

    def at(du: float, dv: float) -> np.ndarray:
        p = g @ np.array([u + du, v + dv, 1.0])
        return np.asarray(p[:2] / p[2], np.float64)

    return float(np.linalg.norm(at(0.5, 0) - at(-0.5, 0))), float(
        np.linalg.norm(at(0, 0.5) - at(0, -0.5))
    )


def taps_for(g: np.ndarray, width: int, cap: int) -> tuple[int, int]:
    """Samples per axis inside one recogniser pixel: the frame pixels it spans, rounded up, at
    most ``cap`` (``cap`` 1 reads the frame at the pixel centre alone, as ``cv2.resize`` does)."""
    su, sv = footprint(g, width)
    return max(1, min(cap, math.ceil(su - 1e-3))), max(1, min(cap, math.ceil(sv - 1e-3)))
