"""Per-image face tables: pick the face a centre-cropped set is about, parse the pairs files."""

from __future__ import annotations

import numpy as np


def main_face(
    face_img: np.ndarray, box: np.ndarray, n_images: int
) -> tuple[np.ndarray, np.ndarray]:
    """Row of each image's detection nearest the image centre (-1 when none) and the face counts.

    ``box`` is the normalised ``[x1, y1, x2, y2]`` of each detected face; ``face_img`` names its image.
    """
    n_faces = np.bincount(face_img, minlength=n_images)
    centre_dist = np.hypot((box[:, 0] + box[:, 2]) / 2 - 0.5, (box[:, 1] + box[:, 3]) / 2 - 0.5)
    chosen = np.full(n_images, -1, dtype=np.int64)
    best = np.full(n_images, np.inf)
    for row in range(len(face_img)):
        img = face_img[row]
        if centre_dist[row] < best[img]:
            best[img], chosen[img] = centre_dist[row], row
    return chosen, n_faces


def parse_lfw_pairs(text: str) -> list[tuple[str, str, int]]:
    """``pairs.txt`` of LFW as (file_a, file_b, same) rows; the header line is skipped."""
    out: list[tuple[str, str, int]] = []
    for line in text.splitlines():
        f = line.split()
        if len(f) == 3:
            out.append((f'{f[0]}_{int(f[1]):04d}.jpg', f'{f[0]}_{int(f[2]):04d}.jpg', 1))
        elif len(f) == 4:
            out.append((f'{f[0]}_{int(f[1]):04d}.jpg', f'{f[2]}_{int(f[3]):04d}.jpg', 0))
    return out


def parse_cross_pairs(text: str) -> list[tuple[str, str, int]]:
    """CALFW and CPLFW pairs files: two ``<file> <label>`` lines make one pair."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    return [(a[0], b[0], int(a[1])) for a, b in zip(rows[0::2], rows[1::2], strict=True)]


def fold_ids(n_pairs: int, folds: int = 10) -> np.ndarray:
    """The standard protocol's ten contiguous folds (600 pairs each for the 6,000-pair files)."""
    return np.arange(n_pairs) // (n_pairs // folds)
