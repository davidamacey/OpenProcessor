"""The parity capture file format: one JSON document and one compressed NPZ per capture.

``capture.json`` holds the schema number, the run metadata (hardware, git sha, image digests,
engine and build record, dataset manifest hash) and the shape and dtype of every array, plus the
SHA-256 of ``arrays.npz``. ``arrays.npz`` holds the per-stage outputs, every array always present
(zero rows when a section was not captured). Vectors are float16 and boxes float32; strings are
unicode. Large files stay outside git; only the small comparison summaries are committed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = 1
META_FILE = 'capture.json'
ARRAYS_FILE = 'arrays.npz'
REQUIRED_META = ('tag', 'created_at', 'manifest', 'engine', 'environment', 'sections')
SECTIONS = ('items', 'frames', 'clusters', 'vlm', 'clip', 'faces', 'ocr')

# name -> (dtype, ndim). 'U' means any unicode string dtype. The prefix is the row group.
ARRAY_SPEC: dict[str, tuple[str, int]] = {
    # one row per stored item (a detector box with its crop)
    'item_id': ('U', 1),
    'item_image': ('U', 1),
    'item_box': ('float32', 2),
    'item_score': ('float32', 1),
    'item_class': ('U', 1),
    'item_crop_rect': ('int32', 2),
    'item_cluster': ('int32', 1),
    'item_cluster_distance': ('float32', 1),
    'item_cluster_similarity': ('float32', 1),
    'item_cluster_core': ('int8', 1),
    'item_label': ('U', 1),
    'item_label_source': ('U', 1),
    'item_gt': ('U', 1),
    'item_vlm': ('U', 1),
    'item_has_vec': ('bool', 1),
    'item_vec': ('float16', 2),
    # one row per image: the whole-frame vector
    'frame_image': ('U', 1),
    'frame_vec': ('float16', 2),
    # core-route probes on a sample of images
    'clip_image': ('U', 1),
    'clip_vec': ('float16', 2),
    'face_image': ('U', 1),
    'face_box': ('float32', 2),
    'face_score': ('float32', 1),
    'face_vec': ('float16', 2),
    'ocr_image': ('U', 1),
    'ocr_box': ('float32', 2),
    'ocr_text': ('U', 1),
}


class CaptureError(ValueError):
    """A capture is malformed, tampered with, or from another schema."""


@dataclass
class Capture:
    meta: dict[str, Any]
    arrays: dict[str, np.ndarray]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def empty_arrays(dim: int, clip_dim: int = 0, face_dim: int = 0) -> dict[str, np.ndarray]:
    """Every array with zero rows; vector widths are ``dim`` (PE), ``clip_dim`` and ``face_dim``."""
    widths = {'item_vec': dim, 'frame_vec': dim, 'clip_vec': clip_dim, 'face_vec': face_dim}
    out: dict[str, np.ndarray] = {}
    for name, (dtype, ndim) in ARRAY_SPEC.items():
        shape = (0,) if ndim == 1 else (0, widths.get(name, 4))
        out[name] = np.zeros(shape, dtype='U1' if dtype == 'U' else dtype)
    return out


def validate(arrays: dict[str, np.ndarray]) -> None:
    missing = sorted(set(ARRAY_SPEC) - set(arrays))
    extra = sorted(set(arrays) - set(ARRAY_SPEC))
    if missing or extra:
        raise CaptureError(f'array set differs from the schema: missing {missing}, extra {extra}')
    rows: dict[str, set[int]] = {}
    for name, (dtype, ndim) in ARRAY_SPEC.items():
        arr = arrays[name]
        kind_ok = arr.dtype.kind == 'U' if dtype == 'U' else arr.dtype == np.dtype(dtype)
        if not kind_ok or arr.ndim != ndim:
            raise CaptureError(
                f'{name}: expected {dtype} with {ndim} dims, got {arr.dtype} {arr.shape}'
            )
        rows.setdefault(name.split('_', 1)[0], set()).add(len(arr))
    ragged = sorted(group for group, sizes in rows.items() if len(sizes) > 1)
    if ragged:
        raise CaptureError(f'row counts differ inside the {ragged} array group')


def _counts(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    return {
        'items': len(arrays['item_id']),
        'frames': len(arrays['frame_image']),
        'clip': len(arrays['clip_image']),
        'faces': len(arrays['face_image']),
        'ocr': len(arrays['ocr_image']),
    }


def save_capture(out_dir: Path, capture: Capture) -> None:
    """Write ``capture.json`` and ``arrays.npz`` into ``out_dir`` (created if needed)."""
    absent = [k for k in REQUIRED_META if k not in capture.meta]
    if absent:
        raise CaptureError(f'capture metadata is missing {absent}')
    validate(capture.arrays)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / ARRAYS_FILE
    np.savez_compressed(npz, **capture.arrays)  # type: ignore[arg-type]  # mypy reads **dict as the allow_pickle flag
    doc = {
        **capture.meta,
        'schema': SCHEMA,
        'counts': _counts(capture.arrays),
        'arrays': {
            k: {'dtype': str(v.dtype), 'shape': list(v.shape)} for k, v in capture.arrays.items()
        },
        'npz_sha256': sha256_file(npz),
    }
    (out_dir / META_FILE).write_text(
        json.dumps(doc, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )


def load_capture(directory: Path) -> Capture:
    """Read and verify a capture; raises :class:`CaptureError` on any inconsistency."""
    meta_path, npz_path = directory / META_FILE, directory / ARRAYS_FILE
    if not meta_path.is_file() or not npz_path.is_file():
        raise CaptureError(f'{directory}: capture.json or arrays.npz is missing')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('schema') != SCHEMA:
        raise CaptureError(f'{directory}: schema {meta.get("schema")}, this tool reads {SCHEMA}')
    if meta.get('npz_sha256') != sha256_file(npz_path):
        raise CaptureError(f'{directory}: arrays.npz sha256 differs from capture.json')
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    validate(arrays)
    return Capture(meta, arrays)
