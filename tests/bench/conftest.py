"""Synthetic parity captures for the bench tests: planted structure, fixed seed, no stack."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from scripts.bench import parity_store as ps, select_baseline_set as sel


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

DIM = 16
N_IMAGES = 60
PER_IMAGE = 3
N_CLUSTERS = 4


def _synthetic(seed: int = 0) -> dict[str, np.ndarray]:
    """``N_IMAGES`` images x ``PER_IMAGE`` boxes; vectors follow ``N_CLUSTERS`` planted centres."""
    rng = np.random.default_rng(seed)
    n = N_IMAGES * PER_IMAGE
    arrays = ps.empty_arrays(DIM)
    centres = rng.normal(size=(N_CLUSTERS, DIM))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    cluster = rng.integers(0, N_CLUSTERS, size=n)
    vec = centres[cluster] + rng.normal(scale=0.02, size=(n, DIM))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    slot = np.tile(np.arange(PER_IMAGE), N_IMAGES)
    x1 = 0.05 + 0.3 * slot
    box = np.stack([x1, np.full(n, 0.1), x1 + 0.25, np.full(n, 0.6)], axis=1)
    classes = np.array(['cat', 'dog', 'car', 'bus'])
    clip_n, text_n = 10, 12
    clip = rng.normal(size=(clip_n, 8))
    clip /= np.linalg.norm(clip, axis=1, keepdims=True)
    face_vec = rng.normal(size=(text_n, 8))
    face_vec /= np.linalg.norm(face_vec, axis=1, keepdims=True)
    vlm = np.array([''] * n, dtype='U8')
    vlm[:30] = classes[cluster[:30]]
    arrays.update(
        item_id=np.array([f'it{i}' for i in range(n)]),
        item_image=np.repeat([f'img{i:03d}.jpg' for i in range(N_IMAGES)], PER_IMAGE),
        item_box=box.astype(np.float32),
        item_score=rng.uniform(0.5, 0.99, size=n).astype(np.float32),
        item_class=classes[cluster],
        item_crop_rect=(box * 100).astype(np.int32),
        item_cluster=cluster.astype(np.int32),
        item_cluster_distance=rng.uniform(0, 1, size=n).astype(np.float32),
        item_cluster_similarity=rng.uniform(0, 1, size=n).astype(np.float32),
        item_cluster_core=np.ones(n, dtype=np.int8),
        item_label=classes[cluster],
        item_label_source=np.array(['detector'] * n),
        item_gt=classes[cluster],
        item_vlm=vlm,
        item_has_vec=np.ones(n, dtype=bool),
        item_vec=vec.astype(np.float16),
        frame_image=np.array([f'img{i:03d}.jpg' for i in range(N_IMAGES)]),
        frame_vec=vec[::PER_IMAGE].astype(np.float16),
        clip_image=np.array([f'img{i:03d}.jpg' for i in range(clip_n)]),
        clip_vec=clip.astype(np.float16),
        face_image=np.array([f'img{i:03d}.jpg' for i in range(text_n)]),
        face_box=np.tile(np.array([[0.2, 0.2, 0.4, 0.5]], np.float32), (text_n, 1)),
        face_score=np.full(text_n, 0.9, np.float32),
        face_vec=face_vec.astype(np.float16),
        ocr_image=np.array([f'img{i:03d}.jpg' for i in range(text_n)]),
        ocr_box=np.tile(np.array([[0.1, 0.7, 0.9, 0.8]], np.float32), (text_n, 1)),
        ocr_text=np.array([f'TEXT {i}' for i in range(text_n)]),
    )
    return arrays


def _meta(tag: str, engine: str, manifest: str = 'abc123') -> dict[str, Any]:
    return {
        'tag': tag,
        'created_at': '2026-10-10T00:00:00+00:00',
        'manifest': {'name': 'm.json', 'sha256': manifest, 'count': N_IMAGES, 'seed': 1},
        'engine': {'label': engine},
        'environment': {'git_sha': 'deadbeef'},
        'project': 'p',
        'sections': list(ps.SECTIONS),
    }


@pytest.fixture
def make_capture() -> Callable[..., ps.Capture]:
    def build(
        tag: str = 'ref',
        *,
        engine: str = 'fp32',
        seed: int = 0,
        manifest: str = 'abc123',
        mutate: Callable[[dict[str, np.ndarray]], None] | None = None,
    ) -> ps.Capture:
        arrays = _synthetic(seed)
        if mutate:
            mutate(arrays)
        return ps.Capture(_meta(tag, engine, manifest), arrays)

    return build


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    from PIL import Image

    cands = []
    for i in range(10):
        p = tmp_path / f'{i}.jpg'
        Image.new('RGB', (400, 400)).save(p, 'JPEG')
        cands.append(sel.Candidate(p, p.stat().st_size, 'unit'))
    out = tmp_path / 'm.txt'
    sel.write_manifest(out, cands, seed=1, today='2026-10-03')
    return out
