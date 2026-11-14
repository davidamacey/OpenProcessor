"""Load the stored embeddings of LFW, CALFW and CPLFW and build the pooled face table."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scripts.research.face_grouping import face_table as ft, quality as q


if TYPE_CHECKING:
    from collections.abc import Callable

SETS = ('lfw', 'calfw', 'cplfw')
PAIRS = {'lfw': 'pairs.txt', 'calfw': 'pairs_CALFW.txt', 'cplfw': 'pairs_CPLFW.txt'}


@dataclass(frozen=True)
class FaceSet:
    """One face per image (the detection nearest the centre); ``has`` marks images with a detection."""

    name: str
    basename: list[str]
    identity: np.ndarray
    has: np.ndarray
    n_faces: np.ndarray
    emb: np.ndarray
    detector_score: np.ndarray
    side_px: np.ndarray
    route_quality: np.ndarray
    pose: np.ndarray


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_set(eval_dir: Path, name: str) -> FaceSet:
    z = np.load(eval_dir / f'{name}.npz')
    n = len(z['files'])
    row, n_faces = ft.main_face(z['face_img'], z['box'], n)
    has = row >= 0
    pick = np.where(has, row, 0)
    box = z['box'][pick]
    emb = z['emb'][pick].astype(np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    side = np.maximum((box[:, 2] - box[:, 0]) * z['width'], (box[:, 3] - box[:, 1]) * z['height'])
    return FaceSet(
        name=name,
        basename=[Path(f).name for f in z['files']],
        identity=z['identity'],
        has=has,
        n_faces=n_faces,
        emb=emb,
        detector_score=z['score'][pick],
        side_px=side.astype(np.float32),
        route_quality=z['quality'][pick],
        pose=q.pose_offset(z['landmarks'][pick]),
    )


@dataclass(frozen=True)
class Pooled:
    """Faces of all sets that had a detection, sets concatenated in ``SETS`` order."""

    emb: np.ndarray
    identity: np.ndarray
    set_id: np.ndarray
    signals: dict[str, np.ndarray]  # higher is better for every signal
    row_of: dict[str, np.ndarray]  # per set: image index -> pooled row, -1 without a face


def pool(sets: dict[str, FaceSet]) -> Pooled:
    parts, rows, offset = [], {}, 0
    for sid, name in enumerate(SETS):
        s = sets[name]
        row = np.full(len(s.has), -1, dtype=np.int64)
        row[s.has] = offset + np.arange(int(s.has.sum()))
        offset += int(s.has.sum())
        rows[name] = row
        parts.append((sid, s))

    def cat(f: Callable[[FaceSet], np.ndarray]) -> np.ndarray:
        return np.concatenate([f(s)[s.has] for _, s in parts])

    return Pooled(
        emb=cat(lambda s: s.emb),
        identity=cat(lambda s: s.identity),
        set_id=np.concatenate([np.full(int(s.has.sum()), sid) for sid, s in parts]),
        signals={
            'detector_score': cat(lambda s: s.detector_score),
            'box_side_px': cat(lambda s: s.side_px),
            'route_quality': cat(lambda s: s.route_quality),
            'frontal_pose': -cat(lambda s: s.pose),
        },
        row_of=rows,
    )


def overlap_report(sets: dict[str, FaceSet]) -> dict[str, int]:
    names = {n: set(sets[n].identity[sets[n].has].tolist()) for n in SETS}
    count: dict[str, int] = {}
    for ident in set().union(*names.values()):
        k = sum(ident in names[n] for n in SETS)
        count[f'in_{k}_sets'] = count.get(f'in_{k}_sets', 0) + 1
    count['identities_total'] = len(set().union(*names.values()))
    count['lfw_and_calfw'] = len(names['lfw'] & names['calfw'])
    count['lfw_and_cplfw'] = len(names['lfw'] & names['cplfw'])
    count['calfw_and_cplfw'] = len(names['calfw'] & names['cplfw'])
    return count


def beds(pooled: Pooled) -> dict[str, np.ndarray]:
    """Row indexes (into ``pooled``) of the three clustering test beds."""
    lfw = np.flatnonzero(pooled.set_id == 0)
    size = dict(zip(*np.unique(pooled.identity, return_counts=True), strict=True))
    n_img = np.array([size[i] for i in pooled.identity])
    return {
        'bed1_lfw': lfw,
        'bed2_pooled': np.arange(len(pooled.emb)),
        'bed3_ge10_plus_singletons': np.flatnonzero((n_img >= 10) | (n_img == 1)),
    }
