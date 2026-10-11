"""Turn the rows a capture collected from the stack into the fixed-schema arrays.

Pure functions over plain Python rows and numpy: ground-truth class assignment from COCO
annotations, the integer crop rectangle, and the sorted, dtype-checked arrays of
``parity_store``. The HTTP side lives in ``parity_capture.py``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench import parity_metrics as pm, parity_store as ps


if TYPE_CHECKING:
    from pathlib import Path

GT_MIN_IOU = 0.5
GT_NONE = 'none'
GroundTruth = dict[str, tuple[np.ndarray, list[str]]]


def crop_rect(box_norm: list[float], width: int, height: int) -> list[int]:
    """The integer pixel rectangle a crop is cut with (the ingest crop rule: rounded, >= 1 px)."""
    x1 = max(0, round(box_norm[0] * width))
    y1 = max(0, round(box_norm[1] * height))
    return [
        x1,
        y1,
        max(x1 + 1, round(box_norm[2] * width)),
        max(y1 + 1, round(box_norm[3] * height)),
    ]


def _annotation_file(root: Path, split: str) -> Path:
    for candidate in (
        root / f'instances_{split}.json',
        root / 'annotations' / f'instances_{split}.json',
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'instances_{split}.json not found under {root}')


def load_gt(annotations_dir: Path, pin: dict[str, Any]) -> GroundTruth:
    """``{file_name: (boxes_norm (n, 4), class names)}`` for the pinned images, crowd boxes dropped."""
    rows = {(r['split'], r['image_id']): r for r in pin['images']}
    out: GroundTruth = {r['file_name']: (np.zeros((0, 4), np.float32), []) for r in pin['images']}
    for split in sorted({r['split'] for r in pin['images']}):
        doc = json.loads(_annotation_file(annotations_dir, split).read_text(encoding='utf-8'))
        names = {c['id']: c['name'] for c in doc['categories']}
        found: dict[str, list[tuple[list[float], str]]] = {}
        for ann in doc['annotations']:
            row = rows.get((split, ann['image_id']))
            if row is None or ann.get('iscrowd'):
                continue
            x, y, w, h = ann['bbox']
            box = [
                x / row['width'],
                y / row['height'],
                (x + w) / row['width'],
                (y + h) / row['height'],
            ]
            found.setdefault(row['file_name'], []).append((box, names[ann['category_id']]))
        for file_name, anns in found.items():
            out[file_name] = (np.array([a[0] for a in anns], np.float32), [a[1] for a in anns])
    return out


def assign_gt(boxes_norm: np.ndarray, gt: tuple[np.ndarray, list[str]]) -> list[str]:
    """Per box, the class of the best-IoU ground-truth annotation (IoU >= 0.5), else ``none``."""
    gt_boxes, gt_names = gt
    if len(boxes_norm) == 0 or len(gt_boxes) == 0:
        return [GT_NONE] * len(boxes_norm)
    iou = pm.box_iou(boxes_norm, gt_boxes)
    best = iou.argmax(axis=1)
    return [
        gt_names[j] if iou[i, j] >= GT_MIN_IOU else GT_NONE for i, j in enumerate(best.tolist())
    ]


def _str_array(values: list[str]) -> np.ndarray:
    return np.array(values, dtype=str) if values else np.zeros(0, dtype='U1')


def _vec_array(rows: list[list[float]], dim: int) -> np.ndarray:
    return np.array(rows, dtype=np.float16).reshape(len(rows), dim)


def build_arrays(
    items: list[dict[str, Any]],
    *,
    image_names: dict[str, str],
    image_dims: dict[str, tuple[int, int]],
    vectors: dict[str, list[float]],
    frame_vectors: dict[str, list[float]],
    gt: GroundTruth | None,
    vlm: dict[str, str],
    probes: dict[str, list[tuple[Any, ...]]],
) -> dict[str, np.ndarray]:
    """The schema arrays from API item rows, vectors read per id, ground truth and core probes.

    ``image_names`` maps an image id to the pinned file name. Items sort by file name then box,
    so two captures of the same set line up row for row where they agree.
    """
    dim = len(next(iter(vectors.values()), [])) or len(next(iter(frame_vectors.values()), []))
    keyed = sorted(
        ((image_names[it['image_id']], tuple(it['bbox_norm']), it) for it in items),
        key=lambda t: (t[0], t[1], t[2]['id']),
    )
    arrays = ps.empty_arrays(dim)
    if keyed:
        files = [k[0] for k in keyed]
        rows = [k[2] for k in keyed]
        boxes = np.array([k[1] for k in keyed], np.float32)
        gt_names = [''] * len(files)
        if gt:
            groups: dict[str, list[int]] = {}
            for i, name in enumerate(files):
                groups.setdefault(name, []).append(i)
            for name, idx in groups.items():
                for i, label in zip(idx, assign_gt(boxes[idx], gt[name]), strict=True):
                    gt_names[i] = label
        has_vec = [r['id'] in vectors for r in rows]
        arrays.update(
            item_id=_str_array([r['id'] for r in rows]),
            item_image=_str_array(files),
            item_box=boxes,
            item_score=np.array(
                [
                    r['detector_confidence']
                    if r.get('detector_confidence') is not None
                    else r['confidence']
                    for r in rows
                ],
                np.float32,
            ),
            item_class=_str_array(
                [r.get('detector_class_name') or r.get('class_name') or '' for r in rows]
            ),
            item_crop_rect=np.array(
                [crop_rect(list(k[1]), *image_dims[k[0]]) for k in keyed], np.int32
            ),
            item_cluster=np.array(
                [-1 if r.get('cluster_id') is None else r['cluster_id'] for r in rows], np.int32
            ),
            item_cluster_distance=np.array(
                [
                    np.nan if r.get('cluster_distance') is None else r['cluster_distance']
                    for r in rows
                ],
                np.float32,
            ),
            item_cluster_similarity=np.array(
                [
                    np.nan if r.get('cluster_similarity') is None else r['cluster_similarity']
                    for r in rows
                ],
                np.float32,
            ),
            item_cluster_core=np.array(
                [
                    -1 if r.get('cluster_is_core') is None else int(r['cluster_is_core'])
                    for r in rows
                ],
                np.int8,
            ),
            item_label=_str_array([r.get('class_name') or '' for r in rows]),
            item_label_source=_str_array([r.get('class_source') or '' for r in rows]),
            item_gt=_str_array(gt_names),
            item_vlm=_str_array([vlm.get(r['id'], '') for r in rows]),
            item_has_vec=np.array(has_vec, dtype=bool),
            item_vec=_vec_array([vectors.get(r['id']) or [0.0] * dim for r in rows], dim),
        )
    frame_files = sorted(set(image_names.values()))
    by_file = {image_names[i]: v for i, v in frame_vectors.items() if i in image_names}
    kept = [f for f in frame_files if f in by_file]
    arrays['frame_image'] = _str_array(kept)
    arrays['frame_vec'] = _vec_array([by_file[f] for f in kept], dim)
    clip = sorted(probes.get('clip', []), key=lambda r: r[0])
    arrays['clip_image'] = _str_array([r[0] for r in clip])
    arrays['clip_vec'] = _vec_array([r[1] for r in clip], len(clip[0][1]) if clip else 0)
    faces = sorted(probes.get('faces', []), key=lambda r: (r[0], tuple(r[1])))
    arrays['face_image'] = _str_array([r[0] for r in faces])
    arrays['face_box'] = np.array([r[1] for r in faces], np.float32).reshape(len(faces), 4)
    arrays['face_score'] = np.array([r[2] for r in faces], np.float32)
    arrays['face_vec'] = _vec_array([r[3] for r in faces], len(faces[0][3]) if faces else 0)
    ocr = sorted(probes.get('ocr', []), key=lambda r: (r[0], tuple(r[1])))
    arrays['ocr_image'] = _str_array([r[0] for r in ocr])
    arrays['ocr_box'] = np.array([r[1] for r in ocr], np.float32).reshape(len(ocr), 4)
    arrays['ocr_text'] = _str_array([r[2] for r in ocr])
    return arrays
