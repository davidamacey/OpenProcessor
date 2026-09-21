"""Side-by-side comparison of a freshly trained run against the incumbent.

After a run finishes, the trainer can evaluate whichever model is *currently*
served by Triton on the same frozen test split the candidate was scored on, and
emit a per-class delta. The frontend renders that as a "does this beat what we
serve today?" panel.

This is **informational, not a promote gate** -- the real gate lives in
``src/services/training/triton_promote.py``. The evaluator here is deliberately
lightweight (greedy IoU matching at a fixed score threshold, no 101-point COCO
interpolation), and it samples at most
:data:`COMPARE_MAX_IMAGES` test images so a 50k holdout doesn't turn into an
hour of inference.

Configuration (all optional; comparison is skipped entirely when the incumbent
list is empty, which is the default):

``OP_TRAIN_INCUMBENT_MODELS``
    Comma-separated Triton model names to probe, in preference order. The first
    one reported ``READY`` by Triton's repository index is used. Unset means "no
    incumbent configured" -- the comparison is skipped and ``compare`` stays
    ``null``, which is what a fresh deployment with nothing to compare against
    should see.
``OP_TRITON_HTTP_URL``
    Triton HTTP endpoint. Default ``http://triton-server:8000``.
``OP_TRITON_MODEL_REPO``
    Triton model repository, if it is mounted into the trainer container. Used
    only to read the incumbent's ``labels.txt`` so per-class rows carry names
    instead of bare ids. Default ``/models``; a missing dir is not an error.
``OP_TRAIN_COMPARE_MAX_IMAGES``
    Cap on test images swept. Default 200.
``OP_TRAIN_COMPARE_INPUT_SIZE``
    Square letterbox size fed to the incumbent. Default 640.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any


log = logging.getLogger('incumbent_compare')


def _env_models() -> tuple[str, ...]:
    raw = os.environ.get('OP_TRAIN_INCUMBENT_MODELS', '')
    return tuple(tok.strip() for tok in raw.split(',') if tok.strip())


TRITON_HTTP_URL = os.environ.get('OP_TRITON_HTTP_URL', 'http://triton-server:8000')
TRITON_MODEL_REPO = Path(os.environ.get('OP_TRITON_MODEL_REPO', '/models'))
COMPARE_MAX_IMAGES = int(os.environ.get('OP_TRAIN_COMPARE_MAX_IMAGES', '200'))
COMPARE_INPUT_SIZE = int(os.environ.get('OP_TRAIN_COMPARE_INPUT_SIZE', '640'))

# Detection filtering for the lightweight evaluator.
IOU_MATCH_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.25

_LETTERBOX_FILL = 114


# ---------------------------------------------------------------------------
# Candidate-side metric extraction (Ultralytics artifacts)
# ---------------------------------------------------------------------------


def read_results_csv_last_row(results_csv: Path) -> dict[str, str] | None:
    """Return the final row of Ultralytics' ``results.csv`` keyed by header.

    ``None`` if the file is missing or unparseable. Ultralytics names the
    columns slightly differently across versions (``metrics/mAP50(B)`` vs
    ``metrics/mAP_0.5``); callers should accept both spellings.
    """
    if not results_csv.is_file():
        return None
    try:
        lines = [ln for ln in results_csv.read_text(encoding='utf-8').splitlines() if ln.strip()]
        if len(lines) < 2:
            return None
        header = [h.strip() for h in lines[0].split(',')]
        last = [v.strip() for v in lines[-1].split(',')]
    except OSError as exc:
        log.warning('eval: failed to parse results.csv: %s', exc)
        return None
    return dict(zip(header, last, strict=False))


def extract_top_level_metrics(row: dict[str, str]) -> dict[str, float]:
    """Pull mAP50 / mAP50-95 out of a ``results.csv`` row.

    Accepts either the new ``metrics/mAP50(B)`` or the legacy
    ``metrics/mAP_0.5`` spelling.
    """
    out: dict[str, float] = {}
    map50 = row.get('metrics/mAP50(B)') or row.get('metrics/mAP_0.5')
    map5095 = row.get('metrics/mAP50-95(B)') or row.get('metrics/mAP_0.5:0.95')
    if map50 is not None:
        with contextlib.suppress(ValueError):
            out['map50'] = float(map50)
    if map5095 is not None:
        with contextlib.suppress(ValueError):
            out['map50_95'] = float(map5095)
    return out


def per_class_from_val_results(val_results: Any) -> list[dict[str, Any]]:
    """Convert Ultralytics' ``DetMetrics`` object into our per-class shape.

    Ultralytics exposes per-class P/R/F1/AP50 via ``box.p``, ``box.r``,
    ``box.f1``, ``box.ap50`` indexed by ``ap_class_index``. ``names`` maps class
    id to display name; ``nt_per_class`` is the per-class support (number of
    ground-truth instances on the eval split).

    Returns ``[]`` if the validator object lacks the expected attributes (older
    Ultralytics) so callers can fall through gracefully.
    """
    out: list[dict[str, Any]] = []
    box = getattr(val_results, 'box', None)
    if box is None:
        return out
    try:
        ap_class_index = list(getattr(box, 'ap_class_index', []))
        precision = list(getattr(box, 'p', []))
        recall = list(getattr(box, 'r', []))
        f1 = list(getattr(box, 'f1', []))
        ap50 = list(getattr(box, 'ap50', []))
        names = getattr(val_results, 'names', {}) or {}
        nt_per_class = getattr(val_results, 'nt_per_class', None)
    except (AttributeError, TypeError) as exc:
        log.warning('eval: per-class extraction failed: %s', exc)
        return out

    for i, cls_id in enumerate(ap_class_index):
        try:
            cid = int(cls_id)
        except (TypeError, ValueError):
            continue
        support = 0
        if nt_per_class is not None:
            try:
                support = int(nt_per_class[cid])
            except (IndexError, TypeError, ValueError):
                support = 0
        out.append(
            {
                'class_id': cid,
                'name': str(names.get(cid, str(cid))),
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1': float(f1[i]) if i < len(f1) else 0.0,
                'ap50': float(ap50[i]) if i < len(ap50) else 0.0,
                'support': support,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Incumbent-side: Triton probe + inference + lightweight scoring
# ---------------------------------------------------------------------------


def resolve_incumbent_model_name(http_url: str = TRITON_HTTP_URL) -> str | None:
    """Return the first configured incumbent that Triton reports as READY.

    ``None`` when no incumbent is configured, Triton is unreachable, or none of
    the candidates are loaded.
    """
    candidates = _env_models()
    if not candidates:
        log.info('compare: no OP_TRAIN_INCUMBENT_MODELS configured; skipping comparison')
        return None
    try:
        import requests
    except ImportError:  # pragma: no cover - requests is in the trainer image
        return None
    try:
        resp = requests.post(f'{http_url}/v2/repository/index', timeout=5)
        resp.raise_for_status()
        loaded = {entry.get('name') for entry in resp.json() if entry.get('state') == 'READY'}
    except Exception as exc:  # any transport failure means "no incumbent"
        log.warning('compare: triton repository probe failed: %s', exc)
        return None
    for cand in candidates:
        if cand in loaded:
            return cand
    log.info('compare: no incumbent model loaded in triton (tried %s)', list(candidates))
    return None


def fetch_incumbent_class_names(
    triton_model: str,
    model_repo: Path = TRITON_MODEL_REPO,
) -> dict[int, str]:
    """Read the incumbent's ``labels.txt`` from the Triton model repository.

    Triton's HTTP API does not serve ``labels.txt``, so this only works when the
    model repo is mounted into the trainer container (``OP_TRITON_MODEL_REPO``).
    Returns ``{}`` when it isn't -- per-class rows then carry numeric ids, which
    is a cosmetic degradation, not an error.
    """
    labels_path = model_repo / triton_model / 'labels.txt'
    if not labels_path.is_file():
        return {}
    try:
        lines = labels_path.read_text(encoding='utf-8').splitlines()
    except OSError as exc:
        log.warning('compare: labels.txt read failed for %s: %s', triton_model, exc)
        return {}
    return {i: line.strip() for i, line in enumerate(lines) if line.strip()}


def list_test_images(data_yaml_path: Path) -> tuple[list[Path], list[Path]]:
    """Return ``(image_paths, label_paths)`` from the test split of a data.yaml.

    Falls back to ``val`` if ``test`` is missing -- Ultralytics does the same
    redirection internally during ``val()``.
    """
    try:
        import yaml
    except ImportError:
        return [], []
    if not data_yaml_path.is_file():
        return [], []
    try:
        with data_yaml_path.open('r', encoding='utf-8') as fh:
            doc = yaml.safe_load(fh) or {}
    except (OSError, ValueError) as exc:
        log.warning('compare: data.yaml read failed: %s', exc)
        return [], []
    images_dir_raw = doc.get('test') or doc.get('val')
    if not images_dir_raw:
        return [], []
    images_dir = Path(images_dir_raw)
    if not images_dir.is_dir():
        return [], []
    image_paths: list[Path] = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        image_paths.extend(sorted(images_dir.glob(ext)))
    # Ultralytics convention: parallel labels/{split}/<stem>.txt next to
    # images/{split}/<stem>.<ext>.
    label_paths = [
        Path(str(img).replace('/images/', '/labels/', 1)).with_suffix('.txt') for img in image_paths
    ]
    return image_paths, label_paths


Detection = tuple[float, float, float, float, float, int]


def run_incumbent_inference(
    image_paths: list[Path],
    triton_model: str,
    http_url: str = TRITON_HTTP_URL,
    max_images: int = COMPARE_MAX_IMAGES,
) -> list[list[Detection]]:
    """Run Triton inference over a sample of the test images.

    Returns per-image detections as ``[(x1, y1, x2, y2, score, class_id), ...]``
    in the *original* image's pixel space. Uses Triton's HTTP v2 ``infer``
    endpoint with a JSON envelope so the trainer image doesn't need
    ``tritonclient``. Assumes the end-to-end NMS output signature
    (``num_dets``/``det_boxes``/``det_scores``/``det_classes``) this repo's
    promote path emits.
    """
    try:
        import cv2
        import numpy as np
        import requests
    except ImportError as exc:
        log.warning('compare: missing deps for incumbent inference: %s', exc)
        return []

    out: list[list[Detection]] = []
    sample = image_paths[:max_images]
    url = f'{http_url}/v2/models/{triton_model}/infer'

    for img_path in sample:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                out.append([])
                continue
            orig_h, orig_w = img.shape[:2]
            scale = min(COMPARE_INPUT_SIZE / orig_h, COMPARE_INPUT_SIZE / orig_w, 1.0)
            new_w, new_h = round(orig_w * scale), round(orig_h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            canvas = np.full(
                (COMPARE_INPUT_SIZE, COMPARE_INPUT_SIZE, 3), _LETTERBOX_FILL, dtype=np.uint8
            )
            pad_w = (COMPARE_INPUT_SIZE - new_w) // 2
            pad_h = (COMPARE_INPUT_SIZE - new_h) // 2
            canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]

            payload = {
                'inputs': [
                    {
                        'name': 'images',
                        'shape': list(chw.shape),
                        'datatype': 'FP32',
                        'data': chw.flatten().tolist(),
                    }
                ],
                'outputs': [
                    {'name': 'num_dets'},
                    {'name': 'det_boxes'},
                    {'name': 'det_scores'},
                    {'name': 'det_classes'},
                ],
            }
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            outputs_by_name = {o['name']: o for o in resp.json().get('outputs', [])}
            num_dets = int(outputs_by_name['num_dets']['data'][0])
            boxes_flat = outputs_by_name['det_boxes']['data']
            scores = outputs_by_name['det_scores']['data']
            classes = outputs_by_name['det_classes']['data']
            dets: list[Detection] = []
            for i in range(num_dets):
                bx = boxes_flat[i * 4 : (i + 1) * 4]
                if len(bx) != 4:
                    continue
                dets.append(
                    (
                        (bx[0] - pad_w) / scale,
                        (bx[1] - pad_h) / scale,
                        (bx[2] - pad_w) / scale,
                        (bx[3] - pad_h) / scale,
                        float(scores[i]),
                        int(classes[i]),
                    )
                )
            out.append(dets)
        except Exception as exc:  # one bad image must not abort the sweep
            log.warning('compare: incumbent infer failed for %s: %s', img_path, exc)
            out.append([])
    return out


def read_yolo_labels(
    label_path: Path, img_w: int, img_h: int
) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into ``[(class_id, x1, y1, x2, y2), ...]``.

    Coordinates are pixel-space. Returns ``[]`` if the file is missing or empty
    (a background image with no boxes).
    """
    if not label_path.is_file():
        return []
    out: list[tuple[int, float, float, float, float]] = []
    try:
        lines = label_path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except (ValueError, IndexError):
            continue
        out.append(
            (
                cls,
                (cx - w / 2) * img_w,
                (cy - h / 2) * img_h,
                (cx + w / 2) * img_w,
                (cy + h / 2) * img_h,
            )
        )
    return out


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Standard IoU between two ``(x1, y1, x2, y2)`` boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def per_class_from_predictions(
    image_paths: list[Path],
    label_paths: list[Path],
    predictions: list[list[Detection]],
    class_names: dict[int, str],
    iou_thresh: float = IOU_MATCH_THRESHOLD,
    score_thresh: float = SCORE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Compute per-class P/R/F1/support from raw predictions + ground truth.

    This is a lightweight evaluator -- it does NOT match Ultralytics' COCO-AP
    implementation byte-for-byte (no 101-point interpolation; thresholded P/R
    instead of curve area). It is good enough to surface a regression flag in
    the side-by-side panel. ``ap50`` is reported as the F1 stand-in so the
    delta table stays computable against the candidate's real AP50 column.
    """
    try:
        import cv2
    except ImportError:
        return []
    per_class: dict[int, dict[str, int]] = {}

    for img_path, lbl_path, dets in zip(image_paths, label_paths, predictions, strict=False):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = read_yolo_labels(lbl_path, w, h)

        gt_by_class: dict[int, list[tuple[float, float, float, float]]] = {}
        for cls, x1, y1, x2, y2 in gts:
            gt_by_class.setdefault(cls, []).append((x1, y1, x2, y2))
            per_class.setdefault(cls, {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0})
            per_class[cls]['support'] += 1

        pred_by_class: dict[int, list[tuple[float, tuple[float, float, float, float]]]] = {}
        for x1, y1, x2, y2, score, cls in dets:
            if score < score_thresh:
                continue
            pred_by_class.setdefault(cls, []).append((score, (x1, y1, x2, y2)))
            per_class.setdefault(cls, {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0})

        # Per-image, per-class greedy matching (mirrors Ultralytics' val).
        for cls in set(gt_by_class) | set(pred_by_class):
            gts_c = gt_by_class.get(cls, [])
            preds_c = sorted(pred_by_class.get(cls, []), key=lambda t: -t[0])
            matched_gt: set[int] = set()
            for _, pbox in preds_c:
                best_iou = 0.0
                best_gt = -1
                for gi, gbox in enumerate(gts_c):
                    if gi in matched_gt:
                        continue
                    iou = box_iou(pbox, gbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gi
                if best_iou >= iou_thresh and best_gt >= 0:
                    matched_gt.add(best_gt)
                    per_class[cls]['tp'] += 1
                else:
                    per_class[cls]['fp'] += 1
            per_class[cls]['fn'] += len(gts_c) - len(matched_gt)

    out: list[dict[str, Any]] = []
    for cls, counts in per_class.items():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out.append(
            {
                'class_id': cls,
                'name': class_names.get(cls, str(cls)),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                # No AP50 is available from threshold counts; F1 stands in so
                # the deltas remain computable.
                'ap50': float(f1),
                'support': int(counts['support']),
            }
        )
    return sorted(out, key=lambda r: r['class_id'])


def per_class_delta(
    candidate: list[dict[str, Any]],
    incumbent: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute ``candidate - incumbent`` deltas keyed by class name.

    Classes present on only one side report a ``status`` marker instead of a
    delta (``1.0`` = new in candidate, ``-1.0`` = dropped) so the frontend can
    surface them as new / removed capabilities.
    """
    by_name_inc = {row['name']: row for row in incumbent}
    by_name_cand = {row['name']: row for row in candidate}
    deltas: dict[str, dict[str, float]] = {}
    metric_keys = ('precision', 'recall', 'f1', 'ap50')
    for name in set(by_name_cand) | set(by_name_inc):
        c = by_name_cand.get(name)
        i = by_name_inc.get(name)
        if c is None:
            deltas[name] = {'status': -1.0}  # removed
            continue
        if i is None:
            deltas[name] = {'status': 1.0}  # new in candidate
            continue
        deltas[name] = {k: float(c.get(k, 0.0)) - float(i.get(k, 0.0)) for k in metric_keys}
    return deltas


def build_compare_block(
    *,
    save_dir: Path,
    candidate_per_class: list[dict[str, Any]],
    candidate_summary: dict[str, float],
    data_yaml_path: Path,
) -> dict[str, Any] | None:
    """Evaluate the incumbent on the candidate's test split; return the block.

    Returns ``None`` (leaving ``status.compare`` null) when no incumbent is
    configured/reachable or the test split is empty. Mirrors the result onto
    ``save_dir/compare.json`` for offline post-mortems.
    """
    incumbent_name = resolve_incumbent_model_name()
    if not incumbent_name:
        return None
    image_paths, label_paths = list_test_images(data_yaml_path)
    if not image_paths:
        log.warning('compare: empty test set; leaving compare=None')
        return None

    log.info(
        'compare: evaluating incumbent=%s on %d test image(s)', incumbent_name, len(image_paths)
    )
    incumbent_names = fetch_incumbent_class_names(incumbent_name)
    predictions = run_incumbent_inference(image_paths=image_paths, triton_model=incumbent_name)
    incumbent_per_class = per_class_from_predictions(
        image_paths=image_paths,
        label_paths=label_paths,
        predictions=predictions,
        class_names=incumbent_names,
    )

    def _avg(key: str) -> float:
        if not incumbent_per_class:
            return 0.0
        return sum(r[key] for r in incumbent_per_class) / len(incumbent_per_class)

    inc_avg_ap50 = _avg('ap50')
    inc_summary = {
        'precision': _avg('precision'),
        'recall': _avg('recall'),
        'f1': _avg('f1'),
        'ap50': inc_avg_ap50,
    }
    delta_per_metric: dict[str, float] = {}
    if candidate_summary.get('map50') is not None:
        delta_per_metric['map50'] = float(candidate_summary['map50']) - inc_avg_ap50

    compare_block: dict[str, Any] = {
        'incumbent_name': incumbent_name,
        'test_images_evaluated': len(image_paths),
        'incumbent': {'summary': inc_summary, 'per_class': incumbent_per_class},
        'candidate': {
            'summary': {
                'map50': candidate_summary.get('map50'),
                'map50_95': candidate_summary.get('map50_95'),
            },
            'per_class': candidate_per_class,
        },
        'delta_per_metric': delta_per_metric,
        'delta_per_class': per_class_delta(candidate_per_class, incumbent_per_class),
    }
    try:
        (save_dir / 'compare.json').write_text(
            json.dumps(compare_block, indent=2, default=str), encoding='utf-8'
        )
    except OSError as exc:
        log.warning('compare: failed to write compare.json: %s', exc)
    return compare_block
