"""Probe model loading and class-posterior extraction.

Builds a predict function for a probe checkpoint that returns a real
per-class posterior (top-1 class, confidence, entropy, margin) instead of
post-NMS detections. See :mod:`src.services.curation.probe_predictions` for
why the posterior matters and how the probe job uses it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.logging import get_logger


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger(__name__)


# =============================================================================
# Raw (pre-NMS) prediction plumbing
# =============================================================================


class _RawPreds:
    """Wraps one image's raw pre-NMS prediction tensor.

    ``BasePredictor.stream_inference``/``write_results`` do bookkeeping
    attribute assignments on each result object post-postprocess (e.g.
    ``speed``, and ultralytics 8.4.x's ``save_dir``) -- deliberately NOT
    ``__slots__``-restricted: a fixed attribute tuple is exactly what
    broke against a newer ultralytics release setting attributes this
    module never anticipated (``'_RawPreds' object has no attribute
    'save_dir' and no __dict__ for setting new attributes``).
    """

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor
        self.speed: dict[str, float] | None = None


def _use_class_score_head(model: Any) -> None:
    """Switch an end-to-end (NMS-free) model to its one-to-many head.

    An end-to-end head (e.g. YOLO26) returns post-selection rows
    ``(max_det, 6)`` = box, confidence, class id, with no per-class scores.
    Its one-to-many branch returns the ``(4 + nc, anchors)`` class-score
    tensor the posterior is built from. Models without the toggle are left
    alone.
    """
    inner = getattr(model, 'model', None)
    if inner is not None and getattr(inner, 'end2end', False):
        inner.end2end = False


def _build_raw_predictor(model: Any) -> Any:
    """Construct a ``DetectionPredictor`` bound to ``model`` whose
    ``postprocess`` is patched to skip NMS and return the raw pre-NMS
    per-anchor tensor per image (see module docstring).

    Mirrors ``ultralytics.engine.model.Model.predict``'s own predictor
    construction so preprocessing (letterbox/normalize) and the
    ``AutoBackend`` wrapping stay identical to the high-level API — only
    the postprocess step differs.
    """
    import types

    from ultralytics.models.yolo.detect import DetectionPredictor

    args = {
        **model.overrides,
        'conf': 1e-6,  # irrelevant post-fix (no NMS filtering runs), kept low defensively
        'batch': 1,
        'mode': 'predict',
        'rect': True,
        'verbose': False,
        # Explicit, not left to ultralytics' own default (``save`` was
        # True by default in 8.4.161): any of these routes through
        # ``BasePredictor.write_results``, which calls ``result.verbose()``
        # and sets ``result.save_dir`` unconditionally -- neither of which
        # ``_RawPreds`` needs, and this pass never wants disk/window output.
        'save': False,
        'save_txt': False,
        'save_crop': False,
        'show': False,
    }
    _use_class_score_head(model)
    predictor = DetectionPredictor(overrides=args, _callbacks=model.callbacks)
    predictor.setup_model(model=model.model, verbose=False)

    def _raw_postprocess(
        _self: Any, preds: Any, _img: Any, _orig_imgs: Any, **_kwargs: Any
    ) -> list[_RawPreds]:
        batch = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [_RawPreds(batch[i]) for i in range(batch.shape[0])]

    predictor.postprocess = types.MethodType(_raw_postprocess, predictor)  # type: ignore[method-assign]
    return predictor


def _summarize_prediction_raw(
    raw_results: list[_RawPreds], model: Any
) -> tuple[str | None, float, float, float]:
    """Pull top-1 class, confidence, entropy, and margin from a raw
    (pre-NMS) prediction tensor.

    Returns ``(class_name, confidence, entropy, margin)``. Empty/degenerate
    predictions return ``(None, 0.0, 0.0, 0.0)``.
    """
    import torch

    if not raw_results:
        return None, 0.0, 0.0, 0.0
    tensor = raw_results[0].tensor
    if tensor is None or tensor.ndim != 2 or tensor.shape[1] == 0 or tensor.shape[0] <= 4:
        return None, 0.0, 0.0, 0.0
    names = getattr(model, 'names', {})
    if names and tensor.shape[0] != 4 + len(names):
        # Not a (4 + nc, anchors) class-score tensor, e.g. end-to-end
        # post-selection rows; reading it as one would invent a posterior.
        logger.warning(
            'probe_raw_tensor_shape_mismatch',
            shape=tuple(tensor.shape),
            expected_rows=4 + len(names),
        )
        return None, 0.0, 0.0, 0.0

    cls_scores = tensor[4:, :]  # (nc, num_anchors) — per-class sigmoid scores
    if cls_scores.shape[0] == 0:
        return None, 0.0, 0.0, 0.0

    # Same box-selection criterion as ultralytics' own NMS best-class-only
    # path (`conf, j = cls.max(1, keepdim=True)`): per-anchor top-1 class
    # confidence, then the single highest-confidence anchor overall (NMS
    # would keep exactly this box as the top detection in its cluster; we
    # skip the cluster-dedup step since we only want one box per crop).
    top1_per_anchor, _ = cls_scores.max(dim=0)
    best_anchor = int(torch.argmax(top1_per_anchor).item())
    raw_row = cls_scores[:, best_anchor]  # (nc,)

    probs = torch.softmax(raw_row, dim=0)
    top1_prob, top1_idx = torch.max(probs, dim=0)
    sorted_probs, _ = torch.sort(probs, descending=True)
    top2_prob = float(sorted_probs[1].item()) if probs.shape[0] > 1 else 0.0
    margin = float(top1_prob.item()) - top2_prob
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())

    cls_id = int(top1_idx.item())
    cls_name = (
        names.get(cls_id)
        if isinstance(names, dict)
        else (names[cls_id] if cls_id < len(names) else None)
    )
    return cls_name, float(top1_prob.item()), entropy, margin


# =============================================================================
# Second-architecture-family raw prediction plumbing
# =============================================================================


YOLOV5_OBJ_INPUT_SIZE = 1280
YOLOV5_OBJ_LETTERBOX_PAD_VALUE = 114


def _letterbox_yolov5_objectness(img: Any, target: int = YOLOV5_OBJ_INPUT_SIZE) -> Any:
    """Letterbox (gray-pad) a PIL image to ``target x target`` -> NCHW
    float32 ``[0, 1]``.

    Mirrors the ingest pipeline's own letterbox convention (bilinear
    resize, centered pad, value 114) so this probe path sees the same
    preprocessing production inference already applies for this
    checkpoint family — not a second, potentially-drifted implementation.
    """
    from PIL import Image

    orig_w, orig_h = img.size
    scale = min(target / orig_h, target / orig_w)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new(
        'RGB',
        (target, target),
        (
            YOLOV5_OBJ_LETTERBOX_PAD_VALUE,
            YOLOV5_OBJ_LETTERBOX_PAD_VALUE,
            YOLOV5_OBJ_LETTERBOX_PAD_VALUE,
        ),
    )
    pad_w = (target - new_w) / 2.0
    pad_h = (target - new_h) / 2.0
    canvas.paste(resized, (int(pad_w), int(pad_h)))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW
    return chw.astype(np.float32, copy=False)


def _summarize_prediction_yolov5_objectness_raw(
    raw_output: Any, class_names: dict[int, str]
) -> tuple[str | None, float, float, float]:
    """Pull top-1 class, confidence, entropy, and margin from the
    second-architecture-family's raw (pre-NMS) output: ``(num_anchors, 5 +
    nc)`` = cx,cy,w,h,obj_conf, +nc per-class sigmoid scores (already
    activated in-graph).

    Box-selection uses ``obj_conf * max(cls_conf)`` per anchor — the same
    criterion that family's own NMS applies (``x[:, 5:] *= x[:, 4:5]`` then
    best-class-only ``.max(1)``) — but the posterior itself is softmax over
    the anchor's *raw* (not objectness-scaled) per-class row, matching this
    module's class-posterior convention (entropy/margin need a genuine
    distribution, not a detection score).

    Returns ``(class_name, confidence, entropy, margin)``. Empty/degenerate
    predictions return ``(None, 0.0, 0.0, 0.0)``.
    """
    if raw_output.ndim != 2 or raw_output.shape[0] == 0 or raw_output.shape[1] <= 5:
        return None, 0.0, 0.0, 0.0

    obj_conf = raw_output[:, 4]
    cls_conf = raw_output[:, 5:]
    if cls_conf.shape[1] == 0:
        return None, 0.0, 0.0, 0.0

    combined_top = obj_conf * cls_conf.max(axis=1)
    best_anchor = int(np.argmax(combined_top))
    raw_row = cls_conf[best_anchor].astype(np.float64)

    shifted = raw_row - raw_row.max()
    exp = np.exp(shifted)
    total = exp.sum()
    if total <= 0:
        return None, 0.0, 0.0, 0.0
    probs = exp / total

    top1_idx = int(np.argmax(probs))
    top1_prob = float(probs[top1_idx])
    sorted_probs = np.sort(probs)[::-1]
    top2_prob = float(sorted_probs[1]) if probs.shape[0] > 1 else 0.0
    margin = top1_prob - top2_prob
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())

    cls_name = class_names.get(top1_idx)
    return cls_name, top1_prob, entropy, margin


_PredictFn = Any  # Callable[[Any], tuple[str | None, float, float, float]]


def _build_yolo11_predictor(model_path: Path) -> tuple[_PredictFn, str]:
    """``(predict_fn, default_version_tag)`` for an ultralytics-family probe
    checkpoint (the primary, validated path)."""
    from ultralytics import YOLO

    model = YOLO(str(model_path), task='detect')
    raw_predictor = _build_raw_predictor(model)

    def predict(crop: Any) -> tuple[str | None, float, float, float]:
        raw_results = raw_predictor(source=crop)
        return _summarize_prediction_raw(raw_results, model)

    return predict, model_path.name


def _build_yolov5_objectness_predictor(model_path: Path) -> tuple[_PredictFn, str]:
    """``(predict_fn, default_version_tag)`` for the second (non-ultralytics)
    architecture family."""
    import onnxruntime as ort

    from src.clients.curation_opensearch import get_class_registry

    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    registry = get_class_registry()
    class_names = {c.class_id: c.class_name for c in registry.load().classes}

    def predict(crop: Any) -> tuple[str | None, float, float, float]:
        chw = _letterbox_yolov5_objectness(crop, YOLOV5_OBJ_INPUT_SIZE)
        raw = session.run([output_name], {input_name: chw})[0]
        raw = raw[0] if raw.ndim == 3 else raw  # strip batch dim -> (num_anchors, 85)
        return _summarize_prediction_yolov5_objectness_raw(raw, class_names)

    return predict, model_path.name


_PROBE_ARCHITECTURES = ('yolo11', 'yolov5_objectness')


def _build_predictor(model_path: Path, architecture: str) -> tuple[_PredictFn, str]:
    """Dispatch to the architecture-specific predictor builder.

    A dedicated (non-underscore-internal) seam so tests can monkeypatch the
    whole thing and avoid loading a real model file.
    """
    if architecture == 'yolo11':
        return _build_yolo11_predictor(model_path)
    if architecture == 'yolov5_objectness':
        return _build_yolov5_objectness_predictor(model_path)
    raise ValueError(
        f'unknown probe architecture {architecture!r} (expected one of {_PROBE_ARCHITECTURES})'
    )
