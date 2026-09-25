"""Probe inference + uncertainty queue (active-learning wave).

The "probe" is a small/fast model (typically a `profile='probe'`
quick-smoke training run — see :mod:`src.services.training.profiles`)
exported ONNX-without-NMS-baked-in. We run it across every non-holdout
item document and record:

- ``probe_pred_class`` — top-1 class name predicted on the crop.
- ``probe_pred_class_id`` — that name's class-registry id (``None`` when the
  registry has no active class of that name).
- ``probe_pred_confidence`` — top-1 softmax score, ``p(ŷ)``.
- ``probe_pred_entropy`` — real Shannon entropy of the class posterior.
- ``probe_pred_margin`` — ``p(top1) - p(top2)`` (feeds the
  ``item_scores.mistakenness`` overlay).
- ``probe_disagreement`` — ``true`` iff the probe's top-1 differs from the
  current ``class_name`` in OpenSearch.

**Class-posterior fix:** naively calling a high-level ``model.predict()``
runs NMS (``ultralytics.utils.nms.non_max_suppression``) before this module
ever sees a result — NMS collapses the raw ``(4+nc, num_anchors)``
per-anchor tensor to top-1 via ``conf, j = cls.max(1, keepdim=True)`` BEFORE
returning. Entropy computed downstream of that call is therefore entropy
over per-*detection* box confidences (already-collapsed top-1 scores across
different boxes), not a real class posterior ``p(y|x)``.

The fix (:func:`_build_raw_predictor` / :func:`_summarize_prediction_raw`)
builds a ``DetectionPredictor`` the same way ``Model.predict()`` does
internally (same preprocessing, same ``AutoBackend``) but replaces its
``postprocess`` with one that returns the RAW pre-NMS per-anchor tensor
instead of running NMS. We then apply the exact same box-selection
criterion ultralytics' own NMS uses (``cls.max(1)`` — best class per
anchor, then the globally highest-confidence anchor) to pick one box per
crop, and softmax-normalize its full ``nc``-length class-score row into a
genuine posterior. Note: the YOLO detection head already applies a
per-class **sigmoid** in its forward pass (each class is an independent
Bernoulli, not mutually exclusive logits) — softmax-normalizing those
sigmoid outputs (rather than raw pre-activation logits) is what this
module means by "class posterior"; it yields a valid distribution that
sums to 1 and preserves the top-1 class (softmax is monotonic), which is
what entropy/margin need. This is a documented approximation, not a
hidden one.

This module is intentionally importable in tests **without** ultralytics or
onnxruntime installed — those heavy imports happen inside
:py:func:`_build_yolo11_predictor` / :py:func:`_build_yolov5_objectness_predictor`, invoked
lazily from :py:func:`run_probe_inference` only when the function is
actually called. Tests that just want to exercise
:py:func:`build_uncertainty_queue` do not need either dependency.

**Second architecture family (``architecture='yolov5_objectness'``):** some deployments
already have an alternate (non-ultralytics) detector checkpoint deployed
in production (e.g. a YOLOv5-fork export) and would rather reuse it as the
probe than train a fresh ultralytics checkpoint. That family's raw export
shape is ``(num_anchors, 4 + 1 + nc)`` — cx,cy,w,h, a separate
**objectness** channel, then nc already-sigmoid'ed per-class scores —
versus YOLO11's ``(4 + nc, num_anchors)`` with no objectness channel. This
module therefore loads that family directly via onnxruntime (CPU only) and
applies the fork's own box-selection criterion (``x[:, 5:] *= x[:, 4:5]``
then best-class-only ``.max(1)``) to pick one anchor per crop, then
softmax-normalizes that anchor's raw per-class row into a genuine
posterior — the same "class posterior" convention
``_summarize_prediction_raw`` uses above.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config import CurationConfig, get_curation_config
from src.core.logging import get_logger


if TYPE_CHECKING:
    from pathlib import Path

    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


# =============================================================================
# Raw (pre-NMS) prediction plumbing
# =============================================================================


class _RawPreds:
    """Wraps one image's raw pre-NMS prediction tensor.

    ``BasePredictor.stream_inference`` unconditionally does
    ``self.results[i].speed = {...}`` bookkeeping after postprocess — this
    thin wrapper just needs to accept that attribute assignment so we can
    skip constructing real ``Results`` objects (which would require running
    NMS first).
    """

    __slots__ = ('speed', 'tensor')

    def __init__(self, tensor: Any) -> None:
        self.tensor = tensor
        self.speed: dict[str, float] | None = None


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
    }
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

    names = getattr(model, 'names', {})
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


# =============================================================================
# Probe inference
# =============================================================================


def _registry_class_ids(cfg: CurationConfig) -> dict[str, int]:
    """Active class name -> id from the configured registry ({} when absent)."""
    from src.clients.curation_opensearch import ClassRegistry

    try:
        reg = ClassRegistry(path=cfg.class_registry_path).load()
    except Exception as exc:
        logger.warning('probe_registry_unavailable', error=str(exc))
        return {}
    return {c.class_name: c.class_id for c in reg.classes if not c.deprecated}


async def run_probe_inference(
    model_path: Path,
    opensearch: AsyncOpenSearch,
    *,
    config: CurationConfig | None = None,
    batch_size: int = 32,  # noqa: ARG001 - kept for call-site compat; raw path is one-crop-at-a-time
    max_crops: int | None = None,
    model_version: str | None = None,
    architecture: str = 'yolo11',
    page_size: int = 1000,
    resume: bool = False,
    class_ids: dict[str, int] | None = None,
) -> int:
    """Run the probe checkpoint over every non-holdout item and record
    uncertainty.

    Args:
        model_path: Path to the probe checkpoint. For ``architecture='yolo11'``
            an ultralytics-loadable ONNX. For ``architecture='yolov5_objectness'`` an
            already-deployed second-family ONNX (reused rather than
            training a fresh probe).
        opensearch: AsyncOpenSearch client.
        config: :class:`CurationConfig` supplying ``items_index`` and the
            filesystem roots used to resolve stored ``image_path`` values
            (defaults to :func:`get_curation_config`).
        batch_size: Unused since the class-posterior fix (raw-tensor path
            processes one crop per forward call); kept for call-site
            compatibility.
        max_crops: Optional cap (useful in tests + smoke runs).
        model_version: Provenance tag stamped onto ``probe_model_version``.
            Defaults to ``model_path.name``.
        architecture: ``'yolo11'`` (default) or ``'yolov5_objectness'``.
        page_size: Items per scroll page. Every page is fully inferred
            before the next scroll call, so a slow (CPU) probe needs a
            page small enough to finish inside the scroll keep-alive.
        resume: Skip items whose ``probe_model_version`` already equals
            this run's version tag, so an interrupted pass picks up
            where it stopped instead of re-scoring everything.
        class_ids: class name -> registry id, for ``probe_pred_class_id``.
            Defaults to the active classes of the configured registry.

    Returns:
        Number of item docs updated.
    """
    # Lazy-import heavy deps so test environments without ultralytics/onnxruntime
    # can still import this module.
    from datetime import UTC, datetime

    from PIL import Image

    cfg = config or get_curation_config()
    if class_ids is None:
        class_ids = _registry_class_ids(cfg)

    logger.info('probe_init', model=str(model_path), architecture=architecture)
    predict_fn, default_version = _build_predictor(model_path, architecture)
    version_tag = model_version or default_version

    body: dict[str, Any] = {
        'size': page_size,
        'query': probe_candidate_query(skip_version=version_tag if resume else None),
        '_source': [
            'crop_id',
            'image_path',
            'bbox_norm',
            'class_name',
        ],
        # F-26: scroll hygiene, no relevance scoring needed here.
        'sort': ['_doc'],
    }
    resp = await opensearch.search(
        index=cfg.items_index,
        body=body,
        scroll='5m',
    )
    scroll_id = resp.get('_scroll_id')
    hits = resp.get('hits', {}).get('hits', [])
    processed = 0
    try:
        while hits:
            # F-26: one bulk() per scroll page instead of one update() per
            # item — page_size items become 1 round-trip instead of N.
            bulk_body: list[dict[str, Any]] = []
            for hit in hits:
                if max_crops is not None and processed >= max_crops:
                    if bulk_body:
                        await opensearch.bulk(body=bulk_body, refresh=False)
                    return processed
                src = hit.get('_source') or {}
                crop_id = src.get('crop_id') or hit.get('_id')
                image_path = src.get('image_path')
                bbox = src.get('bbox_norm')
                if not image_path or bbox is None or len(bbox) != 4:
                    continue
                resolved = _resolve_image(image_path, config=cfg)
                if resolved is None:
                    continue
                try:
                    with Image.open(resolved) as raw:
                        rgb = raw.convert('RGB')
                        w, h = rgb.size
                        x1, y1, x2, y2 = bbox
                        crop = rgb.crop(
                            (
                                round(x1 * w),
                                round(y1 * h),
                                round(x2 * w),
                                round(y2 * h),
                            )
                        )
                except Exception as exc:
                    logger.warning('probe_crop_failed', crop_id=crop_id, err=repr(exc))
                    continue
                # Run the probe, bypassing NMS entirely so we get a real
                # per-class posterior (see module docstring / architecture
                # dispatch in _build_predictor above).
                pred_cls, pred_conf, entropy, margin = predict_fn(crop)
                if pred_cls is None:
                    continue
                disagreement = bool(
                    src.get('class_name') and pred_cls and pred_cls != src['class_name']
                )
                bulk_body.append({'update': {'_index': cfg.items_index, '_id': hit['_id']}})
                bulk_body.append(
                    {
                        'doc': {
                            'probe_pred_class': pred_cls,
                            'probe_pred_class_id': class_ids.get(pred_cls),
                            'probe_pred_confidence': pred_conf,
                            'probe_pred_entropy': entropy,
                            'probe_pred_margin': margin,
                            'probe_disagreement': disagreement,
                            'probe_model_version': version_tag,
                            'probe_scored_at': datetime.now(UTC).isoformat(),
                        }
                    }
                )
                processed += 1
            if bulk_body:
                bulk_resp = await opensearch.bulk(body=bulk_body, refresh=False)
                if isinstance(bulk_resp, dict) and bulk_resp.get('errors'):
                    logger.warning(
                        'probe_bulk_partial_errors', sample=bulk_resp.get('items', [])[:3]
                    )
            resp = await opensearch.scroll(scroll_id=scroll_id, scroll='5m')
            scroll_id = resp.get('_scroll_id')
            hits = resp.get('hits', {}).get('hits', [])
    finally:
        if scroll_id:
            try:
                await opensearch.clear_scroll(scroll_id=scroll_id)
            except Exception as exc:
                logger.warning('probe_clear_scroll_failed', err=str(exc))
    logger.info('probe_done', processed=processed)
    return processed


def probe_candidate_query(*, skip_version: str | None = None) -> dict[str, Any]:
    """Items the probe scores: every non-holdout item, optionally minus those
    already stamped with ``skip_version`` (resume)."""
    must_not: list[dict[str, Any]] = [{'term': {'test_holdout': True}}]
    if skip_version:
        must_not.append({'term': {'probe_model_version': skip_version}})
    return {'bool': {'must_not': must_not}}


async def count_probe_candidates(
    opensearch: AsyncOpenSearch,
    *,
    config: CurationConfig | None = None,
    skip_version: str | None = None,
) -> int:
    """How many items :func:`run_probe_inference` would visit (no model load)."""
    cfg = config or get_curation_config()
    resp = await opensearch.count(
        index=cfg.items_index,
        body={'query': probe_candidate_query(skip_version=skip_version)},
    )
    return int(resp.get('count', 0))


def _entropy(probs: list[float]) -> float:
    """Shannon entropy of a vector (treated as probabilities, normalized).

    Kept for callers that only have a list of scores (not a torch tensor)
    to normalize — e.g. ad-hoc scripts/tests. Production
    :func:`_summarize_prediction_raw` computes entropy directly on the
    softmax'd torch tensor instead.
    """
    s = sum(p for p in probs if p > 0)
    if s <= 0:
        return 0.0
    e = 0.0
    for p in probs:
        if p <= 0:
            continue
        q = p / s
        e -= q * math.log(q + 1e-12)
    return e


def _resolve_image(image_path: str, *, config: CurationConfig) -> Path | None:
    from pathlib import Path as _Path

    from src.services.curation.image_serving import resolve_crop_root

    p = _Path(image_path)
    if p.is_absolute() and p.is_file():
        return p
    root = resolve_crop_root(image_path, config)
    candidate = root / image_path
    if candidate.is_file():
        return candidate
    return None


# =============================================================================
# Uncertainty queue
# =============================================================================


async def build_uncertainty_queue(
    opensearch: AsyncOpenSearch,
    percent: float = 5.0,
    *,
    min_size: int = 50,
    max_size: int = 5000,
    config: CurationConfig | None = None,
) -> list[str]:
    """Return the most-uncertain ``percent``% of items as ``crop_id`` list.

    "Uncertain" = highest entropy OR ``probe_disagreement=true``. We rank
    by ``probe_pred_entropy`` desc with disagreements pushed to the top
    via a ``constant_score`` boost.

    Args:
        opensearch: AsyncOpenSearch client.
        percent: 0-100; the slice to surface.
        min_size: Lower bound on the queue size (avoids degenerate empty
            queues during early labeling).
        max_size: Upper bound to keep the labeler UI responsive.
        config: :class:`CurationConfig` supplying ``items_index``.
    """
    if percent <= 0:
        return []

    cfg = config or get_curation_config()

    # Total non-holdout items with at least a probe entropy score.
    count_body: dict[str, Any] = {
        'query': {
            'bool': {
                'filter': [{'exists': {'field': 'probe_pred_entropy'}}],
                'must_not': [{'term': {'test_holdout': True}}],
            }
        }
    }
    count_resp = await opensearch.count(index=cfg.items_index, body=count_body)
    total = int(count_resp.get('count', 0))
    if total == 0:
        return []

    target = max(min_size, min(max_size, math.ceil(total * percent / 100.0)))

    body: dict[str, Any] = {
        'size': target,
        '_source': ['crop_id'],
        'query': {
            'function_score': {
                'query': count_body['query'],
                'functions': [
                    {
                        'filter': {'term': {'probe_disagreement': True}},
                        'weight': 2.0,
                    }
                ],
                'score_mode': 'sum',
                'boost_mode': 'sum',
            }
        },
        'sort': [
            '_score',
            {'probe_pred_entropy': 'desc'},
        ],
    }
    resp = await opensearch.search(index=cfg.items_index, body=body)
    hits = resp.get('hits', {}).get('hits', [])
    out: list[str] = []
    for hit in hits:
        crop_id = (hit.get('_source') or {}).get('crop_id') or hit.get('_id')
        if crop_id:
            out.append(str(crop_id))
    return out


__all__ = [
    'build_uncertainty_queue',
    'count_probe_candidates',
    'probe_candidate_query',
    'run_probe_inference',
]
