"""Probe inference + uncertainty queue (active-learning wave).

The "probe" is a small/fast model (typically a `profile='probe'`
quick-smoke training run — see :mod:`src.services.training.profiles`)
exported ONNX-without-NMS-baked-in. We run it across every non-holdout
item document and record:

- ``probe_pred_class`` — top-1 class name predicted on the crop.
- ``probe_pred_class_id`` — that name's class-registry id (``None`` when the
  registry has no active class of that name).
- ``probe_pred_confidence`` — top-1 posterior probability, ``p(ŷ)``.
- ``probe_pred_entropy`` — real Shannon entropy of the class posterior.
- ``probe_pred_margin`` — ``p(top1) - p(top2)`` (feeds the
  ``item_scores.mistakenness`` overlay).
- ``probe_disagreement`` — ``true`` iff the probe's top-1 differs from the
  current ``class_name``; ``null`` when that class is not one the probe
  can predict (a probe trained on a class subset has no opinion on it).

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
crop, and normalize its full ``nc``-length class-score row into a
posterior. The YOLO head already applies a per-class **sigmoid** (each
class an independent Bernoulli), so the row is divided by its sum; that
keeps the top-1 class and the scores' relative strength. A softmax over
values already in [0, 1] would flatten every row toward uniform (a 0.9 vs
0.05 row would read as ~0.37 confidence). This is a documented
approximation, not a hidden one.

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
sum-normalizes that anchor's per-class row into a posterior — the same
convention ``_summarize_prediction_raw`` uses above.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any

from src.config import CurationConfig, get_curation_config
from src.core.logging import get_logger
from src.services.curation.probe_models import (  # noqa: F401 - re-exported for callers and tests
    _PROBE_ARCHITECTURES,
    YOLOV5_OBJ_INPUT_SIZE,
    _build_predictor,
    _build_raw_predictor,
    _PredictFn,
    _RawPreds,
    _summarize_prediction_raw,
    _summarize_prediction_yolov5_objectness_raw,
    _use_class_score_head,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


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


def _crop_and_predict(
    image_path: str,
    bbox: list[float],
    predict_fn: _PredictFn,
    cfg: CurationConfig,
    crop_id: str,
) -> tuple[str, float, float, float] | None:
    """Crop one item from its source image and run the probe on it.

    ``None`` when the image is missing, unreadable, or the probe makes no
    prediction. The probe bypasses NMS so it sees a real per-class
    posterior (see the module docstring and ``_build_predictor``).
    """
    from PIL import Image

    resolved = _resolve_image(image_path, config=cfg)
    if resolved is None:
        return None
    try:
        with Image.open(resolved) as raw:
            rgb = raw.convert('RGB')
            w, h = rgb.size
            x1, y1, x2, y2 = bbox
            crop = rgb.crop((round(x1 * w), round(y1 * h), round(x2 * w), round(y2 * h)))
    except Exception as exc:
        logger.warning('probe_crop_failed', crop_id=crop_id, err=repr(exc))
        return None
    pred_cls, pred_conf, entropy, margin = predict_fn(crop)
    if pred_cls is None:
        return None
    return pred_cls, pred_conf, entropy, margin


async def run_probe_inference(
    model_path: Path,
    opensearch: AsyncOpenSearch,
    *,
    config: CurationConfig | None = None,
    batch_size: int = 32,  # noqa: ARG001 - kept for call-site compat; raw path is one-crop-at-a-time
    max_crops: int | None = None,
    model_version: str | None = None,
    architecture: str = 'yolo26',
    page_size: int = 1000,
    resume: bool = False,
    class_ids: dict[str, int] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> int:
    """Run the probe checkpoint over every non-holdout item and record
    uncertainty.

    Args:
        model_path: Path to the probe checkpoint. For ``architecture='yolo26'``
            (default; the only trained family, see G-22) or the older
            ``'yolo11'``, an ultralytics-loadable ONNX. For
            ``architecture='yolov5_objectness'`` an already-deployed
            second-family ONNX (reused rather than training a fresh probe).
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
        architecture: ``'yolo26'`` (default), ``'yolo11'``, or
            ``'yolov5_objectness'``.
        page_size: Items per scroll page. Every page is fully inferred
            before the next scroll call, so a slow (CPU) probe needs a
            page small enough to finish inside the scroll keep-alive.
        resume: Skip items whose ``probe_model_version`` already equals
            this run's version tag, so an interrupted pass picks up
            where it stopped instead of re-scoring everything.
        class_ids: class name -> registry id, for ``probe_pred_class_id``.
            Defaults to the active classes of the configured registry.
        should_cancel: Optional cheap callable checked once per scroll page
            (after that page's bulk write). Defaults to a no-op;
            :mod:`src.services.curation.probe_job` passes its file-backed
            ``is_cancelled`` here.

    Returns:
        Number of item docs updated.
    """
    # Lazy-import heavy deps so test environments without ultralytics/onnxruntime
    # can still import this module.
    from datetime import UTC, datetime

    cfg = config or get_curation_config()
    if class_ids is None:
        class_ids = _registry_class_ids(cfg)

    logger.info('probe_init', model=str(model_path), architecture=architecture)
    predict_fn, default_version = _build_predictor(model_path, architecture)
    version_tag = model_version or default_version
    names = getattr(predict_fn, 'class_names', None)
    probe_classes = set(names) if names is not None else None

    body: dict[str, Any] = {
        'size': page_size,
        'query': probe_candidate_query(skip_version=version_tag if resume else None),
        '_source': [
            'crop_id',
            'image_path',
            'bbox_norm',
            'class_name',
        ],
        # Scroll hygiene, no relevance scoring needed here.
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
            # One bulk() per scroll page instead of one update() per
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
                # Image decode and inference are blocking; off the event loop
                # so the worker keeps serving requests and its job heartbeat.
                prediction = await asyncio.to_thread(
                    _crop_and_predict, image_path, bbox, predict_fn, cfg, crop_id
                )
                if prediction is None:
                    continue
                pred_cls, pred_conf, entropy, margin = prediction
                stored = src.get('class_name')
                # A probe trained on a class subset has no opinion on other
                # classes: their items get null, not a disagreement.
                disagreement: bool | None
                if not stored or (probe_classes is not None and stored not in probe_classes):
                    disagreement = None
                else:
                    disagreement = pred_cls != stored
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
            if should_cancel is not None and should_cancel():
                logger.info('probe_cancelled', processed=processed)
                return processed
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
    normalized torch tensor instead.
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
