"""Mistakenness scorer — confident-learning margin
(curation-strategy plan §2.4).

Northcutt/Jiang/Chuang, "Confident Learning" (JAIR 2021 / cleanlab):
``mistakenness = p(ŷ) - p(y_stored)`` for a crop where the probe's top-1
prediction ``ŷ`` disagrees with the stored label ``y_stored``. This is the
published formulation behind the existing (hand-rolled) "sort
model_disagreements by entropy ascending" heuristic in ``kb_review.py`` —
Phase 1 doesn't touch that tab (plan §8 non-goal #6); this scorer feeds the
*new* additive ``/curation/scores/*`` surface instead.

Two layers, because of a real persistence constraint:

* :func:`compute_mistakenness_from_posterior` — the canonical formula, given
  a full ``(n, nc)`` class-posterior matrix. This is what
  ``tests/curation/test_mistakenness.py`` exercises (synthetic label-flip,
  AUROC acceptance bar) — it needs the true posterior to be a fair test of
  the *math*, independent of what we happen to persist per-crop today.
* :func:`compute_mistakenness_from_margin` — the **production** path. Per
  plan §4.1 we persist only scalar probe summaries per crop
  (``probe_pred_confidence`` = p(ŷ), ``probe_pred_margin`` = p(ŷ) - p(second))
  — not the full ``nc``-length posterior (would bloat the index; not in the
  plan's field list). When the probe agrees with the stored label,
  mistakenness is defined as 0 (probe/human perfectly consistent — no
  evidence of a bad label). When it disagrees, we don't know p(y_stored)
  exactly, but if the stored label is the probe's rank-2 class (the common
  near-miss-confusion case), p(y_stored) ≈ p(ŷ) - margin, which makes
  ``mistakenness ≈ probe_pred_margin``. If the stored label is ranked lower
  than 2, the true p(y_stored) is even smaller, so this *underestimates*
  mistakenness — a conservative, documented approximation, not a silent
  one. Resolved as part of the class-posterior blocker fix in
  ``probe_predictions.py`` (plan §2.2/§2.4/§10.1).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from src.config.curation import IndexRole, get_curation_config, index_name
from src.services.curation.item_scores.base import ScoreResult


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


MISTAKENNESS_VERSION = 'v1'
VEHICLE_CROPS_INDEX = index_name(get_curation_config(), IndexRole.ITEMS)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def compute_mistakenness_from_posterior(
    probs: np.ndarray,
    stored_label_idx: np.ndarray,
) -> np.ndarray:
    """Canonical confident-learning margin: ``p(argmax) - p(stored)``.

    Args:
        probs: ``(n, nc)`` float, each row a class-posterior (need not be
            pre-normalized; this function does not assume it sums to 1 —
            callers should pass an already-normalized posterior for the
            score to be meaningful as a probability difference).
        stored_label_idx: ``(n,)`` int, column index of the stored label
            per row.

    Returns:
        ``(n,)`` float32. 0 when the stored label already IS the top-1
        prediction (probe and stored label agree — p(argmax) == p(stored));
        positive when they disagree, larger when the probe is more
        confident in a different class than the one recorded.
    """
    n = probs.shape[0]
    top1 = probs.max(axis=1)
    stored_p = probs[np.arange(n), stored_label_idx]
    return (top1 - stored_p).astype(np.float32)


def compute_mistakenness_from_margin(
    agrees: np.ndarray,
    probe_pred_confidence: np.ndarray,
    probe_pred_margin: np.ndarray,
) -> np.ndarray:
    """Production approximation from persisted scalar probe fields only.

    ``agrees[i]`` True -> 0.0. False -> ``probe_pred_margin[i]`` (see module
    docstring for the rank-2 approximation this relies on).
    """
    out = np.where(agrees, 0.0, probe_pred_margin)
    # Confidence isn't used in the approximation itself (margin already
    # encodes p(top1) - p(top2)) but a NaN/negative confidence signals a
    # bad upstream write — zero those out defensively rather than
    # propagating garbage into a review-queue sort.
    out = np.where(np.isfinite(probe_pred_confidence) & (probe_pred_confidence >= 0), out, 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class MistakennessScorer:
    """``crop_scores`` registry entry. Ignores the shared embedding matrix —
    reads ``probe_pred_class`` / ``probe_pred_confidence`` / ``probe_pred_margin``
    / ``class_name`` per crop instead (the class-posterior blocker fix in
    ``probe_predictions.py`` is what makes ``probe_pred_margin`` real)."""

    name: ClassVar[str] = 'mistakenness'
    writes: ClassVar[tuple[str, ...]] = (
        'mistakenness_score',
        'mistakenness_method',
        'mistakenness_version',
        'mistakenness_scored_at',
    )
    version: ClassVar[str] = MISTAKENNESS_VERSION

    async def score(
        self,
        ids: list[str],
        embeddings: np.ndarray,  # noqa: ARG002 - protocol uniformity; not embedding-based
        *,
        opensearch: AsyncOpenSearch | None = None,
        progress: Any = None,  # noqa: ARG002 - protocol uniformity
    ) -> ScoreResult:
        if opensearch is None:
            raise ValueError('mistakenness scorer requires an opensearch client')
        if not ids:
            now = _now_iso()
            return ScoreResult(scorer=self.name, version=self.version, scored_at=now, fields={})

        from src.clients.curation_opensearch import mget_crops

        docs = await mget_crops(
            opensearch,
            ids,
            source_includes=[
                'class_name',
                'probe_pred_class',
                'probe_pred_confidence',
                'probe_pred_margin',
            ],
        )

        scored_ids: list[str] = []
        agrees: list[bool] = []
        confs: list[float] = []
        margins: list[float] = []
        for crop_id in ids:
            doc = docs.get(crop_id)
            if not doc:
                continue
            src = doc.get('_source') or {}
            pred_class = src.get('probe_pred_class')
            conf = src.get('probe_pred_confidence')
            margin = src.get('probe_pred_margin')
            if pred_class is None or conf is None or margin is None:
                continue  # not yet probe-scored — skip, not zero (absence != agreement)
            scored_ids.append(crop_id)
            agrees.append(pred_class == src.get('class_name'))
            confs.append(float(conf))
            margins.append(float(margin))

        now = _now_iso()
        if not scored_ids:
            return ScoreResult(scorer=self.name, version=self.version, scored_at=now, fields={})

        scores = compute_mistakenness_from_margin(
            np.asarray(agrees, dtype=bool),
            np.asarray(confs, dtype=np.float32),
            np.asarray(margins, dtype=np.float32),
        )
        fields: dict[str, dict[str, Any]] = {
            crop_id: {
                'mistakenness_score': float(scores[i]),
                'mistakenness_method': 'confident_learning_margin_approx',
                'mistakenness_version': self.version,
                'mistakenness_scored_at': now,
            }
            for i, crop_id in enumerate(scored_ids)
        }
        return ScoreResult(
            scorer=self.name,
            version=self.version,
            scored_at=now,
            fields=fields,
            n_scored=len(fields),
            extra={'n_skipped_unscored_by_probe': len(ids) - len(scored_ids)},
        )


__all__ = [
    'MISTAKENNESS_VERSION',
    'MistakennessScorer',
    'compute_mistakenness_from_margin',
    'compute_mistakenness_from_posterior',
]
