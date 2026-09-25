"""Tests for the mistakenness (confident-learning margin) scorer.

The acceptance-bar test (``test_synthetic_label_flip_auroc``) is the real
gate: synthetic 5% label-flip over n=500 separable synthetic
classes, AUROC >= 0.8 on the canonical ``p(argmax) - p(stored)`` formula. A
second test covers the production scalar-field approximation
(``compute_mistakenness_from_margin``) that ``MistakennessScorer`` actually
uses given only ``probe_pred_confidence``/``probe_pred_margin`` are
persisted per crop (see ``mistakenness.py`` module docstring for why).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.services.curation.item_scores.mistakenness import (
    compute_mistakenness_from_margin,
    compute_mistakenness_from_posterior,
)


def _synthetic_posteriors(
    n: int = 500,
    n_classes: int = 5,
    dim: int = 8,
    flip_rate: float = 0.05,
    seed: int = 0,
    temperature: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separable synthetic classes: well-spaced prototypes in R^dim, each
    sample a noisy draw from its TRUE class. ``stored_label`` starts equal
    to the true class and ``flip_rate`` of rows get relabeled to a
    different random class (the "human mislabeled it" cohort).

    The posterior is a softmax over negative squared distance to each
    prototype (temperature controls how peaked/confident it is) — this
    approximates what a well-trained probe classifier's output would look
    like, without needing an actual model.

    Returns ``(probs, stored_label_idx, is_mislabeled)``.
    """
    rng = np.random.default_rng(seed)
    prototypes = rng.normal(size=(n_classes, dim)) * 3.0  # spread prototypes apart

    true_labels = rng.integers(0, n_classes, size=n)
    samples = prototypes[true_labels] + rng.normal(scale=1.0, size=(n, dim))

    # Posterior: softmax(-dist^2 / temperature) over prototypes.
    dists_sq = ((samples[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    logits = -dists_sq / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)

    stored_labels = true_labels.copy()
    n_flip = round(n * flip_rate)
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    for i in flip_idx:
        choices = [c for c in range(n_classes) if c != true_labels[i]]
        stored_labels[i] = rng.choice(choices)

    is_mislabeled = (stored_labels != true_labels).astype(int)
    return probs, stored_labels, is_mislabeled


def test_synthetic_label_flip_auroc() -> None:
    """Acceptance bar: AUROC >= 0.80 on synthetic 5% label-flip."""
    probs, stored_labels, is_mislabeled = _synthetic_posteriors()
    assert is_mislabeled.sum() >= 15, 'sanity: flip cohort too small to measure AUROC meaningfully'

    scores = compute_mistakenness_from_posterior(probs, stored_labels)
    assert scores.shape == (500,)
    assert np.all(scores >= -1e-6)  # p(argmax) >= p(anything), including p(stored)

    auroc = roc_auc_score(is_mislabeled, scores)
    assert auroc >= 0.80, f'AUROC {auroc:.3f} below the acceptance bar (0.80)'


def test_agreement_rows_score_zero() -> None:
    """When the stored label IS the top-1 prediction, mistakenness is
    exactly 0 by construction (p(argmax) - p(stored) == 0)."""
    probs, stored_labels, is_mislabeled = _synthetic_posteriors(seed=1)
    scores = compute_mistakenness_from_posterior(probs, stored_labels)
    agree_mask = ~is_mislabeled.astype(bool)
    # Not every agreeing row has stored==argmax (a noisy sample can still
    # have its true/stored class NOT be the posterior's top-1), so we only
    # assert the tautology for rows where stored *is* the top-1 class.
    top1 = probs.argmax(axis=1)
    exact_top1_and_agree = agree_mask & (top1 == stored_labels)
    assert exact_top1_and_agree.sum() > 0
    np.testing.assert_allclose(scores[exact_top1_and_agree], 0.0, atol=1e-6)


# =============================================================================
# Production scalar-field approximation
# =============================================================================


def test_margin_approximation_zero_on_agreement() -> None:
    agrees = np.array([True, True, False, False])
    conf = np.array([0.9, 0.7, 0.6, 0.55])
    margin = np.array([0.3, 0.2, 0.25, 0.1])
    out = compute_mistakenness_from_margin(agrees, conf, margin)
    np.testing.assert_allclose(out[:2], [0.0, 0.0])
    np.testing.assert_allclose(out[2:], margin[2:])


def test_margin_approximation_defensive_against_bad_confidence() -> None:
    agrees = np.array([False, False])
    conf = np.array([np.nan, -1.0])
    margin = np.array([0.5, 0.4])
    out = compute_mistakenness_from_margin(agrees, conf, margin)
    np.testing.assert_allclose(out, [0.0, 0.0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


@pytest.mark.asyncio
async def test_scorer_skips_items_outside_the_probe_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An item whose stored class the probe cannot predict (null
    ``probe_disagreement``) gets no mistakenness score, not a high one."""
    from src.clients import curation_opensearch
    from src.services.curation.item_scores.mistakenness import MistakennessScorer

    docs = {
        'in': {
            '_source': {
                'class_name': 'suv',
                'probe_pred_class': 'sedan',
                'probe_pred_confidence': 0.8,
                'probe_pred_margin': 0.6,
                'probe_disagreement': True,
            }
        },
        'out': {
            '_source': {
                'class_name': 'bus',
                'probe_pred_class': 'sedan',
                'probe_pred_confidence': 0.8,
                'probe_pred_margin': 0.6,
                'probe_disagreement': None,
            }
        },
        'legacy': {
            '_source': {
                'class_name': 'suv',
                'probe_pred_class': 'sedan',
                'probe_pred_confidence': 0.8,
                'probe_pred_margin': 0.6,
            }
        },
    }

    async def fake_mget(_opensearch: object, ids: list[str], **_kwargs: object) -> dict:
        return {i: docs[i] for i in ids}

    monkeypatch.setattr(curation_opensearch, 'mget_crops', fake_mget)

    result = await MistakennessScorer().score(
        ['in', 'out', 'legacy'], np.zeros((3, 1), dtype=np.float32), opensearch=object()
    )

    assert set(result.fields) == {'in', 'legacy'}
    assert result.fields['in']['mistakenness_score'] == pytest.approx(0.6)
