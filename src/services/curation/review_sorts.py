"""Review-queue sort/filter strategy registry (curation-strategy plan §3.2).

Mirrors ``cluster_methods/__init__.py``'s ``get_method``/``available_methods``
pattern, but for the *sort* axis — a :class:`ReviewSort` never assigns a crop
to a cluster or mutates any doc; it only picks the OpenSearch ``sort`` clause
``GET /curation/review/{tab}`` uses to order an already-built query.

Before this module existed, the review router hardcoded a ``sort = [...]``
literal per tab. ``default_sort_for_tab`` is now the single source of truth
for what each of the 9 existing tabs used to hardcode — its return values
MUST byte-match those legacy literals exactly (the golden-body regression
guard in ``tests/curation/test_review_sorts.py`` is the enforcement).

Selection semantics (plan §3.6 / §7 Phase 3): ``?sort`` absent or literal
``'default'`` resolves per-tab and always succeeds. An explicit, *unknown*
``sort_id``, or one whose current :class:`ReviewSort.status` is ``'shadow'``
or ``'disabled'``, is a real user-facing selection error — :func:`build_sort`
raises :class:`ValueError` so the router turns it into
``HTTPException(400, ...)`` ("shadow entries are computed+logged but
``?sort=<shadow-id>`` returns 400", plan §3). This is deliberately NOT a
silent fallback to the tab default — that would hide a mistaken/stale
client request behind a result that looks fine but isn't what was asked for.
Graceful degradation for a sort whose backing field simply isn't backfilled
yet (0% coverage) is a *different* concern the frontend owns per Phase 0
(only offer sorts ``/curation/scores/coverage`` reports nonzero for) — nothing in
this module invents that behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config.region_fields import get_region_fields


if TYPE_CHECKING:
    from src.services.curation.strategy_registry import StrategyStatus


@dataclass(frozen=True)
class ReviewSort:
    """One selectable (or not-yet-selectable) review-queue sort strategy."""

    id: str
    label: str
    clause: list[dict[str, Any]]
    """OpenSearch ``sort`` body fragment. Empty for the ``'default'``
    sentinel entry, which never reaches OpenSearch as-is — ``build_sort``
    intercepts it before that would happen."""
    requires_field: str | None
    """Field whose presence this sort actually depends on for a meaningful
    order (``None`` for fields like ``updated_at`` every crop always has).
    Informational only here — the frontend uses ``/curation/scores/coverage`` on
    this field to decide whether to offer the sort at all (plan §5)."""
    status: StrategyStatus
    description: str


def _mistakenness_status() -> StrategyStatus:
    """``mistakenness``'s status must reflect whatever
    ``strategy_registry.py``'s live ``OP_SCORES_ENABLED``/``OP_SCORES_SHADOW``
    + ``VALIDATED_SCORERS`` promotion currently computes — NOT a hardcoded
    ``'experimental'`` literal, even though that promotion is real today
    (docs/design/curation_scores.md §3: synthetic label-flip gate passed
    outright). Delegates to the shared helper so this can never drift from
    what ``GET /curation/methods`` reports for the same scorer id."""
    from src.services.curation.strategy_registry import effective_scorer_status

    return effective_scorer_status('mistakenness')


def _build_review_sorts() -> dict[str, ReviewSort]:
    """Construct a fresh registry snapshot. Called by :func:`get_review_sorts`
    on every access (not cached at import) so ``mistakenness``'s env-driven
    status is always current — mirrors ``strategy_registry._score_strategy_status``'s
    "read fresh each call" convention, needed so tests can
    ``monkeypatch.setenv`` around a ``build_sort()`` call without a module
    reload."""
    fields = get_region_fields()
    entries: list[ReviewSort] = [
        ReviewSort(
            id='default',
            label='Default (per tab)',
            clause=[],
            requires_field=None,
            status='stable',
            description=(
                "Delegates to this tab's built-in legacy default sort — see "
                'default_sort_for_tab(). Equivalent to omitting ?sort entirely.'
            ),
        ),
        ReviewSort(
            id='recent',
            label='Recently updated',
            clause=[{'updated_at': {'order': 'desc'}}],
            requires_field=None,
            status='stable',
            description=(
                'Most recently touched crop first. Legacy default for the '
                'mismatches and gemma_low_conf tabs.'
            ),
        ),
        ReviewSort(
            id='representativeness',
            label='Representativeness',
            clause=[
                {
                    'cluster_distance': {
                        'order': 'asc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                }
            ],
            requires_field='cluster_distance',
            status='stable',
            description=(
                'Closest to the IVF cluster centroid first (plan §2.1) — proximity '
                'to centroid, i.e. FiftyOne compute_representativeness renamed and '
                'exposed. Zero new math: same cluster_distance field atypicality '
                'uses, just ascending instead of descending.'
            ),
        ),
        ReviewSort(
            id='atypicality',
            label='Atypicality (outlier-first)',
            clause=[
                {
                    'cluster_distance': {
                        'order': 'desc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                }
            ],
            requires_field='cluster_distance',
            status='stable',
            description=(
                "Farthest from the IVF cluster centroid first — today's outliers "
                "tab and the 'all' tab's ordering, unchanged, just named."
            ),
        ),
        ReviewSort(
            id='uncertainty_entropy',
            label='Uncertainty (probe entropy)',
            clause=[
                {
                    'probe_pred_entropy': {
                        'order': 'desc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                }
            ],
            requires_field='probe_pred_entropy',
            status='stable',
            description=(
                'Highest active-learning probe entropy first — real per-class '
                'entropy since the Phase 1 probe-posterior fix. Legacy default '
                'for the uncertainty tab.'
            ),
        ),
        ReviewSort(
            id='mistakenness',
            label='Mistakenness',
            clause=[
                {
                    'mistakenness_score': {
                        'order': 'desc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                }
            ],
            requires_field='mistakenness_score',
            status=_mistakenness_status(),
            description=(
                'Confident-learning mistakenness (Northcutt/Jiang/Chuang, JAIR '
                '2021). The only scorer whose complete Phase 2 gate passed '
                '(synthetic 5% label-flip: AUROC=0.997, precision@100=0.98 at '
                'n=5000 — docs/design/curation_scores.md §3); status here always '
                "mirrors strategy_registry.py's live promotion, never hardcoded."
            ),
        ),
        ReviewSort(
            id='uniqueness',
            label='Uniqueness',
            clause=[
                {
                    'uniqueness_score': {
                        'order': 'desc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                }
            ],
            requires_field='uniqueness_score',
            status='shadow',
            description=(
                'k-NN density uniqueness. Spearman pre-screen passed on real data '
                '(rho=0.384 >= 0.25 bar) but the real gate — a blind 200-vs-200 '
                'operator A/B — has not run (docs/design/curation_scores.md §2). '
                'Stays shadow (never selectable via ?sort) until that gate clears; '
                'unlike mistakenness this is NOT tied to OP_SCORES_ENABLED/SHADOW '
                '— the validation gap is the reason, not the feature flag.'
            ),
        ),
        ReviewSort(
            id='region_score',
            label='Region detection score',
            clause=[
                {
                    fields.score: {
                        'order': 'desc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                }
            ],
            requires_field=fields.score,
            status='stable',
            description=(
                'Highest-confidence region detection first. Legacy default for the plates tab.'
            ),
        ),
        ReviewSort(
            id='disagreement_entropy_asc',
            label='Model disagreement (most confident first)',
            clause=[
                {
                    'probe_pred_entropy': {
                        'order': 'asc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                }
            ],
            requires_field='probe_pred_entropy',
            status='stable',
            description=(
                'Lowest probe entropy (most confident disagreement) first. Legacy '
                'default for the model_disagreements tab.'
            ),
        ),
        ReviewSort(
            id='primary_low_conf_default',
            label='Largest subject, least confident',
            clause=[
                {
                    'crop_area_norm': {
                        'order': 'desc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                },
                {
                    'v6_raw_confidence': {
                        'order': 'asc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                },
            ],
            requires_field='crop_area_norm',
            status='stable',
            description='Legacy default for the primary_low_conf tab.',
        ),
        ReviewSort(
            id='coco_blind_spots_default',
            label='Largest COCO blind spot',
            clause=[
                {
                    'crop_area_norm': {
                        'order': 'desc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                },
                {
                    'confidence': {
                        'order': 'desc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                },
            ],
            requires_field='crop_area_norm',
            status='stable',
            description='Legacy default for the coco_blind_spots tab.',
        ),
    ]
    return {rs.id: rs for rs in entries}


REVIEW_SORTS: dict[str, ReviewSort] = _build_review_sorts()
"""Import-time snapshot — fine for introspection/docs, but ``build_sort``
calls :func:`get_review_sorts` internally so ``mistakenness``'s status is
never stale relative to live ``OP_SCORES_ENABLED``/``OP_SCORES_SHADOW``."""


def get_review_sorts() -> dict[str, ReviewSort]:
    """Fresh registry snapshot — see :func:`_build_review_sorts`."""
    return _build_review_sorts()


_TAB_DEFAULTS: dict[str, str] = {
    'all': 'atypicality',
    'mismatches': 'recent',
    'gemma_low_conf': 'recent',
    'outliers': 'atypicality',
    'uncertainty': 'uncertainty_entropy',
    'plates': 'region_score',
    'model_disagreements': 'disagreement_entropy_asc',
    'primary_low_conf': 'primary_low_conf_default',
    'coco_blind_spots': 'coco_blind_spots_default',
}
"""The literal legacy ``sort = [...]`` each of the 9 ``GET /curation/review/{tab}``
tabs hardcoded before this registry existed, keyed by the sort id whose
``clause`` byte-matches it. Read directly off ``kb_review.py`` — do not
edit without re-checking the router against this table."""


def default_sort_for_tab(tab: str) -> str:
    """The legacy default sort id for ``tab``. Raises :class:`ValueError`
    for a tab this registry doesn't know about (should never happen in
    practice — ``kb_review.py`` validates ``tab`` against its own known
    set before this is ever called)."""
    try:
        return _TAB_DEFAULTS[tab]
    except KeyError as exc:
        raise ValueError(f'no default review sort registered for tab {tab!r}') from exc


def build_sort(sort_id: str | None, *, tab: str) -> tuple[list[dict[str, Any]], str, str | None]:
    """Resolve a ``?sort=`` query value for ``tab`` into an OpenSearch sort
    clause.

    Returns ``(clause, applied_id, fallback_reason)``:

    * ``sort_id`` is ``None`` or ``'default'`` → resolves via
      :func:`default_sort_for_tab`; always succeeds; ``fallback_reason`` is
      ``None``. This is the byte-identical-to-legacy path every existing
      tab must hit when a client doesn't pass ``?sort`` at all.
    * ``sort_id`` names a ``'stable'``/``'experimental'`` entry → that
      entry's clause is used; ``fallback_reason`` is ``None``.
    * ``sort_id`` is unknown, or names a ``'shadow'``/``'disabled'`` entry →
      raises :class:`ValueError` (the router converts this to
      ``HTTPException(400, ...)``). This is an explicit user-facing
      selection error, not a silent fallback (plan §3/§3.6) — the caller
      asked for something that either doesn't exist or isn't ready, and
      the honest response is "that's not a valid choice," not quietly
      substituting the tab default.

    ``fallback_reason`` is always ``None`` in this phase — the field exists
    per the plan's response-envelope contract so the frontend can render
    it, but nothing in this module currently produces a non-``None`` value;
    graceful degradation for an unbackfilled field is the frontend's job
    (plan §0/§5: only offer a sort once ``/curation/scores/coverage`` reports
    nonzero for its ``requires_field``), not a backend fallback path.
    """
    registry = get_review_sorts()

    if sort_id is None or sort_id == 'default':
        applied_id = default_sort_for_tab(tab)
        return list(registry[applied_id].clause), applied_id, None

    rs = registry.get(sort_id)
    if rs is None:
        raise ValueError(f'unknown review sort {sort_id!r}; valid: {sorted(registry)}')
    if rs.status in ('shadow', 'disabled'):
        raise ValueError(f'review sort {sort_id!r} is not selectable (status={rs.status!r})')
    return list(rs.clause), rs.id, None


__all__ = [
    'REVIEW_SORTS',
    'ReviewSort',
    'build_sort',
    'default_sort_for_tab',
    'get_review_sorts',
]
