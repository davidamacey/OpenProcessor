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
A *resolved default* (the tab's own, or the deployment-pinned one) whose
backing field no item carries (0% coverage) is different: it would order
nothing, so :func:`build_sort` walks the tab's fallback chain
(:data:`_TAB_FALLBACKS`, always ending at ``'recent'``) and reports why in
``fallback_reason``. An explicit ``?sort=`` is honored as asked.
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
                'mismatches and vlm_low_conf tabs.'
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
                },
                # A verifier-rejected candidate never has `region_score`
                # (only `region_candidate_score`) -- without this second
                # key every rejected item ties on the first key's
                # `missing: '_last'` and falls back to shard order among
                # themselves. Ordering by the candidate's own score keeps
                # them sorted sanely instead of an arbitrary tie; it never
                # changes the order of items that DO have region_score,
                # since that first key already fully orders them.
                {
                    fields.candidate_score: {
                        'order': 'desc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                },
            ],
            requires_field=fields.score,
            status='stable',
            description=(
                'Highest-confidence region detection first (falling back to a '
                "rejected candidate's own score when there is no accepted "
                'region score). Legacy default for the regions tab.'
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
                    # D-1 (F-6): classifier_raw_confidence is never written
                    # in production -- sort on the stored `confidence`
                    # field instead (ascending: least confident first).
                    'confidence': {
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
            id='classifier_blind_spots_default',
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
            description='Legacy default for the classifier_blind_spots tab.',
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
    'vlm_low_conf': 'recent',
    'outliers': 'atypicality',
    'uncertainty': 'uncertainty_entropy',
    'regions': 'region_score',
    'model_disagreements': 'disagreement_entropy_asc',
    'primary_low_conf': 'primary_low_conf_default',
    'classifier_blind_spots': 'classifier_blind_spots_default',
}
"""The literal legacy ``sort = [...]`` each of the 9 ``GET /curation/review/{tab}``
tabs hardcoded before this registry existed, keyed by the sort id whose
``clause`` byte-matches it. Read directly off the reference review-queries module — do not
edit without re-checking the router against this table."""


_TAB_FALLBACKS: dict[str, tuple[str, ...]] = {
    'all': ('mistakenness',),
    'uncertainty': ('mistakenness', 'atypicality'),
}
"""Sorts tried, in order, after a resolved default with 0% field coverage
and before the terminal ``'recent'`` (``updated_at``, which every item
has). Entries not currently selectable (shadow/disabled) are skipped."""


async def _first_covered(
    primary: str, tab: str, registry: dict[str, ReviewSort], opensearch: Any
) -> tuple[str, str | None]:
    """``(sort_id, fallback_reason)``: ``primary`` unless its field has
    zero coverage, else the first covered sort in the tab's chain. Unknown
    coverage (count failed) counts as covered -- a transient error must
    never reorder a queue."""
    chain = [primary]
    for sid in (*_TAB_FALLBACKS.get(tab, ()), 'recent'):
        rs = registry.get(sid)
        if rs is not None and sid not in chain and rs.status in ('stable', 'experimental'):
            chain.append(sid)
    fields = frozenset(f for sid in chain if (f := registry[sid].requires_field))
    if registry[primary].requires_field is None or not fields:
        return primary, None
    from src.services.curation.strategy_registry import field_coverage

    coverage = await field_coverage(opensearch, fields)
    chosen = next(
        (
            sid
            for sid in chain
            if (f := registry[sid].requires_field) is None or coverage.get(f) != 0
        ),
        chain[-1],
    )
    if chosen == primary:
        return primary, None
    field = registry[primary].requires_field
    return chosen, (
        f'default sort {primary!r} orders by {field!r}, which no item has yet; '
        f'using {chosen!r} instead'
    )


def default_sort_for_tab(tab: str) -> str:
    """The legacy default sort id for ``tab``. Raises :class:`ValueError`
    for a tab this registry doesn't know about (should never happen in
    practice — the reference review-queries module validates ``tab`` against its own known
    set before this is ever called)."""
    try:
        return _TAB_DEFAULTS[tab]
    except KeyError as exc:
        raise ValueError(f'no default review sort registered for tab {tab!r}') from exc


async def build_sort(
    sort_id: str | None, *, tab: str, opensearch: Any | None = None
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Resolve a ``?sort=`` query value for ``tab`` into an OpenSearch sort
    clause.

    Returns ``(clause, applied_id, fallback_reason)``:

    * ``sort_id`` is ``None`` or ``'default'`` → the tab's own default
      (:func:`default_sort_for_tab`) when it has one; otherwise the
      shared-settings override for the ``'sort'`` axis (see
      ``src.services.curation.strategy_registry.resolve_effective_default``);
      otherwise ``'recent'``. A deployment default never overrides a
      tab's own default. Always succeeds; ``fallback_reason`` is ``None``.
    * ``sort_id`` names a ``'stable'``/``'experimental'`` entry → that
      entry's clause is used; ``fallback_reason`` is ``None``.
    * ``sort_id`` is unknown, or names a ``'shadow'``/``'disabled'`` entry →
      raises :class:`ValueError` (the router converts this to
      ``HTTPException(400, ...)``). This is an explicit user-facing
      selection error, not a silent fallback (plan §3/§3.6) — the caller
      asked for something that either doesn't exist or isn't ready, and
      the honest response is "that's not a valid choice," not quietly
      substituting the tab default.

    ``fallback_reason`` is non-``None`` only when a resolved default (the
    first bullet) orders by a field with 0% coverage in the items index:
    the returned clause/id are then the first covered sort in the tab's
    fallback chain (see :func:`_first_covered`). Needs ``opensearch``.

    ``opensearch``, when given, is threaded into
    :func:`~src.services.curation.strategy_registry.resolve_effective_default`
    for a tab without its own default. ``None`` skips the lookup.
    """
    registry = get_review_sorts()

    if sort_id is None or sort_id == 'default':
        # A tab's own default wins: it is tuned to what the tab surfaces (the
        # regions tab sorts by region score; a deployment default on a field
        # the tab's items may not carry would scramble it). A deployment
        # default applies only to a tab without one.
        applied_id = _TAB_DEFAULTS.get(tab)
        if applied_id is None and opensearch is not None:
            from src.services.curation.strategy_registry import resolve_effective_default

            applied_id = await resolve_effective_default('sort', opensearch)
        if applied_id is None:
            applied_id = 'recent'
        reason = None
        if opensearch is not None:
            applied_id, reason = await _first_covered(applied_id, tab, registry, opensearch)
        return list(registry[applied_id].clause), applied_id, reason

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
