"""Review-queue sort/filter strategy registry.

A ``get_method``/``available_methods``-style registry for the *sort* axis — a :class:`ReviewSort` never assigns a crop
to a cluster or mutates any doc; it only picks the OpenSearch ``sort`` clause
``GET /curation/review/{tab}`` uses to order an already-built query.

``default_sort_for_tab`` is the single source of truth for the sort each queue opens with
(``tests/curation/test_review_sorts.py`` pins the clauses).

Selection semantics: ``?sort`` absent or literal
``'default'`` resolves per-tab and always succeeds. An explicit, *unknown*
``sort_id``, or one whose current :class:`ReviewSort.status` is ``'shadow'``
or ``'disabled'``, is a real user-facing selection error — :func:`build_sort`
raises :class:`ValueError` so the router turns it into
``RequestRejectedError(400, ...)`` ("shadow entries are computed+logged but
``?sort=<shadow-id>`` returns 400"). This is deliberately NOT a
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
from src.services.curation.crop_browse import CROP_SORT_LABELS, parse_crop_sort


if TYPE_CHECKING:
    from src.schemas.curation.views import SortOption
    from src.services.curation.strategy_flags import StrategyStatus


# ``vlm_confidence`` is a keyword, so sorting the field would order its words
# (high, low, medium). Hard cases come first: a missing or unrecognised value
# ranks 0, then low, medium, high.
VLM_CONFIDENCE_ORDER: tuple[str, ...] = ('low', 'medium', 'high')
VLM_CONFIDENCE_RANK_SORT: dict[str, Any] = {
    '_script': {
        'type': 'number',
        'order': 'asc',
        'script': {
            'lang': 'painless',
            'source': (
                "doc.containsKey('vlm_confidence') && doc['vlm_confidence'].size() > 0 "
                "? params.ranks.getOrDefault(doc['vlm_confidence'].value, 0) : 0"
            ),
            'params': {'ranks': {v: i + 1 for i, v in enumerate(VLM_CONFIDENCE_ORDER)}},
        },
    }
}


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
    this field to decide whether to offer the sort at all."""
    status: StrategyStatus
    description: str


AUDIT_SAMPLE_ORDER = 'crop_id:asc'
REGION_SCORE_SORT = 'region_score'
"""The served id of the regions queue's score sort; it orders by ``region_max_score``."""


def _crop_sort(sort_id: str, *, requires_field: str) -> ReviewSort:
    """A review sort that is also a ``GET /crops`` sort id, so the queue and the Detections view
    name the same order the same way."""
    field, _, direction = sort_id.partition(':')
    return ReviewSort(
        id=sort_id,
        label=CROP_SORT_LABELS[field][direction],
        clause=parse_crop_sort(sort_id)[1],
        requires_field=requires_field,
        status='stable',
        description=f'The same order as the {sort_id!r} sort of GET /crops.',
    )


def _mistakenness_status() -> StrategyStatus:
    """``mistakenness``'s status must reflect whatever
    ``strategy_flags.py``'s live ``OP_SCORES_ENABLED``/``OP_SCORES_SHADOW``
    + ``VALIDATED_SCORERS`` promotion currently computes — NOT a hardcoded
    ``'experimental'`` literal, even though that promotion is real today
    (synthetic label-flip gate passed
    outright). Delegates to the shared helper so this can never drift from
    what ``GET /curation/methods`` reports for the same scorer id."""
    from src.services.curation.strategy_flags import effective_scorer_status

    return effective_scorer_status('mistakenness')


def _uniqueness_status() -> StrategyStatus:
    """``uniqueness``'s status, same "delegate, never hardcode" shape as
    :func:`_mistakenness_status`. D2 (2026-09-25 F8 acceptance): uniqueness
    was previously pinned to a hardcoded ``'shadow'`` literal here
    regardless of the live ``OP_SCORES_ENABLED``/``OP_SCORES_SHADOW`` +
    ``VALIDATED_SCORERS`` promotion — which meant ``sort=uniqueness`` 400'd
    even once ``strategy_flags.VALIDATED_SCORERS`` promoted it to
    ``'experimental'``, and ``GET /curation/methods`` disagreed with this
    registry about the same scorer's status. Delegating closes that gap."""
    from src.services.curation.strategy_flags import effective_scorer_status

    return effective_scorer_status('uniqueness')


def _build_review_sorts() -> dict[str, ReviewSort]:
    """Construct a fresh registry snapshot. Called by :func:`get_review_sorts`
    on every access (not cached at import) so ``mistakenness``'s env-driven
    status is always current — mirrors ``strategy_flags._score_strategy_status``'s
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
            description=('Most recently touched crop first. The fallback every sort ends at.'),
        ),
        _crop_sort('detector_confidence:desc', requires_field='detector_confidence'),
        _crop_sort('confidence:asc', requires_field='confidence'),
        ReviewSort(
            id=AUDIT_SAMPLE_ORDER,
            label='Sample order',
            # No sort key: the crop_id tiebreak every review query ends in is the order.
            clause=[],
            requires_field=None,
            status='stable',
            description='The audit sample in a stable order. Default for the audit queue.',
        ),
        ReviewSort(
            id='representativeness',
            label='Representativeness',
            clause=[
                {
                    'cluster_score': {
                        'order': 'desc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                }
            ],
            requires_field='cluster_score',
            status='stable',
            description=(
                'Closest to its cluster centroid first: the served cluster_score (cosine to the '
                'centroid of the serving partition), highest first.'
            ),
        ),
        ReviewSort(
            id='atypicality',
            label='Atypicality (outlier-first)',
            clause=[
                {
                    'cluster_score': {
                        'order': 'asc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                }
            ],
            requires_field='cluster_score',
            status='stable',
            description=(
                'Farthest from its cluster centroid first: the served cluster_score, lowest first.'
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
                'entropy since the Phase 1 probe-posterior fix.'
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
                'n=5000); status here always '
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
            status=_uniqueness_status(),
            description=(
                'k-NN density uniqueness. Spearman pre-screen passed on real data '
                '(rho=0.384 >= 0.25 bar); scores are computed and populated in '
                'production. D2 (2026-09-25 F8 acceptance): promoted alongside '
                'mistakenness in strategy_registry.VALIDATED_SCORERS; status here '
                "always mirrors strategy_registry.py's live promotion, never "
                'hardcoded.'
            ),
        ),
        ReviewSort(
            id=REGION_SCORE_SORT,
            label='Region detection score',
            # Sort on the flat `region_max_score` item
            # field, not the legacy `region_score`/`region_candidate_score`
            # scalars. Nothing in the region worker ever writes those two --
            # only dead code (verify.py's now-uncalled doc builders) and
            # human single-box paths do -- so on an index the worker wrote this
            # sort tied on `missing: '_last'` for every item and fell back
            # to the crop_id tiebreak, silently burying high-confidence
            # items last. `region_max_score` is maintained by every box
            # writer (worker, per-box routes, region_meta/batch_status) via
            # `boxes_write_fields`, is flat (needs no nested sort, unlike a
            # per-box field), and already covers a rejected-only item since
            # `boxes_write_fields` takes the max over every box's score
            # regardless of state -- so no second sort key is needed.
            clause=[
                {
                    fields.max_score: {
                        'order': 'desc',
                        'missing': '_last',
                        'unmapped_type': 'double',
                    }
                },
            ],
            requires_field=fields.max_score,
            status='stable',
            description=(
                'Highest-confidence region detection first (falling back to a '
                "rejected candidate's own score when there is no accepted "
                'region score). Default for the regions queue.'
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
                'Lowest probe entropy (most confident disagreement) first. Default for '
                'the disagreements queue.'
            ),
        ),
        ReviewSort(
            id='detector_disagreement_default',
            label='Least VLM confidence first, then detector confidence',
            clause=[
                VLM_CONFIDENCE_RANK_SORT,
                {'confidence': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}},
            ],
            requires_field='vlm_confidence',
            status='stable',
            description=(
                "The VLM's own confidence from low to high (an answer with no recorded "
                "confidence first), then the detector's confidence high to low. The disagreements "
                "queue's fallback when no probe has run."
            ),
        ),
    ]
    return {rs.id: rs for rs in entries}


def get_review_sorts() -> dict[str, ReviewSort]:
    """Fresh registry snapshot — see :func:`_build_review_sorts`."""
    return _build_review_sorts()


_TAB_DEFAULTS: dict[str, str] = {
    'to_label': 'detector_confidence:desc',
    'check_machine': 'confidence:asc',
    'disagreements': 'disagreement_entropy_asc',
    'audit': AUDIT_SAMPLE_ORDER,
    'regions': REGION_SCORE_SORT,
}
"""The sort id each queue opens with. ``new_class_proposals`` has none and takes the deployment
default, else ``recent``."""


_TAB_FALLBACKS: dict[str, tuple[str, ...]] = {
    'disagreements': ('detector_disagreement_default',),
}
"""Sorts tried, in order, after a resolved default with 0% field coverage
and before the terminal ``'recent'`` (``updated_at``, which every item
has). Entries not currently selectable (shadow/disabled) are skipped."""


_FIELD_NEEDS: dict[str, str] = {
    'cluster_score': 'a clustering run',
    'probe_pred_entropy': 'a probe run',
    'mistakenness_score': 'item scores',
    'uniqueness_score': 'item scores',
    'vlm_confidence': 'VLM labels',
}
"""What produces each sort field, in the words the fallback notice uses."""


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
    from src.services.curation.field_coverage import field_coverage

    coverage = await field_coverage(opensearch, fields, live=True)
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
    needs = _FIELD_NEEDS.get(field or '', f'items that carry {field}')
    return chosen, (
        f'Sorted by {registry[chosen].label}. {registry[primary].label!r} needs {needs}.'
    )


_NOT_OFFERED: frozenset[str] = frozenset({'default', 'detector_disagreement_default'})
"""The sentinel and the disagreements queue's fallback order: they name a queue's own order, not
a choice."""

_ONLY_IN: dict[str, frozenset[str]] = {
    REGION_SCORE_SORT: frozenset({'regions'}),
    'disagreement_entropy_asc': frozenset({'disagreements'}),
    AUDIT_SAMPLE_ORDER: frozenset({'audit'}),
}
"""Sorts that only make sense in one queue."""


async def sort_options(tab: str, opensearch: Any) -> list[SortOption]:
    """The sorts ``GET /review/{tab}`` can be read in, as ``{id, label}`` in registry order.

    Selectable entries only (``shadow``/``disabled`` ones would be a 400), minus the sentinel and
    the sorts of another queue, and minus a sort whose field no item carries (confirmed zero
    coverage orders nothing); the queue's own default always stays. Unknown coverage counts as
    covered, like :func:`_first_covered`."""
    from src.services.curation.field_coverage import field_coverage

    default = _TAB_DEFAULTS.get(tab)
    offered = [
        rs
        for rs in get_review_sorts().values()
        if rs.status in ('stable', 'experimental')
        and rs.id not in _NOT_OFFERED
        and tab in _ONLY_IN.get(rs.id, frozenset({tab}))
    ]
    fields = frozenset(rs.requires_field for rs in offered if rs.requires_field)
    coverage = await field_coverage(opensearch, fields, live=True) if fields else {}
    return [
        {'id': rs.id, 'label': rs.label}
        for rs in offered
        if rs.id == default or rs.requires_field is None or coverage.get(rs.requires_field) != 0
    ]


def default_sort_for_tab(tab: str) -> str:
    """The default sort id for the queue ``tab``. Raises :class:`ValueError`
    for a tab this registry doesn't know about (should never happen in
    practice — the review-queries module validates ``tab`` against its own known
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
      ``RequestRejectedError(400, ...)``). This is an explicit user-facing
      selection error, not a silent fallback — the caller
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
    'ReviewSort',
    'build_sort',
    'default_sort_for_tab',
    'get_review_sorts',
    'sort_options',
]
