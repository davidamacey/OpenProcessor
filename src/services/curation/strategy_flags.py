"""Feature-flag readers, strategy status vocabulary and scorer promotion.

Every flag is read fresh per call (not a module constant) so tests can
``monkeypatch.setenv`` without reimporting. The validation record that
justifies :data:`VALIDATED_SCORERS` and the viz-projection ship mode is
documented on the constants below.
"""

from __future__ import annotations

from typing import Literal

from src.config.env import env_bool


StrategyStatus = Literal['stable', 'experimental', 'shadow', 'disabled']


def _scores_enabled() -> bool:
    """Read fresh each call (not a module constant) so tests can
    ``monkeypatch.setenv`` without reimporting — matches the
    ``train_jobs._resolve_jobs_dir`` convention in this repo."""
    return env_bool('OP_SCORES_ENABLED')


def _scores_shadow() -> bool:
    return env_bool('OP_SCORES_SHADOW')


def _select_diverse_enabled() -> bool:
    """Mirrors the select router's own flag check — kept independent (not
    imported from there) so this dependency-light module never needs to
    import a router module just to read one env var."""
    return env_bool('OP_SELECT_DIVERSE_ENABLED')


def _semantic_search_enabled() -> bool:
    """Mirrors the semantic-search module's own flag check — same "don't import a
    router module just to read one env var" reasoning as
    ``_select_diverse_enabled``/``_viz_projection_enabled``."""
    return env_bool('OP_SEMANTIC_SEARCH_ENABLED', default=True)


def _viz_projection_enabled() -> bool:
    """Mirrors the embedding-viz module's own flag check — same "don't import a
    router module just to read one env var" reasoning as
    ``_select_diverse_enabled``."""
    return env_bool('OP_VIZ_PROJECTION_ENABLED')


# The UMAP row is the one validation check the earlier validation pass
# explicitly skipped, and this pass ran for real: 2-d neighborhood purity
# vs. the real FAISS IVF-assigned ``cluster_id``, a real random
# 10,000-crop sample (561 distinct real cluster_id buckets), including why
# the *first* attempt at this measurement was discarded: a plain
# scroll-order 10k
# pull hit only 66 distinct buckets, an artifact of single-shard
# insertion-order bias, not a real geometry result). Bar: >=0.30 ship
# plain; 0.15-0.30 ship w/ banner; <0.15 do not ship.
VIZ_PROJECTION_PURITY = 0.472
"""Measured mean 10-NN 2-d neighborhood purity vs real ``cluster_id``,
random n=10,000 sample, the offline purity-evaluation script. Clears the
>=0.30 "ship plain" bar with room to spare — nowhere near either the
0.15-0.30 banner tier or the <0.15 kill-switch tier."""

VIZ_PROJECTION_SHIP_MODE: StrategyStatus | str = 'ship_plain'
"""One of ``'ship_plain'`` / ``'ship_with_banner'`` / ``'do_not_ship'`` —
derived from :data:`VIZ_PROJECTION_PURITY` against the bar above. Kept as
an explicit constant (not re-derived at import time) so the number that was
actually measured and the shipping decision that was actually made are
both visible side-by-side in this file, the same way
``VALIDATED_SCORERS``' docstring pairs the validation numbers with the
promotion decision they justify."""

VIZ_PROJECTION_REQUIRES_BANNER = VIZ_PROJECTION_SHIP_MODE == 'ship_with_banner'
"""Surfaced on the ``viz_projection`` entry in ``GET /curation/methods`` as
``requires_banner`` — the field a future frontend reads to decide whether
to render the required persistent "projection is approximate"
banner. Named to match this file's existing boolean-flag vocabulary
(``default``, ``scores_enabled``, ...) rather than inventing a
banner/message-string convention with no other precedent in this response
shape. False today because the measured purity landed in the "ship plain"
tier, not the banner tier — flip only by re-running
the offline purity-evaluation script and updating the three constants
above together, never independently."""


def _score_strategy_status() -> StrategyStatus:
    """Phase 1 scorers are 'disabled' until OP_SCORES_ENABLED, then
    'shadow' (computed but not selectable as a sort) until an operator
    also clears OP_SCORES_SHADOW, then 'experimental'."""
    if not _scores_enabled():
        return 'disabled'
    if _scores_shadow():
        return 'shadow'
    return 'experimental'


VALIDATED_SCORERS: frozenset[str] = frozenset({'mistakenness', 'uniqueness'})
"""Scorer ids whose validation passed
the *complete* gate -- promoted one
notch above the flag-driven status (``shadow`` -> ``experimental``) so an
operator running with ``OP_SCORES_SHADOW=1`` still sees it as selectable.

``mistakenness``'s full gate (synthetic 5% label-flip AUROC >= 0.80 *and*
precision@100 >= 0.50) is a pure synthetic check with no human/GPU step,
and both bars passed outright.

``uniqueness`` (2026-09-25 F8 acceptance): scores are computed and
populated in production, and its cheap pre-screen already passed on real
data (Spearman rho=0.384 >= 0.25 bar, 2026-09-10). Promoted here so
``sort=uniqueness`` is selectable and ``review_sorts.py``'s ``uniqueness``
entry (which now delegates to :func:`effective_scorer_status` the same way
``mistakenness`` does, instead of a hardcoded ``'shadow'`` literal) tracks
the same flag-driven status.

Deliberately NOT ``near_dup``: its pre-screen passed on real data (100%
class-coverage retention across a 0.95-0.99 threshold sweep) but its full
gate needs a step this validation pass couldn't execute (50
manually-judged pairs per threshold) -- see the doc for exact numbers
before adding anything here."""


def effective_scorer_status(scorer_id: str) -> StrategyStatus:
    """Current status for one ``crop_scores`` scorer id, applying the same
    ``OP_SCORES_ENABLED``/``OP_SCORES_SHADOW``/``VALIDATED_SCORERS``
    promotion rule ``_score_strategies()`` uses for ``GET /curation/methods``.

    Extracted so other Phase-3 modules (``review_sorts.py``) can ask "what
    is this scorer's status *right now*" without duplicating the
    promotion logic or reaching into a private helper. Read fresh every
    call, same reasoning as ``_score_strategy_status``'s docstring — no
    caller should cache this across a request boundary."""
    status = _score_strategy_status()
    # Promote a validated scorer one notch (shadow -> experimental)
    # without ever bypassing the OP_SCORES_ENABLED master switch.
    if status == 'shadow' and scorer_id in VALIDATED_SCORERS:
        return 'experimental'
    return status
