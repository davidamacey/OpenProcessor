"""Single source of truth for ``GET /curation/methods`` (curation-strategy plan §3).

Phase 0 reports only what exists in production today: the real
``cluster_methods`` registry entries (``ivf`` / ``ahc`` / ``hdbscan``), all
``status='stable'`` except ``ivf`` which also carries ``default=True``
(mirrors :data:`src.services.curation.clustering.methods.DEFAULT_METHOD` — that
constant does NOT move, see plan §8 non-goal #1).

Phase 1 adds placeholder entries for the new ``crop_scores/`` scorers
(``uniqueness`` / ``mistakenness`` / ``near_dup``). Their ``status`` tracks
the ``OP_SCORES_ENABLED`` feature flag: ``'disabled'`` until an operator
opts in, ``'shadow'`` (computed + logged, not selectable — ``OP_SCORES_SHADOW``)
once enabled-but-shadow, ``'experimental'`` once out of shadow. The frontend
renders only ``stable``/``experimental`` entries (plan §3, "capability-
discovery linchpin"); ``shadow``/``disabled`` entries are never offered as a
``?sort=`` value.

Phase 2 (curation-strategy plan §6/§9, 2026-09-10) ran every method's
go/no-go validation protocol against the real ~350k-crop pool (see
``docs/design/curation_scores.md`` for the full writeup + numbers).
``VALIDATED_SCORERS`` below promotes the scorers whose *complete* gate
passed (no human-in-the-loop or GPU-training step left unexecuted) one
notch above the flag-driven status computed for the rest — i.e. from
``shadow`` to ``experimental`` while ``OP_SCORES_SHADOW`` is still set.
Only ``mistakenness`` qualifies today: its full gate (synthetic 5%
label-flip AUROC >= 0.80 *and* precision@100 >= 0.50) is a pure synthetic
check with no human/GPU step, and both bars passed. ``uniqueness`` and
``near_dup`` passed their cheap pre-screens on real data (Spearman
rho=0.38 vs >=0.25; 100% class-coverage retention across a 0.95-0.99
threshold sweep) but each method's plan-table gate also requires a step
this validation pass could not execute (a blind operator A/B for
uniqueness; 50 manually-judged pairs per threshold for near_dup) — they
stay at whatever the flag-driven status says (``shadow``/``disabled``)
until that step runs. This never overrides ``OP_SCORES_ENABLED=false``
(disabled stays disabled regardless of validation history — the flag is
a master kill switch, not a per-method opt-in).

This module is deliberately dependency-light (no OpenSearch, no faiss) so
``GET /curation/methods`` never fails or blocks — it just reflects config + the
static cluster-method registry.
"""

from __future__ import annotations

import os
import time
from typing import Any, Literal

from src.core.logging import get_logger


logger = get_logger(__name__)

StrategyStatus = Literal['stable', 'experimental', 'shadow', 'disabled']
StrategyAxis = Literal[
    'cluster', 'score', 'sort', 'overlay', 'export', 'detection_profile', 'prompt_pack'
]


def _scores_enabled() -> bool:
    """Read fresh each call (not a module constant) so tests can
    ``monkeypatch.setenv`` without reimporting — matches the
    ``train_jobs._resolve_jobs_dir`` convention in this repo."""
    return os.environ.get('OP_SCORES_ENABLED', '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _scores_shadow() -> bool:
    return os.environ.get('OP_SCORES_SHADOW', '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _select_diverse_enabled() -> bool:
    """Mirrors ``kb_select.py``'s own flag check — kept independent (not
    imported from there) so this dependency-light module never needs to
    import a router module just to read one env var."""
    return os.environ.get('OP_SELECT_DIVERSE_ENABLED', '').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def _semantic_search_enabled() -> bool:
    """Mirrors ``kb_semantic.py``'s own flag check — same "don't import a
    router module just to read one env var" reasoning as
    ``_select_diverse_enabled``/``_viz_projection_enabled``."""
    return os.environ.get('OP_SEMANTIC_SEARCH_ENABLED', '').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def _viz_projection_enabled() -> bool:
    """Mirrors ``kb_viz.py``'s own flag check — same "don't import a
    router module just to read one env var" reasoning as
    ``_select_diverse_enabled``."""
    return os.environ.get('OP_VIZ_PROJECTION_ENABLED', '').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


# Curation-strategy plan §6's UMAP row / §7 Phase 5 — the one validation
# protocol Phase 2 explicitly skipped ("Phase 5 scope",
# docs/design/curation_scores.md §7) and this pass ran for real: 2-d
# neighborhood purity vs. the real FAISS IVF-assigned ``cluster_id``, a
# real random 10,000-crop sample (561 distinct real cluster_id buckets --
# see docs/design/curation_scores.md's new UMAP-viz section for the full
# write-up, including why the *first* attempt at this measurement was
# discarded: a plain scroll-order 10k pull hit only 66 distinct buckets,
# an artifact of single-shard insertion-order bias, not a real geometry
# result). Bar (plan §6): >=0.30 ship plain; 0.15-0.30 ship w/ banner;
# <0.15 do not ship.
VIZ_PROJECTION_PURITY = 0.472
"""Measured mean 10-NN 2-d neighborhood purity vs real ``cluster_id``,
random n=10,000 sample, the offline purity-evaluation script. Clears the
>=0.30 "ship plain" bar with room to spare — nowhere near either the
0.15-0.30 banner tier or the <0.15 kill-switch tier."""

VIZ_PROJECTION_SHIP_MODE: StrategyStatus | str = 'ship_plain'
"""One of ``'ship_plain'`` / ``'ship_with_banner'`` / ``'do_not_ship'`` —
derived from :data:`VIZ_PROJECTION_PURITY` against the plan §6 bar. Kept as
an explicit constant (not re-derived at import time) so the number that was
actually measured and the shipping decision that was actually made are
both visible side-by-side in this file, the same way
``VALIDATED_SCORERS``' docstring pairs the Phase 2 numbers with the
promotion decision they justify."""

VIZ_PROJECTION_REQUIRES_BANNER = VIZ_PROJECTION_SHIP_MODE == 'ship_with_banner'
"""Surfaced on the ``viz_projection`` entry in ``GET /curation/methods`` as
``requires_banner`` — the field a future frontend reads to decide whether
to render the plan's required persistent "projection is approximate"
banner. Named to match this file's existing boolean-flag vocabulary
(``default``, ``kb_scores_enabled``, ...) rather than inventing a
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


VALIDATED_SCORERS: frozenset[str] = frozenset({'mistakenness'})
"""Scorer ids whose Phase 2 validation (curation-strategy plan §6) passed
the *complete* gate in ``docs/design/curation_scores.md`` -- promoted one
notch above the flag-driven status (``shadow`` -> ``experimental``) so an
operator running with ``OP_SCORES_SHADOW=1`` still sees it as selectable.
Deliberately NOT ``uniqueness``/``near_dup``: their pre-screens passed on
real data but the plan's full gate for each needs a step this validation
pass couldn't execute (human blind A/B; manually-judged near-dup pairs) --
see the doc for exact numbers before adding anything here."""


def _cluster_strategies() -> list[dict[str, Any]]:
    """Reflect the real ``cluster_methods`` registry — never hand-duplicated
    names, so this can't drift from ``get_method``/``available_methods``."""
    from src.services.curation.clustering.methods import DEFAULT_METHOD, available_methods

    return [
        {
            'id': name,
            'axis': 'cluster',
            'label': name.upper() if name in {'ivf', 'ahc'} else name.replace('_', ' ').title(),
            'status': 'stable',
            'default': name == DEFAULT_METHOD,
        }
        for name in available_methods()
    ]


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


def _score_strategies() -> list[dict[str, Any]]:
    """Phase 1 crop_scores/ registry entries. Field/scorer names mirror
    ``src.services.curation.item_scores.available_scorers()`` — imported
    lazily so this module never pulls in faiss/numpy unless a caller
    actually wants the score axis."""
    try:
        from src.services.curation.item_scores import SCORER_METADATA
    except ImportError:
        # crop_scores/ not importable (e.g. faiss missing in a minimal
        # environment) — degrade to an empty score axis rather than
        # failing /curation/methods entirely.
        return []

    entries: list[dict[str, Any]] = []
    for scorer_id, meta in SCORER_METADATA.items():
        entries.append(
            {
                'id': scorer_id,
                'axis': 'score',
                'label': meta['label'],
                'status': effective_scorer_status(scorer_id),
                'default': False,
                'requires_field': meta.get('requires_field'),
                'writes': list(meta.get('writes', ())),
            }
        )
    return entries


def _sort_strategies() -> list[dict[str, Any]]:
    """Phase 3 ``review_sorts.py`` registry entries (curation-strategy plan
    §3.2) — the ``'sort'`` axis this module's ``StrategyAxis`` type has
    declared since Phase 3 but ``get_registry()`` never actually populated
    until now (found via a live end-to-end check: real ``GET /curation/methods``
    reported zero sort-axis entries, so nothing this registry marks
    ``stable``/``experimental`` — ``representativeness``, ``atypicality``,
    ``uncertainty_entropy``, etc. — was ever discoverable by the frontend).

    Imported lazily, same reasoning ``_score_strategies()`` uses for
    ``crop_scores``: this module stays import-safe even in an environment
    where ``review_sorts.py``'s dependency chain isn't available.

    The ``'default'`` sentinel id is excluded on purpose. It isn't a real,
    independently-selectable sort — ``default_sort_for_tab`` resolves it
    differently per review tab (plan §3.2) — so surfacing it here would
    just be a confusing, always-present duplicate of whichever tab-specific
    entry is actually in effect for the tab currently open."""
    try:
        from src.services.curation.review_sorts import get_review_sorts
    except ImportError:
        return []

    entries: list[dict[str, Any]] = []
    for sort_id, sort in get_review_sorts().items():
        if sort_id == 'default':
            continue
        entries.append(
            {
                'id': sort.id,
                'axis': 'sort',
                'label': sort.label,
                'status': sort.status,
                'default': False,
                'requires_field': sort.requires_field,
            }
        )
    return entries


def _overlay_strategies() -> list[dict[str, Any]]:
    """Phase 4 ``selection/`` overlays (curation-strategy plan §2.6/§3.4).
    One entry today: ``diverse`` (k-center-greedy). Status tracks
    ``OP_SELECT_DIVERSE_ENABLED`` the same live-read pattern
    ``effective_scorer_status`` uses for the score axis, but capped at
    ``experimental`` — never ``stable`` — regardless of the flag, because
    only the cheap pre-screen has passed
    (``docs/design/curation_scores.md`` §6, 1.50x >= 1.3x bar); the full
    training A/B gate (mAP50-95 >= +1.0pt via ``/curation/bakeoff``) has not run,
    and plan §6/§10.2 is explicit that diversity is not promotable past
    ``experimental`` without it."""
    return [
        {
            'id': 'diverse',
            'axis': 'overlay',
            'label': 'Diversity (k-center-greedy)',
            'status': 'experimental' if _select_diverse_enabled() else 'disabled',
            'default': False,
            'requires_field': None,
            'writes': [],
        },
        *_viz_projection_strategy(),
        *_semantic_search_strategy(),
    ]


def _semantic_search_strategy() -> list[dict[str, Any]]:
    """P2-14 ``kb_semantic.py`` overlay — PE-Core text-to-image kNN search.
    Same "flag on -> experimental, flag off -> disabled, never stable
    without a full validation gate" shape ``diverse`` uses above: no
    real-usage go/no-go protocol has run yet, so it can never surface as
    anything past ``experimental`` regardless of the flag. Degrades
    invisibly exactly like ``diverse``/``viz_projection`` — a caller that
    doesn't special-case this id sees the same shape either way, and the
    frontend only renders ``stable``/``experimental`` entries anyway."""
    return [
        {
            'id': 'semantic_search',
            'axis': 'overlay',
            'label': 'Semantic Text Search (PE-Core)',
            'status': 'experimental' if _semantic_search_enabled() else 'disabled',
            'default': False,
            'requires_field': 'pe_embedding',
            'writes': [],
        }
    ]


def _viz_projection_status() -> StrategyStatus:
    """``do_not_ship`` is a hard kill switch — stays ``disabled`` even if
    an operator sets ``OP_VIZ_PROJECTION_ENABLED=1`` (same "flag can only
    turn a validated thing on, never revive a failed one" rule
    ``VALIDATED_SCORERS`` enforces for the score axis). Otherwise tracks
    the flag, capped at ``experimental`` — never ``stable`` — because the
    plan §6 UMAP protocol has two halves (2-d neighborhood purity +
    interactive perf) and this backend-only pass could only execute the
    first; the labeler's ``EmbeddingPlot.svelte`` interactive-perf check
    is Phase 5's frontend half and hasn't run yet (same reasoning
    ``diverse``'s permanent experimental cap uses for its own
    still-outstanding gate half)."""
    if VIZ_PROJECTION_SHIP_MODE == 'do_not_ship':
        return 'disabled'
    if not _viz_projection_enabled():
        return 'disabled'
    return 'experimental'


def _viz_projection_strategy() -> list[dict[str, Any]]:
    """Phase 5 ``embedding_viz.py`` overlay (curation-strategy plan
    §2.7/§3.5). One entry: ``viz_projection`` (2-d UMAP, visualization
    only — never feeds clustering, see ``embedding_viz.py``'s module
    docstring). ``purity``/``requires_banner`` are the fields
    docs/design/curation_scores.md's new UMAP-viz section calls for: a
    future frontend reads ``requires_banner`` to decide whether to render
    the plan's required persistent "projection is approximate" banner,
    and ``purity`` so the number backing that decision is never hidden
    behind just a status string."""
    return [
        {
            'id': 'viz_projection',
            'axis': 'overlay',
            'label': 'UMAP Projection (viz only)',
            'status': _viz_projection_status(),
            'default': False,
            'requires_field': 'viz_x',
            'writes': ['viz_x', 'viz_y', 'viz_projection_version'],
            'purity': VIZ_PROJECTION_PURITY,
            'requires_banner': VIZ_PROJECTION_REQUIRES_BANNER,
        }
    ]


def _export_strategies() -> list[dict[str, Any]]:
    """Dataset-export capability axis (cropwright_backend_integration_plan.md
    §4.3/T-C2).

    Advertises which export *kinds* ``POST {prefix}/export/{kind}`` can
    actually produce on this deployment, so a consumer gates an export UI
    on capability rather than probing a write endpoint (``POST
    /export/lpr`` would kick off a real dataset build) with a throwaway
    request just to see whether it 404s.

    ``lpr`` — the reference implementation's proprietary single-class
    license-plate export (Bucket B, out of scope — never ported) — is
    deliberately absent from this list rather than listed with
    ``status='disabled'``: a status implies "not yet, but this deployment
    could serve it later", which isn't true for a proprietary overlay this
    repo doesn't contain.
    """
    return [
        {
            'id': 'yolo',
            'axis': 'export',
            'label': 'YOLO detection dataset export',
            'status': 'stable',
            'default': True,
        }
    ]


def _detection_profile_strategies() -> list[dict[str, Any]]:
    """Configured sub-region ``DetectionProfile`` axis (labeling-assist
    plan task (b)).

    Reads :mod:`src.services.detection.profile_registry` -- a real,
    process-lifetime registry a deployment can add more than one profile
    to (e.g. a license-plate profile AND a shipping-label profile) --
    rather than hardcoding the single ``DEFAULT_PROFILE`` here. Today
    exactly one profile is ever registered (importing
    ``src.services.detection.cascade_detect`` registers its own
    ``DEFAULT_PROFILE`` as the default), so this axis lists exactly one
    entry, but the mechanism is not limited to one.
    """
    # Import triggers cascade_detect's module-level `register_profile`
    # call if it hasn't run yet in this process.
    from src.services.detection import cascade_detect  # noqa: F401
    from src.services.detection.profile_registry import get_default_profile_name, get_profiles

    default_name = get_default_profile_name()
    return [
        {
            'id': profile.name,
            'axis': 'detection_profile',
            'label': profile.name,
            'status': 'stable',
            'default': profile.name == default_name,
        }
        for profile in get_profiles().values()
    ]


def _prompt_pack_strategies() -> list[dict[str, Any]]:
    """Configured VLM ``PromptPack`` axis (labeling-assist plan task (c)).

    Lists whatever pack :func:`~src.services.labeling.vlm_prompts.
    resolve_prompt_pack` actually resolves for this process -- a
    deployment-supplied pack via ``OP_PROMPT_PACK_PATH``, or the built-in
    generic pack when unset/missing. Always exactly one entry (there is
    only ever one active pack per process), always ``default=True``.
    """
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    pack = resolve_prompt_pack()
    return [
        {
            'id': pack.name,
            'axis': 'prompt_pack',
            'label': pack.name,
            'status': 'stable',
            'default': True,
        }
    ]


_COVERAGE_CACHE: dict[str, int | None] | None = None
_COVERAGE_CACHE_AT = 0.0
_COVERAGE_TTL_S = float(os.environ.get('OP_FIELD_COVERAGE_TTL_S', '60'))
_COVERAGE_TOTAL_KEY = '__total__'
"""Sentinel key the pool-size count is cached under, alongside the
per-field exists counts, in the same dict -- avoids a second cache
structure just to hold one extra number."""


async def _compute_field_coverage(opensearch: Any, fields: frozenset[str]) -> dict[str, int | None]:
    """``{field: exists_count}`` for every field in ``fields``, plus the
    pool total under :data:`_COVERAGE_TOTAL_KEY` -- the real fix for P1-2/
    P1-3 (audit-remediation plan Phase 6). Mirrors
    ``crop_scores/job.py::compute_coverage``'s query shape (one
    ``opensearch.count`` per field, one ``match_all`` count for the
    denominator) but keyed by raw field name across every axis, not just
    the three ``crop_scores`` scorer fields -- ``/curation/scores/coverage`` has
    no answer for ``probe_pred_entropy``/``cluster_distance``/
    ``plate_score``/``crop_area_norm``, which is exactly where the inert
    sorts live (see this module's Phase 6 note above).

    TTL-cached at module scope (``_COVERAGE_CACHE`` / ``time.monotonic()``
    freshness check), the same pattern ``kb_select.py``'s ``_ORDER_CACHE``
    and ``cluster_outliers.py``'s ``_CACHE`` use -- ``GET /curation/methods``
    must stay an O(1)-per-request endpoint (its own docstring promises it
    "never fails or blocks"), not an O(distinct-fields) OpenSearch round
    trip on every page load.

    A field whose count query fails is cached as ``None`` -- "unknown",
    never ``0`` ("empty") -- so a transient OpenSearch hiccup can never
    make a working sort/chip disappear (plan Phase 6: "fall back to None
    (not 0) on OpenSearch failure, so a transient error hides nothing that
    already works"). The whole helper never raises; callers get a dict
    with real ints, ``None``s, or (only if OpenSearch is totally
    unreachable for the total-count call) a ``None`` total too.
    """
    global _COVERAGE_CACHE, _COVERAGE_CACHE_AT  # noqa: PLW0603 - module-level TTL cache, same pattern as select.py

    now = time.monotonic()
    if (
        _COVERAGE_CACHE is not None
        and (now - _COVERAGE_CACHE_AT) < _COVERAGE_TTL_S
        and fields <= (_COVERAGE_CACHE.keys() - {_COVERAGE_TOTAL_KEY})
    ):
        return _COVERAGE_CACHE

    from src.config.curation import IndexRole, get_curation_config, index_name

    index = index_name(get_curation_config(), IndexRole.ITEMS)
    counts: dict[str, int | None] = {}

    try:
        total_resp = await opensearch.count(index=index, body={'query': {'match_all': {}}})
        counts[_COVERAGE_TOTAL_KEY] = int(total_resp.get('count', 0))
    except Exception as exc:
        logger.warning('kb_methods_field_coverage_total_failed', error=str(exc))
        counts[_COVERAGE_TOTAL_KEY] = None

    for field in sorted(fields):
        try:
            resp = await opensearch.count(index=index, body={'query': {'exists': {'field': field}}})
            counts[field] = int(resp.get('count', 0))
        except Exception as exc:
            logger.warning('kb_methods_field_coverage_count_failed', field=field, error=str(exc))
            counts[field] = None

    _COVERAGE_CACHE = counts
    _COVERAGE_CACHE_AT = now
    return counts


def _reset_field_coverage_cache() -> None:
    """Test-only escape hatch -- the module-level TTL cache otherwise leaks
    across test cases that monkeypatch a fake OpenSearch client per-test."""
    global _COVERAGE_CACHE, _COVERAGE_CACHE_AT  # noqa: PLW0603
    _COVERAGE_CACHE = None
    _COVERAGE_CACHE_AT = 0.0


async def get_registry(opensearch: Any | None = None) -> dict[str, Any]:
    """Full ``/curation/methods`` payload: every strategy across every axis.

    ``opensearch`` is optional so every existing caller (docs, scripts,
    other tests) that only wants the config/flag-driven shape keeps
    working without a client. When provided, each entry that declares a
    ``requires_field`` gets a real ``field_coverage`` (exists-count) and
    ``field_coverage_total`` (pool size) computed via
    :func:`_compute_field_coverage` -- Phase 6's fix for the frontend
    conflating "coverage unknown" with "coverage zero". Entries with no
    ``requires_field`` (cluster methods, the ``'default'``/``'recent'``
    sorts) always carry ``field_coverage: None`` -- coverage doesn't apply
    to a field every crop always has.
    """
    strategies = [
        *_cluster_strategies(),
        *_sort_strategies(),
        *_score_strategies(),
        *_overlay_strategies(),
        *_export_strategies(),
        *_detection_profile_strategies(),
        *_prompt_pack_strategies(),
    ]

    fields = frozenset(e['requires_field'] for e in strategies if e.get('requires_field'))

    coverage: dict[str, int | None] = {}
    if opensearch is not None and fields:
        try:
            coverage = await _compute_field_coverage(opensearch, fields)
        except Exception as exc:
            # Defense in depth: _compute_field_coverage already catches
            # per-query failures internally and never raises in practice,
            # but this module's whole reason for being dependency-light is
            # "GET /curation/methods never fails or blocks" -- honor that even
            # against a totally broken client (e.g. one whose .count
            # attribute isn't even callable).
            logger.warning('kb_methods_field_coverage_failed', error=str(exc))
            coverage = {}

    total = coverage.get(_COVERAGE_TOTAL_KEY)
    for entry in strategies:
        field = entry.get('requires_field')
        if field:
            entry['field_coverage'] = coverage.get(field)
            entry['field_coverage_total'] = total
        else:
            entry['field_coverage'] = None
            entry['field_coverage_total'] = None

    return {
        'strategies': strategies,
        'flags': {
            'kb_scores_enabled': _scores_enabled(),
            'kb_scores_shadow': _scores_shadow(),
            'kb_select_diverse_enabled': _select_diverse_enabled(),
            'kb_viz_projection_enabled': _viz_projection_enabled(),
            'kb_semantic_search_enabled': _semantic_search_enabled(),
        },
    }


__all__ = [
    'VALIDATED_SCORERS',
    'VIZ_PROJECTION_PURITY',
    'VIZ_PROJECTION_REQUIRES_BANNER',
    'VIZ_PROJECTION_SHIP_MODE',
    'StrategyAxis',
    'StrategyStatus',
    'effective_scorer_status',
    'get_registry',
]
