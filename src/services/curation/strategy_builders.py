"""Per-axis strategy entry builders for ``GET /curation/methods``.

Each builder returns the list of entries for one axis (cluster, sort,
score, overlay, export, prompt_pack). Heavy dependencies (cluster
methods, scorers, review sorts) are imported lazily so the registry stays
import-safe in a minimal environment.
"""

from __future__ import annotations

from typing import Any

from src.services.curation.diverse_limits import DIVERSE_BROWSE_MAX_K, DIVERSE_SELECT_MAX_K
from src.services.curation.strategy_flags import (
    VIZ_PROJECTION_PURITY,
    VIZ_PROJECTION_REQUIRES_BANNER,
    VIZ_PROJECTION_SHIP_MODE,
    StrategyStatus,
    _select_diverse_enabled,
    _semantic_search_enabled,
    _viz_projection_enabled,
    effective_scorer_status,
)
from src.services.detection.frame_dedup import (
    DEFAULT_FRAME_DEDUP_THRESHOLD,
    FRAME_DEDUP_THRESHOLD_MAX,
    FRAME_DEDUP_THRESHOLD_MIN,
)


def _cluster_strategies(default_id: str | None) -> list[dict[str, Any]]:
    """Reflect the real ``cluster_methods`` registry — never hand-duplicated
    names, so this can't drift from ``get_method``/``available_methods``.
    ``default_id`` is :func:`resolve_effective_default`'s answer for the
    ``'cluster'`` axis, resolved once by the caller (``get_registry``) so
    every axis's default resolution happens against the same settings-doc
    snapshot within one request."""
    from src.services.curation.clustering.methods import available_methods

    return [
        {
            'id': name,
            'axis': 'cluster',
            'label': name.upper() if name in {'ivf', 'ahc'} else name.replace('_', ' ').title(),
            'status': 'stable',
            'default': name == default_id,
        }
        for name in available_methods()
    ]


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


def _sort_strategies(default_id: str | None) -> list[dict[str, Any]]:
    """Phase 3 ``review_sorts.py`` registry entries — the ``'sort'`` axis this module's ``StrategyAxis`` type has
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
    differently per review tab — so surfacing it here would
    just be a confusing, always-present duplicate of whichever tab-specific
    entry is actually in effect for the tab currently open.

    ``default_id`` is
    :func:`resolve_effective_default`'s answer for the ``'sort'`` axis —
    an *additional*, opt-in global default an operator can set via ``PUT
    /curation/settings`` on top of the untouched per-tab defaults; ``None``
    when no override is set (the pre-existing, always-``False`` behavior)."""
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
                'default': sort.id == default_id,
                'requires_field': sort.requires_field,
            }
        )
    return entries


def _overlay_strategies() -> list[dict[str, Any]]:
    """Phase 4 ``selection/`` overlays.
    One entry today: ``diverse`` (k-center-greedy). Status tracks
    ``OP_SELECT_DIVERSE_ENABLED`` the same live-read pattern
    ``effective_scorer_status`` uses for the score axis, but capped at
    ``experimental`` — never ``stable`` — regardless of the flag, because
    only the cheap pre-screen has passed
    (1.50x >= 1.3x bar); the full
    training A/B gate (mAP50-95 >= +1.0pt via ``/curation/bakeoff``) has not run,
    and diversity is not promotable past
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
            'max_k': DIVERSE_BROWSE_MAX_K,
            'select_max_k': DIVERSE_SELECT_MAX_K,
        },
        *_viz_projection_strategy(),
        *_semantic_search_strategy(),
    ]


def _semantic_search_strategy() -> list[dict[str, Any]]:
    """Semantic-search overlay — PE-Core text-to-image kNN search.
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
    UMAP protocol has two halves (2-d neighborhood purity +
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
    """Phase 5 ``embedding_viz.py`` overlay.
    One entry: ``viz_projection`` (2-d UMAP, visualization
    only — never feeds clustering, see ``embedding_viz.py``'s module
    docstring). ``purity``/``requires_banner`` are the fields
    the UMAP-viz validation calls for: a
    future frontend reads ``requires_banner`` to decide whether to render
    the required persistent "projection is approximate" banner,
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


# What the export request's ``dedup_threshold`` accepts, and the value a UI
# should pre-fill when the operator turns near-duplicate collapsing on (the
# request itself defaults to ``None`` = off).
_DEDUP_FIELDS: dict[str, float] = {
    'dedup_threshold_default': DEFAULT_FRAME_DEDUP_THRESHOLD,
    'dedup_threshold_min': FRAME_DEDUP_THRESHOLD_MIN,
    'dedup_threshold_max': FRAME_DEDUP_THRESHOLD_MAX,
}


def _export_strategies() -> list[dict[str, Any]]:
    """Dataset-export capability axis.

    Advertises which export *kinds* ``POST {prefix}/export/{kind}`` can
    actually produce on this deployment, so a consumer gates an export UI
    on capability rather than probing a write endpoint (a single-class
    export route would kick off a real dataset build) with
    a throwaway request just to see whether it 404s.

    ``single_class`` takes the target class ids from the request rather
    than hardcoding a domain vocabulary, so it covers any narrowed-export
    use case without this repo carrying a deployment-specific overlay.
    There is deliberately no domain-named export kind id — that would be
    exactly the hardcoding this axis exists to avoid.
    """
    return [
        {
            'id': 'yolo',
            'axis': 'export',
            'label': 'YOLO detection dataset export',
            'status': 'stable',
            'default': True,
            **_DEDUP_FIELDS,
        },
        {
            'id': 'single_class',
            'axis': 'export',
            'label': 'Single-class / class-subset dataset export',
            'status': 'stable',
            'default': False,
            **_DEDUP_FIELDS,
        },
    ]


def _prompt_pack_strategies(default_id: str | None) -> list[dict[str, Any]]:
    """Selectable VLM ``PromptPack`` axis.

    Lists every pack :func:`~src.services.labeling.vlm_prompts.
    available_prompt_packs` can load, keyed by pack ``name``: the built-in
    generic pack, each ``OP_PROMPT_PACK_PATHS`` pack, and the
    ``OP_PROMPT_PACK_PATH`` pack (the process default). ``default_id`` is
    :func:`resolve_effective_default`'s answer for the ``'prompt_pack'``
    axis -- the config store's active pack, else the
    ``OP_PROMPT_PACK_PATH`` pack (or the generic pack when unset). A
    run selects among these by name (e.g. ``POST
    /pipeline/auto_label/start?prompt_pack=``).
    """
    from src.services.labeling.vlm_prompts import available_prompt_packs

    return [
        {
            'id': name,
            'axis': 'prompt_pack',
            'label': name,
            'status': 'stable',
            'default': name == default_id,
        }
        for name in available_prompt_packs()
    ]
