"""Single source of truth for ``GET /curation/methods``.

Phase 0 reports only what exists in production today: the real
``cluster_methods`` registry entries (``ivf`` / ``ahc`` / ``hdbscan``), all
``status='stable'`` except ``ivf`` which also carries ``default=True``
(mirrors :data:`src.services.curation.clustering.methods.DEFAULT_METHOD` — that
constant does NOT move).

Phase 1 adds placeholder entries for the new ``crop_scores/`` scorers
(``uniqueness`` / ``mistakenness`` / ``near_dup``). Their ``status`` tracks
the ``OP_SCORES_ENABLED`` feature flag: ``'disabled'`` until an operator
opts in, ``'shadow'`` (computed + logged, not selectable — ``OP_SCORES_SHADOW``)
once enabled-but-shadow, ``'experimental'`` once out of shadow. The frontend
renders only ``stable``/``experimental`` entries ("capability-
discovery linchpin"); ``shadow``/``disabled`` entries are never offered as a
``?sort=`` value.

A validation pass (2026-09-10) ran every method's
go/no-go validation protocol against the real ~350k-crop pool.
``VALIDATED_SCORERS`` below promotes the scorers whose *complete* gate
passed (no human-in-the-loop or GPU-training step left unexecuted) one
notch above the flag-driven status computed for the rest — i.e. from
``shadow`` to ``experimental`` while ``OP_SCORES_SHADOW`` is still set.
``mistakenness`` and ``uniqueness`` qualify today. ``mistakenness``'s full
gate (synthetic 5% label-flip AUROC >= 0.80 *and* precision@100 >= 0.50)
is a pure synthetic check with no human/GPU step, and both bars passed.
``uniqueness`` passed its cheap pre-screen on real data (Spearman
rho=0.38 vs >=0.25) and, as of the 2026-09-25 F8 acceptance pass, its
scores are computed and populated in production — promoted the same
notch as ``mistakenness`` (see :data:`VALIDATED_SCORERS`). ``near_dup``
passed its own pre-screen (100% class-coverage retention across a
0.95-0.99 threshold sweep) but its full gate still requires a step this
validation pass could not execute (50 manually-judged pairs per
threshold) — it stays at whatever the flag-driven status says
(``shadow``/``disabled``) until that step runs. This never overrides
``OP_SCORES_ENABLED=false`` (disabled stays disabled regardless of
validation history — the flag is a master kill switch, not a per-method
opt-in).

This module is deliberately dependency-light (no OpenSearch, no faiss) so
``GET /curation/methods`` never fails or blocks — it just reflects config + the
static cluster-method registry.
"""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger
from src.services.curation.axis_copy import axis_copy, detection_profile_strategies
from src.services.curation.field_coverage import _COVERAGE_TOTAL_KEY, _compute_field_coverage
from src.services.curation.strategy_builders import (
    _cluster_strategies,
    _export_strategies,
    _overlay_strategies,
    _prompt_pack_strategies,
    _score_strategies,
    _sort_strategies,
)
from src.services.curation.strategy_defaults import SETTABLE_DEFAULT_AXES, resolve_effective_default
from src.services.curation.strategy_flags import (
    _scores_enabled,
    _scores_shadow,
    _select_diverse_enabled,
    _semantic_search_enabled,
    _viz_projection_enabled,
)
from src.services.curation.vlm_strategies import vlm_strategies


logger = get_logger(__name__)


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
    # Fetched at most once per request (None if opensearch is None) so
    # every axis's 'default' flag below reflects a single consistent
    # settings-doc snapshot, and this endpoint issues one settings lookup
    # total rather than one per axis.
    settings_doc: dict[str, Any] | None = None
    if opensearch is not None:
        try:
            from src.clients.curation_opensearch.settings_doc import get_curation_settings

            settings_doc = await get_curation_settings(opensearch)
        except Exception as exc:
            logger.warning('curation_methods_settings_lookup_failed', error=str(exc))
            settings_doc = None

    cluster_default = await resolve_effective_default('cluster', settings_doc=settings_doc)
    sort_default = await resolve_effective_default('sort', settings_doc=settings_doc)
    detection_profile_default = await resolve_effective_default(
        'detection_profile', settings_doc=settings_doc
    )
    prompt_pack_default = await resolve_effective_default('prompt_pack', settings_doc=settings_doc)

    strategies = [
        *_cluster_strategies(cluster_default),
        *_sort_strategies(sort_default),
        *_score_strategies(),
        *_overlay_strategies(),
        *_export_strategies(),
        *detection_profile_strategies(detection_profile_default),
        *_prompt_pack_strategies(prompt_pack_default),
        *await vlm_strategies(opensearch),
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
            logger.warning('curation_methods_field_coverage_failed', error=str(exc))
            coverage = {}

    total = coverage.get(_COVERAGE_TOTAL_KEY)
    for entry in strategies:
        # Whether PUT /settings (and any per-run override) can select this
        # axis; read-only axes (e.g. detection_profile) only report state.
        entry['settable'] = entry['axis'] in SETTABLE_DEFAULT_AXES
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
            'scores_enabled': _scores_enabled(),
            'scores_shadow': _scores_shadow(),
            'select_diverse_enabled': _select_diverse_enabled(),
            'viz_projection_enabled': _viz_projection_enabled(),
            'semantic_search_enabled': _semantic_search_enabled(),
        },
        'axes': axis_copy(),
    }
