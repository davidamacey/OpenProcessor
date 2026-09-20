"""Shared-settings default resolution for the curation-strategy axes
(curation deployment-settings plan).

Split out of :mod:`src.services.curation.strategy_registry` (which grew
past the pre-commit 700-LOC ratchet once this landed) rather than
grandfathered onto that file — this is a genuinely separate concern:
:mod:`strategy_registry` builds the full ``GET /methods`` payload (every
axis's entries, status, field coverage); this module answers one narrow
question, "what id should axis X resolve to right now," backed by the
single shared curation-settings document
(``src.clients.curation_opensearch.get_curation_settings``).

:func:`resolve_effective_default` is re-exported from
``strategy_registry`` (and imported directly from here by
``review_sorts.py`` / the clustering orchestrator / the settings router)
so it remains the ONE function both ``GET /methods``'s per-axis
``default`` flag and every real endpoint's omitted-param resolution call
— see docs/design/curation_api_contract.md's settings section for the
full list of call sites.
"""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger


logger = get_logger(__name__)


def _hardcoded_default_for_axis(axis: str) -> str | None:
    """The pre-settings-feature hardcoded default id for ``axis`` --
    :func:`resolve_effective_default`'s fallback when no shared-settings
    override applies. Only the four axes the shared-defaults feature
    covers (curation deployment-settings plan) have an entry; every other
    axis (``score``/``overlay``/``export``) has no single-id "default"
    concept and returns ``None`` here, same as before this feature
    existed.

    ``'sort'`` deliberately returns ``None`` -- there has never been one
    hardcoded default sort id for the whole axis, only a per-tab mapping
    (:func:`~src.services.curation.review_sorts.default_sort_for_tab`).
    The per-tab defaults are untouched; a shared-settings override for
    ``'sort'`` is an *additional*, opt-in global choice layered on top
    (see :mod:`src.services.curation.review_sorts`'s ``build_sort``),
    not a replacement for them.
    """
    if axis == 'cluster':
        from src.services.curation.clustering.methods import DEFAULT_METHOD

        return DEFAULT_METHOD
    if axis == 'detection_profile':
        from src.services.detection import cascade_detect  # noqa: F401 - registers DEFAULT_PROFILE
        from src.services.detection.profile_registry import get_default_profile_name

        return get_default_profile_name()
    if axis == 'prompt_pack':
        from src.services.labeling.vlm_prompts import resolve_prompt_pack

        return resolve_prompt_pack().name
    return None


def _advertised_ids_for_axis(axis: str) -> frozenset[str]:
    """Every id currently selectable for ``axis`` -- what a shared-settings
    override must belong to for :func:`resolve_effective_default` to honor
    it rather than falling back. For ``'sort'`` this deliberately excludes
    the ``'shadow'``/``'disabled'`` entries (and the ``'default'``
    sentinel) -- an override that names a not-yet-selectable sort would
    otherwise silently break every tab's queue instead of falling back."""
    if axis == 'cluster':
        from src.services.curation.clustering.methods import available_methods

        return frozenset(available_methods())
    if axis == 'sort':
        from src.services.curation.review_sorts import get_review_sorts

        return frozenset(
            sort_id
            for sort_id, rs in get_review_sorts().items()
            if sort_id != 'default' and rs.status in ('stable', 'experimental')
        )
    if axis == 'detection_profile':
        from src.services.detection import cascade_detect  # noqa: F401 - registers DEFAULT_PROFILE
        from src.services.detection.profile_registry import get_profiles

        return frozenset(get_profiles())
    if axis == 'prompt_pack':
        from src.services.labeling.vlm_prompts import resolve_prompt_pack

        return frozenset({resolve_prompt_pack().name})
    return frozenset()


# Axes a shared-settings override can actually change (curation
# deployment-settings plan). Deliberately a subset of every axis
# GET /methods advertises: 'score'/'overlay'/'export' have no
# single-selectable-id "default" concept a shared override could apply to
# today (score/overlay are additive, not mutually-exclusive choices; export
# has exactly one kind), so PUT /curation/settings rejects them rather than
# silently accepting a value nothing will ever honor.
SETTABLE_DEFAULT_AXES: frozenset[str] = frozenset(
    {'cluster', 'sort', 'detection_profile', 'prompt_pack'}
)


async def resolve_effective_default(
    axis: str, opensearch: Any | None = None, *, settings_doc: dict[str, Any] | None = None
) -> str | None:
    """The effective default strategy id for ``axis`` right now.

    Looks up the shared curation-settings document's ``defaults.get(axis)``
    (see ``src.routers.curation.settings`` / ``GET,PUT /curation/settings``);
    if that override is present AND still a currently-advertised id for
    this axis, returns it. Otherwise falls back to ``axis``'s hardcoded
    default constant (:func:`_hardcoded_default_for_axis`) -- the same
    fallback this axis used before the shared-settings feature existed, so
    an operator who never touches ``PUT /curation/settings``, or whose
    stored override has since become invalid (the axis id it names is no
    longer advertised), sees byte-identical behavior to before this
    function existed.

    ``opensearch=None`` skips the settings lookup entirely and returns the
    hardcoded default straight away -- this mirrors ``get_registry``'s own
    "opensearch is optional" contract (docs, scripts, and tests that only
    want the config/flag-driven shape keep working with no client), and it
    is also what keeps this module dependency-light: a caller that only
    wants "the hardcoded default id" never has to touch OpenSearch.

    ``settings_doc``, if given, is used instead of fetching -- lets a
    caller resolving several axes in one request (``get_registry``) read
    the settings document exactly once and reuse it, rather than issuing
    one OpenSearch ``get`` per axis for what is, by construction, always
    the same document.

    This is the ONE place both ``GET /methods``'s per-axis ``default``
    flag and every real endpoint that applies an omitted axis param must
    call -- see docs/design/curation_api_contract.md's settings section
    for the exact call sites.
    """
    hardcoded = _hardcoded_default_for_axis(axis)
    if opensearch is None and settings_doc is None:
        return hardcoded

    if settings_doc is None:
        try:
            from src.clients.curation_opensearch import get_curation_settings

            settings_doc = await get_curation_settings(opensearch)
        except Exception as exc:
            logger.warning('legacy_settings_resolve_default_failed', axis=axis, error=str(exc))
            return hardcoded

    override = (settings_doc.get('defaults') or {}).get(axis)
    if not override:
        return hardcoded
    if override in _advertised_ids_for_axis(axis):
        return override
    return hardcoded


__all__ = [
    'SETTABLE_DEFAULT_AXES',
    'resolve_effective_default',
]
