"""``GET,PUT /curation/settings`` — shared, backend-stored curation-strategy
defaults.

There is no user-account system (single shared instance), so a per-axis
UI default (cluster method, sort order, detection profile, prompt pack —
Cropwright's StrategyBar/AssistScopeBar) is stored once, server-side,
instead of resetting to a hardcoded client default on every reload.

New module (not touching ``methods.py`` / ``review.py`` / the clustering
orchestrator directly). Side-effect import: registers the ``@router``
handlers on the shared ``_common.router``.

The consistency guarantee this module exists to uphold: ``GET
/methods``'s per-axis ``default`` flag and every real endpoint that
applies a hardcoded default when a request omits that axis's param both
read through the same
:func:`~src.services.curation.strategy_defaults.resolve_effective_default`
this module's ``PUT`` writes into. Setting a shared default here changes
actual server behavior, not just what ``/methods`` displays — see
docs/design/curation_api_contract.md's settings section for the exact
call sites that were rewired.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.core.logging import get_logger
from src.routers.curation._common import (
    CurationSettingsResponse,
    CurationSettingsUpdateRequest,
    OpenSearchDep,
    _ensure_indexes,
    router,
)


logger = get_logger(__name__)


async def _validate_defaults(defaults: dict[str, str | None], opensearch: Any) -> None:
    """422 with a clear, valid-ids-listing message for any axis/id pair a
    ``PUT`` can't actually honor.

    Two independent checks, both against
    :data:`~src.services.curation.strategy_defaults.SETTABLE_DEFAULT_AXES`
    and the *live* registry (``get_registry``) rather than a hand-maintained
    copy, so this can never drift from what ``GET /methods`` itself
    advertises:

    1. ``axis`` must be one of the axes a shared default can actually
       change (``SETTABLE_DEFAULT_AXES`` -- ``score``/``overlay``/``export``
       have no single-selectable-id "default" concept today; see that
       constant's docstring).
    2. ``id`` must be a currently-advertised id for that axis -- unless
       ``value`` is ``None``, which always validates: clearing an axis's
       override never needs an id, that's the whole point of clearing it.
    3. An id with a ``requires_field`` must have nonzero coverage of it
       (unknown coverage passes): a default ordering by a field no item
       has orders nothing. Request-time fallback
       (``review_sorts.build_sort``) still covers a pin whose coverage
       later drops to 0, or one stored before this check.
    """
    from src.services.curation.strategy_defaults import SETTABLE_DEFAULT_AXES
    from src.services.curation.strategy_registry import get_registry, invalidate_field_coverage

    # A PUT is rare; judge coverage on live counts, not a TTL-cached one
    # taken before a backfill finished.
    invalidate_field_coverage()
    registry = await get_registry(opensearch)
    ids_by_axis: dict[str, set[str]] = {}
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in registry['strategies']:
        ids_by_axis.setdefault(entry['axis'], set()).add(entry['id'])
        entries[(entry['axis'], entry['id'])] = entry

    errors: list[str] = []
    for axis, value in defaults.items():
        if axis not in SETTABLE_DEFAULT_AXES:
            errors.append(
                f'axis {axis!r} does not accept a shared default; '
                f'valid axes: {sorted(SETTABLE_DEFAULT_AXES)}'
            )
            continue
        if value is None:
            continue
        valid_ids = ids_by_axis.get(axis, set())
        if value not in valid_ids:
            errors.append(
                f'{value!r} is not a currently-advertised id for axis {axis!r}; '
                f'valid ids: {sorted(valid_ids)}'
            )
            continue
        entry = entries[(axis, value)]
        if entry.get('requires_field') and entry.get('field_coverage') == 0:
            errors.append(
                f'{value!r} orders by {entry["requires_field"]!r}, which no item has yet '
                f'(0 of {entry.get("field_coverage_total")}); a deployment default '
                'must order by a field items carry'
            )

    if errors:
        raise HTTPException(status_code=422, detail='; '.join(errors))


@router.get('/settings', response_model=CurationSettingsResponse)
async def get_curation_settings_route(opensearch: OpenSearchDep) -> CurationSettingsResponse:
    """Current shared curation-strategy defaults.

    No document yet (nothing has ever been ``PUT``) is not an error --
    it means "no shared override for any axis," reported as
    ``defaults: {}``, ``updated_at: null``, ``updated_by: null``.
    """
    from src.clients.curation_opensearch import get_curation_settings

    await _ensure_indexes(opensearch)
    doc = await get_curation_settings(opensearch)
    return CurationSettingsResponse(**doc)


@router.put('/settings', response_model=CurationSettingsResponse)
async def update_curation_settings_route(
    body: CurationSettingsUpdateRequest, opensearch: OpenSearchDep
) -> CurationSettingsResponse:
    """Partially update the shared curation-strategy defaults.

    Only the axes present in ``body.defaults`` are validated and merged;
    axes already stored (and not mentioned here) are left untouched. Each
    axis must be one of
    :data:`~src.services.curation.strategy_defaults.SETTABLE_DEFAULT_AXES`
    and each id must be currently advertised for that axis on
    ``GET /methods`` — otherwise this returns ``422`` listing the valid
    axes/ids. A value of ``null`` clears that axis's shared override
    (see ``CurationSettingsUpdateRequest``'s docstring) — the axis simply
    won't appear in the next ``GET``'s ``defaults`` map. ``updated_by`` is
    always ``None`` — no user-account system exists yet.
    """
    from src.clients.curation_opensearch import update_curation_settings

    await _ensure_indexes(opensearch)
    await _validate_defaults(body.defaults, opensearch)
    doc = await update_curation_settings(opensearch, body.defaults)
    logger.info('curation_settings_updated', axes=sorted(body.defaults))
    return CurationSettingsResponse(**doc)
