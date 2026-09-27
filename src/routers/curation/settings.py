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

from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from src.services.config_store.index import ConfigAxis

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


# Axes W2 moved off the generic settings-doc ``defaults`` map onto the
# config store's activation docs (any_domain_plan.md §3.7, §9 W2): a PUT
# here delegates to ``store.activate_axis`` instead of writing into
# ``defaults``, and a GET merges the store's active name back in so the
# response shape is unchanged for callers. See ``_config_store_axes``.
_CONFIG_STORE_AXES = frozenset({'prompt_pack', 'detection_profile'})


async def _config_store_axis_defaults(opensearch: Any) -> dict[str, str | None]:
    """The active name for each config-store-backed axis, for merging
    into ``GET /settings``'s ``defaults`` map."""
    from src.services.config_store import get_config_store

    store = get_config_store()
    await store.ensure_fresh(opensearch)
    snapshot = store.current
    result: dict[str, str | None] = {}
    for axis, ref in (
        ('prompt_pack', snapshot.active_pack),
        ('detection_profile', snapshot.active_profile),
    ):
        if ref is None or ref == 'off':
            continue
        result[axis] = ref[0]
    return result


@router.get('/settings', response_model=CurationSettingsResponse)
async def get_curation_settings_route(opensearch: OpenSearchDep) -> CurationSettingsResponse:
    """Current shared curation-strategy defaults.

    No document yet (nothing has ever been ``PUT``) is not an error --
    it means "no shared override for any axis," reported as
    ``defaults: {}``, ``updated_at: null``, ``updated_by: null``.

    ``prompt_pack`` / ``detection_profile`` are reported from the config
    store's activation (W2), not this settings document -- see
    :func:`_config_store_axis_defaults`.
    """
    from src.clients.curation_opensearch import get_curation_settings

    await _ensure_indexes(opensearch)
    doc = await get_curation_settings(opensearch)
    doc['defaults'] = {**doc.get('defaults', {}), **await _config_store_axis_defaults(opensearch)}
    return CurationSettingsResponse(**doc)


@router.put('/settings', response_model=CurationSettingsResponse)
async def update_curation_settings_route(
    body: CurationSettingsUpdateRequest, opensearch: OpenSearchDep
) -> CurationSettingsResponse:
    """Partially update the shared curation-strategy defaults.

    ``prompt_pack`` / ``detection_profile`` in ``body.defaults`` delegate
    to the config store's activation (W2, any_domain_plan.md §3.7): the
    id must be currently advertised for the axis (or ``off`` for
    ``detection_profile``), else ``422``; ``null`` deactivates the axis
    through the store rather than clearing a settings-doc override. Every
    other axis keeps the prior behavior: it must be one of
    :data:`~src.services.curation.strategy_defaults.SETTABLE_DEFAULT_AXES`
    and each id must be currently advertised for that axis on
    ``GET /methods``. ``updated_by`` is always ``None`` — no user-account
    system exists yet.
    """
    from src.clients.curation_opensearch import update_curation_settings

    await _ensure_indexes(opensearch)
    config_store_defaults = {k: v for k, v in body.defaults.items() if k in _CONFIG_STORE_AXES}
    settings_doc_defaults = {k: v for k, v in body.defaults.items() if k not in _CONFIG_STORE_AXES}

    await _validate_defaults(settings_doc_defaults, opensearch)
    for axis, value in config_store_defaults.items():
        await _activate_config_store_axis(axis, value, opensearch)

    doc = await update_curation_settings(opensearch, settings_doc_defaults)
    doc['defaults'] = {**doc.get('defaults', {}), **await _config_store_axis_defaults(opensearch)}
    logger.info('curation_settings_updated', axes=sorted(body.defaults))
    return CurationSettingsResponse(**doc)


async def _activate_config_store_axis(axis: str, value: str | None, opensearch: Any) -> None:
    """``PUT /settings {"defaults": {axis: value}}`` for a config-store
    axis: resolve ``value`` (or ``None``/``'off'``) against the live
    registry, then activate the latest revision through the store."""
    from src.routers.curation._config_common_models import api_error
    from src.services.config_store import ActiveConflictError, get_config_store
    from src.services.config_store.store import activate_axis
    from src.services.curation.strategy_defaults import _advertised_ids_for_axis

    store = get_config_store()
    await store.ensure_fresh(opensearch)
    current_ref = (
        store.current.active_pack if axis == 'prompt_pack' else store.current.active_profile
    )
    expected_active: dict[str, Any] | None = None
    if isinstance(current_ref, tuple):
        expected_active = {'name': current_ref[0], 'revision': current_ref[1]}

    target_name: str | None
    if value is None or value == 'off':
        target_name = None
    else:
        valid_ids = _advertised_ids_for_axis(axis) | (
            {'off'} if axis == 'detection_profile' else set()
        )
        if value not in valid_ids:
            raise api_error(
                422,
                'unknown_pack' if axis == 'prompt_pack' else 'unknown_profile',
                f'{value!r} is not a currently-advertised id for axis {axis!r}',
                axis=axis,
                valid_ids=sorted(valid_ids),
            )
        target_name = value

    # M6: a stored config activates at its OWN current revision, never
    # `None` -- `_axis_ref` on every OTHER process's read coerces a
    # `None` revision to 0, so `(name, None)` written here and
    # `(name, 0)` read there disagree about the stamped revision
    # (any_domain_plan.md §3.7). An env/file id (never saved to the
    # store) genuinely has no revision, so it keeps `None`.
    target_revision: int | None = None
    if target_name is not None:
        stored = (store.current.packs if axis == 'prompt_pack' else store.current.profiles).get(
            target_name
        )
        if stored is not None:
            target_revision = stored.revision

    try:
        await activate_axis(
            store,
            opensearch,
            axis=cast('ConfigAxis', axis),
            name=target_name,
            revision=target_revision,
            expected_active=expected_active,
        )
    except ActiveConflictError as exc:
        raise api_error(
            409,
            'active_conflict',
            f'axis {axis!r} was activated by another writer since this request started',
            current=exc.current,
        ) from exc
