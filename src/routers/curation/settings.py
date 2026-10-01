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

from typing import TYPE_CHECKING, Any, NamedTuple, cast


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
    from src.services.curation.vlm_strategies import active_default_id
    from src.services.labeling.vlm_endpoints import refresh_vlm_state

    store = get_config_store()
    await store.ensure_fresh(opensearch)
    snapshot = store.current
    result: dict[str, str | None] = {}
    # W9: the third config-store axis is the project's VLM endpoint.
    await refresh_vlm_state(opensearch)
    if (vlm_default := active_default_id()) is not None:
        result['vlm'] = vlm_default
    for axis, ref in (
        ('prompt_pack', snapshot.active_pack),
        ('detection_profile', snapshot.active_profile),
    ):
        if ref is None:
            continue
        # Minor 4 (W2 review): 'off' (an explicit deactivation) is a real,
        # distinct state from "never activated" (the store's own
        # docstring, src/services/config_store/store.py's `AxisRef`) --
        # report it, don't fold it into the same absence as `None`.
        result[axis] = 'off' if ref == 'off' else ref[0]
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
    settings_doc_defaults = {
        k: v for k, v in body.defaults.items() if k not in _CONFIG_STORE_AXES and k != 'vlm'
    }

    await _validate_defaults(settings_doc_defaults, opensearch)
    # m1 fix (W3/W4 round-4 review): resolve + gate EVERY config-store axis
    # before writing ANY of them. The old loop activated axis 1, then
    # 422'd resolving axis 2, leaving axis 1's activation committed and
    # the settings-doc write skipped -- a half-applied PUT. Resolve is
    # read-only (raises on any validation/gate failure); apply only runs
    # once every axis has cleared resolve+gate.
    #
    # R5-1 fix (W3/W4 round-5 review): resolving axis 1's TARGET, then
    # immediately gating axis 1 against axis 2's OLD stored value, then
    # resolving+gating axis 2 the same way, let a combined PUT pair a
    # multi-box-stripped pack with a multi-region profile in one request
    # -- each half looked fine against the OTHER axis's PRE-request
    # state, but the actual NEW pairing (both axes applied together) is
    # invalid, exactly what a standalone `/activate` (even with `force`)
    # still correctly 422s. Fix: resolve BOTH axes' target bodies first
    # (no gate calls yet), THEN gate each axis using the OTHER axis's
    # PENDING target from this same request -- not its stored value.
    resolved = {
        axis: await _resolve_config_store_axis(axis, value, opensearch)
        for axis, value in config_store_defaults.items()
    }
    combined = len(resolved) > 1
    # W9: the VLM is the third side of the pack <-> profile <-> VLM triple.
    # It is resolved and gated against the PENDING pack/profile of this same
    # request, and the pack/profile gates below pair with the PENDING VLM
    # (never each against the other sides' stored values: R5-1).
    vlm_plan = None
    if 'vlm' in body.defaults:
        vlm_plan = await _plan_vlm(body.defaults['vlm'], resolved, opensearch)
    for axis, res in resolved.items():
        if res.target_name is None:
            continue
        pending_sibling: Any = None
        has_sibling = False
        if combined:
            other_axis = 'detection_profile' if axis == 'prompt_pack' else 'prompt_pack'
            other = resolved.get(other_axis)
            if other is not None:
                has_sibling = True
                pending_sibling = await _pending_sibling_for_gate(other_axis, other)
        await _run_activation_gate(
            axis,
            res.target_name,
            res.target_revision,
            opensearch,
            **({'pending_sibling': pending_sibling} if has_sibling else {}),
            **({'pending_vlm': vlm_plan.endpoint} if vlm_plan is not None else {}),
        )
    for plan in resolved.values():
        await _apply_config_store_axis(plan, opensearch)
    if vlm_plan is not None:
        await _apply_vlm(vlm_plan, opensearch)

    doc = await update_curation_settings(opensearch, settings_doc_defaults)
    doc['defaults'] = {**doc.get('defaults', {}), **await _config_store_axis_defaults(opensearch)}
    logger.info('curation_settings_updated', axes=sorted(body.defaults))
    return CurationSettingsResponse(**doc)


class _AxisActivationPlan(NamedTuple):
    axis: str
    target_name: str | None
    target_revision: int | None
    expected_active: dict[str, Any] | None


async def _resolve_config_store_axis(
    axis: str, value: str | None, opensearch: Any
) -> _AxisActivationPlan:
    """``PUT /settings {"defaults": {axis: value}}`` for a config-store
    axis: resolve ``value`` (or ``None``/``'off'``) against the live
    registry. Read-only -- writes nothing, and does NOT run the
    activation gate (R5-1 fix, W3/W4 round-5 review: gating happens
    after every axis in the request has been resolved, so a combined
    PUT can gate each axis against the OTHER axis's PENDING target
    instead of its stored one -- see the caller,
    ``update_curation_settings_route``)."""
    from src.routers.curation._config_common_models import api_error
    from src.services.config_store import get_config_store
    from src.services.curation.strategy_defaults import _advertised_ids_for_axis

    store = get_config_store()
    await store.ensure_fresh(opensearch)
    current_ref = (
        store.current.active_pack if axis == 'prompt_pack' else store.current.active_profile
    )
    # R4-2 fix (W3/W4 round-4 review): the same `expected_active` shape
    # bug N3a fixed in clone.py. `index.activate`'s OCC compares against
    # the REAL stored activation doc -- `None` only when no doc has ever
    # been written for this axis. Once any doc exists, an explicit
    # deactivation ('off') stores `{'name': None, 'revision': None}`,
    # which is NOT equal to Python `None`. Folding 'off' into `None` here
    # made every PUT after a deactivation 409 `active_conflict`.
    expected_active: dict[str, Any] | None = None
    if isinstance(current_ref, tuple):
        expected_active = {'name': current_ref[0], 'revision': current_ref[1]}
    elif current_ref == 'off':
        expected_active = {'name': None, 'revision': None}

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

    return _AxisActivationPlan(
        axis=axis,
        target_name=target_name,
        target_revision=target_revision,
        expected_active=expected_active,
    )


async def _pending_sibling_for_gate(other_axis: str, other: _AxisActivationPlan) -> Any:
    """The value to pass as ``run_activation_gate``'s ``pending_sibling``
    for a combined two-axis ``PUT /settings`` (R5-1 fix): the OTHER
    axis's PENDING target from THIS SAME request, resolved to the same
    shape the gate itself validates against for that axis (a
    ``DetectionProfile`` for a pending profile target, a ``PromptPack``
    for a pending pack target)."""
    if other_axis == 'detection_profile':
        if other.target_name is None:
            return None
        from src.services.config_store.profiles import build_record as build_profile_record
        from src.services.detection.profile_registry import region_profile_from_dict

        record = build_profile_record(other.target_name, revision=other.target_revision)
        if record is None:
            return None
        return region_profile_from_dict(
            {**record.body, 'name': other.target_name}, source='validate'
        )
    # other_axis == 'prompt_pack' -- a pack axis always resolves to SOME
    # pack (env/file default when target_name is None, i.e. deactivated).
    from src.services.labeling.vlm_prompts import PromptPack, resolve_prompt_pack

    if other.target_name is None:
        return resolve_prompt_pack()
    from src.services.config_store.packs import build_record as build_pack_record

    pack_record = build_pack_record(other.target_name, revision=other.target_revision)
    if pack_record is None:
        return resolve_prompt_pack()
    return PromptPack.from_dict({**pack_record.body, 'name': other.target_name})


async def _plan_vlm(value: str | None, resolved: dict[str, Any], opensearch: Any) -> Any:
    """Resolve and gate the ``vlm`` default of a ``PUT /settings`` (write
    nothing): 422 ``unknown_vlm`` for an id nobody advertises, then the one
    VLM gate against the request's PENDING pack and profile."""
    from src.routers.curation._config_common_models import api_error
    from src.services.config_store.vlm_activation import UNSET, plan_default_vlm
    from src.services.curation.vlm_strategies import advertised_ids
    from src.services.labeling.vlm_endpoints import refresh_vlm_state

    await refresh_vlm_state(opensearch)
    if value is not None and value not in advertised_ids():
        raise api_error(
            422,
            'unknown_vlm',
            f"{value!r} is not a currently-advertised id for axis 'vlm'",
            axis='vlm',
            requested=value,
            valid_ids=sorted(advertised_ids()),
        )
    pack: Any = UNSET
    profile: Any = UNSET
    if 'prompt_pack' in resolved:
        pack = await _pending_sibling_for_gate('prompt_pack', resolved['prompt_pack'])
    if 'detection_profile' in resolved:
        profile = await _pending_sibling_for_gate(
            'detection_profile', resolved['detection_profile']
        )
    return await plan_default_vlm(opensearch, value, pack=pack, profile=profile)


async def _apply_vlm(plan: Any, opensearch: Any) -> None:
    from src.routers.curation._config_common_models import api_error
    from src.services.config_store import ActiveConflictError
    from src.services.config_store.vlm_activation import apply_default_vlm

    try:
        await apply_default_vlm(opensearch, plan)
    except ActiveConflictError as exc:
        raise api_error(
            409,
            'active_conflict',
            'the VLM was activated by another writer since this request started',
            current=exc.current,
        ) from exc


async def _apply_config_store_axis(plan: _AxisActivationPlan, opensearch: Any) -> None:
    """Write side of a resolved axis activation (see
    :func:`_resolve_config_store_axis`). Only called once every axis in
    the request has cleared resolve+gate."""
    from src.routers.curation._config_common_models import api_error
    from src.services.config_store import ActiveConflictError, get_config_store
    from src.services.config_store.store import activate_axis

    store = get_config_store()
    try:
        await activate_axis(
            store,
            opensearch,
            axis=cast('ConfigAxis', plan.axis),
            name=plan.target_name,
            revision=plan.target_revision,
            expected_active=plan.expected_active,
        )
    except ActiveConflictError as exc:
        raise api_error(
            409,
            'active_conflict',
            f'axis {plan.axis!r} was activated by another writer since this request started',
            current=exc.current,
        ) from exc


_NO_PENDING_OVERRIDE: Any = object()


async def _run_activation_gate(
    axis: str,
    target_name: str,
    target_revision: int | None,
    opensearch: Any,
    *,
    pending_sibling: Any = _NO_PENDING_OVERRIDE,
    pending_vlm: Any = _NO_PENDING_OVERRIDE,
) -> None:
    """N1 fix (W3/W4 round-3 review), now routed through the single
    shared gate (round-4 R4-1): the Cropwright default-pack dropdown
    calls ``PUT /settings``, not ``POST .../activate`` -- this bridge must
    run the exact same never-bypassable ``for_activation`` validation the
    dedicated activate routes (and rollback) run, or a multi-box-stripped
    revision that ``/activate`` correctly 422s (even with ``force: true``)
    goes live silently through this route instead. This bridge passes no
    ``force``, so it never bypasses -- any blocking error 422s.

    ``pending_sibling`` (R5-1 fix): forwarded to
    :func:`~src.services.config_store.activation_gate.run_activation_gate`
    only when explicitly supplied (a combined two-axis PUT) -- see
    :func:`_pending_sibling_for_gate`. Left as the default sentinel for a
    single-axis PUT, so the gate falls back to its own stored-value
    lookup, unchanged.
    """
    from src.services.config_store.activation_gate import run_activation_gate

    kwargs: dict[str, Any] = {}
    if pending_sibling is not _NO_PENDING_OVERRIDE:
        kwargs['pending_sibling'] = pending_sibling
    if pending_vlm is not _NO_PENDING_OVERRIDE:
        kwargs['pending_vlm'] = pending_vlm
    await run_activation_gate(
        cast('ConfigAxis', axis), target_name, target_revision, client=opensearch, **kwargs
    )
