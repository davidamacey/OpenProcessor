"""Shared activate/rollback write-and-apply logic (N4 fix, W3/W4 round-3
review), split out of ``store.py`` to stay under the 700-LOC ratchet.

Both :func:`~src.services.config_store.store.activate_axis` and
:func:`rollback_axis` (used by ``packs.rollback_pack`` /
``profiles.rollback_profile``) need to: resolve the target's pinned body
BEFORE writing the new activation doc, write through
``index.activate``, then apply the result to the in-memory
:class:`~src.services.config_store.store.ConfigStore`. Resolving the body
first means a transient error there aborts cleanly with nothing written,
instead of surfacing a 500 to the caller AFTER the activation doc had
already durably committed (the old write-then-resolve order).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from src.services.config_store.index import ConfigAxis, ConfigKind
    from src.services.config_store.store import AxisRef, ConfigStore


async def activate_and_apply(
    store: ConfigStore,
    client: Any,
    *,
    axis: ConfigAxis,
    name: str | None,
    revision: int | None,
    expected_active: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the pinned body, write the activation, apply it to
    ``store``. B1 fix: pin the exact body activated now, not a later
    by-name re-derivation a subsequent PUT could change."""
    from src.services.config_store.index import activate as _activate
    from src.services.config_store.store import _resolve_active_body

    ref: AxisRef = (name, revision) if name else 'off'
    kind: ConfigKind = 'prompt_pack' if axis == 'prompt_pack' else 'region_profile'
    current_map = store.current.packs if axis == 'prompt_pack' else store.current.profiles
    body_ref = await _resolve_active_body(
        client, store.index, kind=kind, ref=ref, current=current_map
    )
    result = await _activate(
        client,
        store.index,
        axis=axis,
        name=name,
        revision=revision,
        expected_active=expected_active,
    )
    patch: dict[str, Any] = {'config_revision': result['config_revision']}
    if axis == 'prompt_pack':
        patch['active_pack'] = ref
        patch['active_pack_body'] = body_ref
    else:
        patch['active_profile'] = ref
        patch['active_profile_body'] = body_ref
    store.apply_local(**patch)
    return result


async def rollback_axis(
    store: ConfigStore,
    client: Any,
    *,
    axis: ConfigAxis,
    expected_active: dict[str, Any] | None,
) -> dict[str, Any]:
    """Re-activate ``axis``'s ``previous`` ref, replacing
    ``packs.rollback_pack`` / ``profiles.rollback_profile``'s own
    write-then-resolve call into ``index.rollback``.
    ``LookupError('no_previous')`` with nothing written when there is no
    ``previous``."""
    from src.services.config_store.index import get_activation

    current_doc = await get_activation(client, store.index, axis)
    previous = (current_doc or {}).get('previous')
    if not previous:
        msg = 'no_previous'
        raise LookupError(msg)

    # R4-1 fix (W3/W4 round-4 review): rollback re-activates `previous`,
    # which must run the exact same never-bypassable `for_activation` gate
    # `activate` and the settings bridge run. A revision that was validly
    # active once is not necessarily valid NOW -- the gate is cross-axis
    # (e.g. the multi-box-key check depends on the *other* axis's current
    # state), so "it passed before" doesn't make rolling back to it safe.
    # No `force` here: rollback has no bypass flag, mirroring the
    # settings bridge.
    previous_name = previous.get('name')
    if previous_name is not None:
        from src.services.config_store.activation_gate import run_activation_gate

        await run_activation_gate(
            axis, previous_name, previous.get('revision'), force=False, client=client
        )

    return await activate_and_apply(
        store,
        client,
        axis=axis,
        name=previous.get('name'),
        revision=previous.get('revision'),
        expected_active=expected_active,
    )


__all__ = ['activate_and_apply', 'rollback_axis']
