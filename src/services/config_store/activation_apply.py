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


#: Per axis: the stored kind, the snapshot field holding the stored configs,
#: and the snapshot field holding the activation (``<field>_body`` is the
#: pinned activated body). The one table every activate/rollback path reads.
AXIS_STORAGE: dict[str, tuple[ConfigKind, str, str]] = {
    'prompt_pack': ('prompt_pack', 'packs', 'active_pack'),
    'detection_profile': ('region_profile', 'profiles', 'active_profile'),
    'open_vocab': ('open_vocab_set', 'open_vocab_sets', 'active_open_vocab'),
}


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
    kind, stored_field, active_field = AXIS_STORAGE[axis]
    current_map = getattr(store.current, stored_field)
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
    store.apply_local(
        config_revision=result['config_revision'],
        **{active_field: ref, f'{active_field}_body': body_ref},
    )
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

    # m-a fix (W3/W4 round-5 review): refresh the in-memory snapshot
    # BEFORE gating -- the gate's cross-axis inputs
    # (`get_active_region_profile()` / `active_prompt_pack()`) and the
    # deleted-target check just below both read `store.current`, which
    # the activate routes and the settings bridge already refresh first
    # but this path never did, so it could gate against a snapshot
    # another process's write had already superseded.
    await store.ensure_fresh(client)

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
    previous_revision = previous.get('revision')
    if previous_name is not None:
        # m-b fix (W3/W4 round-5 review): the gate validates the
        # immutable `<kind>:<name>@<rev>` revision-copy doc, which
        # survives a `DELETE` of the CURRENT doc -- so without this
        # check, rolling back to a target deleted while it was
        # `previous` resurrects it as active (and `GET /active` used to
        # mislabel its `source` too, fixed separately in
        # activation_view.py). A non-`None` `previous_revision` proves
        # this target was a stored config at activation time (env/file
        # ids never carry a revision, M6); if it is no longer in the
        # store's current names, it was deleted since, and rollback must
        # refuse, not resurrect it.
        stored_names = getattr(store.current, AXIS_STORAGE[axis][1])
        if previous_revision is not None and previous_name not in stored_names:
            msg = 'previous_deleted'
            raise LookupError(msg)

        from src.services.config_store.activation_gate import run_activation_gate

        await run_activation_gate(
            axis, previous_name, previous_revision, force=False, client=client
        )

    return await activate_and_apply(
        store,
        client,
        axis=axis,
        name=previous.get('name'),
        revision=previous.get('revision'),
        expected_active=expected_active,
    )


__all__ = ['AXIS_STORAGE', 'activate_and_apply', 'rollback_axis']
