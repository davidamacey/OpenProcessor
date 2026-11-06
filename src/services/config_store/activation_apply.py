"""Shared activate/rollback write-and-apply logic, split out of ``store.py``.

Both :func:`~src.services.config_store.store.activate_axis` and :func:`rollback_axis` (used by
``packs.rollback_pack`` / ``profiles.rollback_profile``) need to resolve the target's pinned
body BEFORE writing the activation, write it through
:func:`~src.services.config_store.documents.activate`, then apply the result to the in-memory
:class:`~src.services.config_store.store.ConfigStore`. Resolving the body first means a transient
error there aborts cleanly with nothing written, instead of surfacing a 500 to the caller AFTER
the activation had already durably committed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast


if TYPE_CHECKING:
    from src.services.config_store.documents import Activated, ConfigAxis, ConfigKind, Registered
    from src.services.config_store.snapshot import StoredConfig
    from src.services.config_store.store import ConfigStore


#: Per axis: the stored kind, the snapshot field holding the stored configs,
#: and the snapshot field holding the activation (``<field>_body`` is the
#: pinned activated body). The one table every activate/rollback path reads.
AXIS_STORAGE: dict[str, tuple[ConfigKind, str, str]] = {
    'prompt_pack': ('prompt_pack', 'packs', 'active_pack'),
    'detection_profile': ('region_profile', 'profiles', 'active_profile'),
    'open_vocab': ('open_vocab_set', 'open_vocab_sets', 'active_open_vocab'),
    'ingest_profile': ('ingest_profile', 'ingest_profiles', 'active_ingest_profile'),
}

#: The ``ConfigStore.apply_local`` key that replaces one stored config of each kind.
LOCAL_PATCH_KEY: dict[str, str] = {
    'prompt_pack': 'pack',
    'region_profile': 'profile',
    'open_vocab_set': 'open_vocab_set',
    'ingest_profile': 'ingest_profile',
}


async def _pinned_body(
    store: ConfigStore, axis: ConfigAxis, name: str, revision: int
) -> StoredConfig | None:
    """The exact body of ``name@revision``: the loaded current config when it is that revision,
    else the immutable revision row, so a later save never changes what this resolves to."""
    from src.services.config_store.documents import get_revision

    kind, stored_field, _ = AXIS_STORAGE[axis]
    local = getattr(store.current, stored_field).get(name)
    if local is not None and local.revision == revision:
        return cast('StoredConfig', local)
    return await get_revision(kind=kind, name=name, revision=revision)


def registered_for(axis: ConfigAxis, name: str) -> Registered | None:
    """The body of a built-in or file pack, or an env or registered profile, named ``name``;
    ``None`` for a name that is not one of those (a stored or unknown name)."""
    from src.services.config_store.documents import Registered

    if axis == 'prompt_pack':
        from src.services.config_store.packs import build_record as build_pack

        pack = build_pack(name)
        if pack is not None and pack.source in ('builtin', 'file'):
            return Registered(pack.body, pack.description)
    elif axis == 'detection_profile':
        from src.services.config_store.profiles import build_record as build_profile

        profile = build_profile(name)
        if profile is not None and profile.source in ('env', 'registered'):
            return Registered(profile.body, profile.description)
    return None


async def activate_and_apply(
    store: ConfigStore,
    *,
    axis: ConfigAxis,
    name: str | None,
    revision: int | None,
    expected_active: dict[str, Any] | None,
    registered: Registered | None = None,
    reason: Literal['activate', 'rollback', 'clone'] = 'activate',
) -> Activated:
    """Resolve the pinned body, write the activation, apply it to ``store``.

    Pin the exact body activated now, not a later by-name re-derivation a subsequent save could
    change. A ``revision`` of ``None`` with a ``name`` is a built-in, file-registered or
    environment config: ``registered`` carries its body, which the write stores as a system
    revision, and the store then lists it as a stored config.
    """
    from src.services.config_store.documents import activate as _activate

    _, _, active_field = AXIS_STORAGE[axis]
    body: StoredConfig | None = None
    if name is not None and revision is not None:
        body = await _pinned_body(store, axis, name, revision)
    elif name is not None and registered is None:
        registered = registered_for(axis, name)
    result = await _activate(
        axis=axis,
        name=name,
        revision=revision,
        expected_active=expected_active,
        registered=registered,
        reason=reason,
    )
    store.apply_local(
        **{'config_revision': result.epoch, active_field: result.ref, f'{active_field}_body': body}
    )
    if axis == 'detection_profile':
        # The profile's region class must exist in the project registry the
        # moment it is active; a no-op when it already does or none is named.
        from src.services.curation.region_class import ensure_region_class

        await ensure_region_class()
    return result


async def rollback_axis(
    store: ConfigStore, *, axis: ConfigAxis, expected_active: dict[str, Any] | None
) -> Activated:
    """Re-activate what ``axis`` moved away from, through the same gate as an activation.

    ``LookupError('no_previous')`` or ``LookupError('previous_deleted')`` with nothing written.
    """
    from src.services.config_store.documents import previous_target

    # Refresh the in-memory snapshot BEFORE gating: the gate's cross-axis inputs
    # (`get_active_region_profile()` / `active_prompt_pack()`) read `store.current`, which the
    # activate routes and the settings bridge already refresh first.
    await store.ensure_fresh()
    target = await previous_target(axis)

    # A revision that was validly active once is not necessarily valid NOW: the gate is
    # cross-axis (the multi-box-key check depends on the *other* axis's current state), so "it
    # passed before" doesn't make rolling back to it safe. No `force` here: rollback has no
    # bypass flag, mirroring the settings bridge.
    if target.name is not None:
        from src.services.config_store.activation_gate import run_activation_gate

        await run_activation_gate(axis, target.name, target.revision, force=False)

    return await activate_and_apply(
        store,
        axis=axis,
        name=target.name,
        revision=target.revision,
        expected_active=expected_active,
        reason='rollback',
    )


__all__ = ['AXIS_STORAGE', 'LOCAL_PATCH_KEY', 'activate_and_apply', 'rollback_axis']
