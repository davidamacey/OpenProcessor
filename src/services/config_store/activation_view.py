"""Shared ``ActiveConfigResponse`` builder for the pack/profile activation
routes (W3/W4, any_domain_plan.md §7.2/§7.3) -- one axis-agnostic builder
so ``GET /prompt_packs/active`` and ``GET /region_profiles/active`` (and
each axis's activate/rollback response) serve byte-identical shapes.
"""

from __future__ import annotations

from typing import Any, Literal

from src.routers.curation._config_common_models import (
    ActiveConfigResponse,
    ActiveRef,
    AppliedRuntime,
)
from src.services.config_store import get_config_store
from src.services.config_store.activation_apply import AXIS_STORAGE
from src.services.config_store.index import get_activation, get_runtime_docs


Axis = Literal['prompt_pack', 'detection_profile', 'open_vocab']


def _env_default_ref(axis: Axis) -> ActiveRef:
    """The env/file default a never-activated axis really runs, named by the
    same resolver the worker applies (so ``active`` and the worker's
    ``applied[]`` row cannot disagree). Nameless only when there is no env
    default (no region profile configured)."""
    if axis == 'open_vocab':
        return ActiveRef()
    if axis == 'prompt_pack':
        from src.services.labeling.vlm_prompt_resolution import active_prompt_pack

        return ActiveRef(name=active_prompt_pack().name)
    from src.services.detection.profile_registry import get_active_region_profile

    profile = get_active_region_profile()
    return ActiveRef(name=profile.name) if profile is not None else ActiveRef()


async def build_active_config_response(client: Any, *, axis: Axis) -> ActiveConfigResponse:
    store = get_config_store()
    await store.ensure_fresh(client)
    snapshot = store.current
    _, stored_field, active_field = AXIS_STORAGE[axis]
    ref = getattr(snapshot, active_field)
    stored_names = getattr(snapshot, stored_field)

    if ref is None:
        active = _env_default_ref(axis)
        source: Literal['stored', 'env', 'off'] = 'env'
    elif ref == 'off':
        active = ActiveRef()
        source = 'off'
    else:
        active = ActiveRef(name=ref[0], revision=ref[1])
        # Minor m-b fix (W3/W4 round-5 review): `ref[0] in stored_names`
        # only sees names the store CURRENTLY lists -- a pack/profile
        # deleted while it is (or becomes, via rollback) the activation's
        # target drops out of `stored_names` even though the activated
        # revision is real (the gate validated it via the immutable
        # `<kind>:<name>@<rev>` copy, not this list). Reporting `'env'`
        # for that case is misleading: env/file ids never carry a
        # revision (M6 above) at all, so a non-`None` `ref[1]` already
        # proves this was a stored config, deleted or not.
        source = 'stored' if (ref[1] is not None or ref[0] in stored_names) else 'env'

    activation_doc = await get_activation(client, store.index, axis)
    activated_at = (activation_doc or {}).get('activated_at') if source != 'env' else None
    previous_doc = (activation_doc or {}).get('previous')
    previous = (
        ActiveRef(name=previous_doc.get('name'), revision=previous_doc.get('revision'))
        if previous_doc
        else None
    )

    docs: list[dict[str, Any]] = []
    # No worker applies the open-vocabulary axis: a run reads the activation
    # when it starts, so there is nothing to lag behind.
    if axis != 'open_vocab':
        try:
            docs = await get_runtime_docs(client, store.index, process='detection_worker')
        except Exception:  # pragma: no cover - defensive; applied[] degrades to empty
            docs = []

    return ActiveConfigResponse(
        axis=axis,
        active=active,
        source=source,
        activated_at=activated_at,
        previous=previous,
        config_revision=snapshot.config_revision,
        stale=snapshot.stale,
        applied=[
            AppliedRuntime.from_runtime_doc(d, config_revision=snapshot.config_revision)
            for d in docs
        ],
    )


__all__ = ['build_active_config_response']
