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
from src.services.config_store.index import get_activation, get_runtime_docs


Axis = Literal['prompt_pack', 'detection_profile']


async def build_active_config_response(client: Any, *, axis: Axis) -> ActiveConfigResponse:
    store = get_config_store()
    await store.ensure_fresh(client)
    snapshot = store.current
    ref = snapshot.active_pack if axis == 'prompt_pack' else snapshot.active_profile
    stored_names = snapshot.packs if axis == 'prompt_pack' else snapshot.profiles

    if ref is None:
        active = ActiveRef()
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

    applied: list[AppliedRuntime] = []
    try:
        docs = await get_runtime_docs(client, store.index, process='detection_worker')
    except Exception:  # pragma: no cover - defensive; applied[] degrades to empty
        docs = []
    for doc in docs:
        profile_ref = doc.get('profile') or {}
        pack_ref = doc.get('pack') or {}
        applied_rev = int(doc.get('applied_config_revision') or 0)
        applied.append(
            AppliedRuntime(
                process=doc.get('process', 'detection_worker'),
                host=doc.get('host', ''),
                applied_config_revision=applied_rev,
                profile=ActiveRef(**profile_ref) if profile_ref else ActiveRef(),
                pack=ActiveRef(**pack_ref) if pack_ref else ActiveRef(),
                applied_at=doc.get('applied_at'),
                lagging=applied_rev < snapshot.config_revision,
            )
        )

    return ActiveConfigResponse(
        axis=axis,
        active=active,
        source=source,
        activated_at=activated_at,
        previous=previous,
        config_revision=snapshot.config_revision,
        stale=snapshot.stale,
        applied=applied,
    )


__all__ = ['build_active_config_response']
