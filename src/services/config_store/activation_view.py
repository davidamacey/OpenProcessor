"""Shared ``ActiveConfigResponse`` builder for the pack/profile activation
routes -- one axis-agnostic builder
so ``GET /prompt_packs/active`` and ``GET /region_profiles/active`` (and
each axis's activate/rollback response) serve byte-identical shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from src.schemas.curation.config_common import ActiveConfigResponse, ActiveRef, AppliedRuntime
from src.services.config_store import documents, get_config_store
from src.services.config_store.activation_apply import AXIS_STORAGE


if TYPE_CHECKING:
    from src.db.repos.config import RuntimeReport
    from src.services.config_store.snapshot import AxisRef, ConfigSnapshot


Axis = Literal['prompt_pack', 'detection_profile', 'open_vocab', 'ingest_profile']


def _env_default_ref(axis: Axis) -> ActiveRef:
    """The env/file default a never-activated axis really runs, named by the
    same resolver the worker applies (so ``active`` and the worker's
    ``applied[]`` row cannot disagree). Nameless only when there is no env
    default (no region profile configured)."""
    if axis in ('open_vocab', 'ingest_profile'):
        return ActiveRef()
    if axis == 'prompt_pack':
        from src.services.labeling.vlm_prompt_resolution import active_prompt_pack

        return ActiveRef(name=active_prompt_pack().name)
    from src.services.detection.profile_registry import get_active_region_profile

    profile = get_active_region_profile()
    return ActiveRef(name=profile.name) if profile is not None else ActiveRef()


def applied_runtime(report: RuntimeReport, snapshot: ConfigSnapshot) -> AppliedRuntime:
    """One worker's "what did I apply" row from its epoch report.

    A report carries the epoch the worker applied, not the refs: a worker at the current epoch
    runs the current activations (an axis never activated runs its env default), and one behind
    it is ``lagging`` with its refs unknown.
    """
    current = report.config_epoch >= snapshot.config_revision

    def ref(axis_ref: AxisRef, default: ActiveRef) -> ActiveRef:
        if not current:
            return ActiveRef()
        if isinstance(axis_ref, tuple):
            return ActiveRef(name=axis_ref[0], revision=axis_ref[1])
        return default if axis_ref is None else ActiveRef()

    return AppliedRuntime(
        process=report.process,
        host=report.host,
        applied_config_revision=report.config_epoch,
        profile=ref(snapshot.active_profile, _env_default_ref('detection_profile')),
        pack=ref(snapshot.active_pack, _env_default_ref('prompt_pack')),
        vlm=ref(snapshot.active_vlm, ActiveRef()),
        applied_at=report.applied_at.isoformat(),
        lagging=not current,
    )


async def build_active_config_response(*, axis: Axis) -> ActiveConfigResponse:
    store = get_config_store()
    await store.ensure_fresh()
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
        # A stored revision outlives the document's presence in the current listing only
        # while it is active, so any activation of a stored name reports `stored`.
        source = 'stored' if (ref[1] is not None or ref[0] in stored_names) else 'env'

    detail = await documents.get_activation_detail(axis)
    activated_at = detail.activated_at if detail is not None and source != 'env' else None
    previous = (
        ActiveRef(name=detail.previous[0], revision=detail.previous[1])
        if detail is not None and detail.previous is not None
        else None
    )

    reports: list[RuntimeReport] = []
    # No worker applies the open-vocabulary or ingest-profile axes: a run reads
    # the activation when it starts, so there is nothing to lag behind.
    if axis not in ('open_vocab', 'ingest_profile'):
        try:
            reports = await documents.live_runtime_reports(process='detection_worker')
        except Exception:  # pragma: no cover - defensive; applied[] degrades to empty
            reports = []

    return ActiveConfigResponse(
        axis=axis,
        active=active,
        source=source,
        activated_at=activated_at,
        previous=previous,
        config_revision=snapshot.config_revision,
        stale=snapshot.stale,
        applied=[applied_runtime(r, snapshot) for r in reports],
    )


__all__ = ['applied_runtime', 'build_active_config_response']
