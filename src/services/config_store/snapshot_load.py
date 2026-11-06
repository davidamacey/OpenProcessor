"""Build a :class:`~src.services.config_store.snapshot.ConfigSnapshot` from the database.

A project snapshot holds the project's stored configs at their current revision, each axis's
activation and the activated revision's body pinned (a later save never changes what is live).
The deployment snapshot holds the VLM endpoint registry, each endpoint revision's last probe and
the desired local model.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.db.repos import config as repo, settings as settings_repo
from src.services.config_store.documents import VLM_ACK_KEY, ref_of, stored_from_doc, tx
from src.services.config_store.registered import is_registered
from src.services.config_store.snapshot import ConfigSnapshot, StoredConfig
from src.services.labeling.vlm_endpoints import ENV_ENDPOINT_NAME, probe_key


if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

LOCAL_VLM_DESIRED_KEY = 'local_vlm_desired'

# axis -> (kind, snapshot field of the stored configs, snapshot field of the activation)
_PROJECT_AXES = {
    'prompt_pack': ('prompt_pack', 'packs', 'active_pack'),
    'detection_profile': ('region_profile', 'profiles', 'active_profile'),
    'open_vocab': ('open_vocab_set', 'open_vocab_sets', 'active_open_vocab'),
    'ingest_profile': ('ingest_profile', 'ingest_profiles', 'active_ingest_profile'),
}


async def _pinned(
    session: AsyncSession, active: repo.ActiveRef | None, current: dict[str, StoredConfig]
) -> StoredConfig | None:
    """The activated revision's body: the loaded current document when it is that revision,
    otherwise the immutable revision row."""
    if (
        active is None
        or active.state != 'active'
        or active.name is None
        or active.config_id is None
    ):
        return None
    local = current.get(active.name)
    if local is not None and local.revision == active.revision:
        return local
    doc = await repo.get_config_by_id(session, active.config_id, active.revision)
    return None if doc is None else stored_from_doc(doc)


async def load_snapshot(project_id: uuid.UUID | None) -> ConfigSnapshot:
    """The snapshot of one project, or the deployment-wide one (``project_id`` ``None``)."""
    async with tx('load config snapshot') as session:
        if project_id is None:
            return await _load_deployment(session)
        return await _load_project(session, project_id)


async def _load_project(session: AsyncSession, project_id: uuid.UUID) -> ConfigSnapshot:
    epoch = await repo.get_epoch(session, project_id)
    stored: dict[str, dict[str, StoredConfig]] = {k: {} for k, _, _ in _PROJECT_AXES.values()}
    axis_of_kind = {kind: axis for axis, (kind, _, _) in _PROJECT_AXES.items()}
    for doc in await repo.list_configs(session, project_id=project_id):
        # a system revision of a registered config is not a stored config
        if doc.kind in stored and not is_registered(axis_of_kind[doc.kind], doc.name):
            stored[doc.kind][doc.name] = stored_from_doc(doc)
    activations = await repo.get_activations(session, project_id)
    fields: dict[str, Any] = {}
    for axis, (kind, stored_field, active_field) in _PROJECT_AXES.items():
        active = activations.get(axis)
        fields[stored_field] = stored[kind]
        fields[active_field] = ref_of(active)
        registered = active is not None and is_registered(axis, active.name)
        fields[f'{active_field}_body'] = (
            None if registered else await _pinned(session, active, stored[kind])
        )
    vlm = activations.get('vlm')
    acked = await settings_repo.get_state(session, project_id, VLM_ACK_KEY)
    acked_refs: dict[str, str] = dict(
        ((acked.value if acked else {}) or {}).get('acked_refs') or {}
    )
    vlm_ref = ref_of(vlm)
    vlm_body: StoredConfig | None = None
    ack_at: str | None = None
    if isinstance(vlm_ref, tuple):
        ack_at = acked_refs.get(f'{vlm_ref[0]}@{vlm_ref[1]}')
        if vlm_ref[0] != ENV_ENDPOINT_NAME:
            vlm_body = await _pinned(session, vlm, {})
    return ConfigSnapshot(
        config_revision=epoch,
        loaded_at=time.monotonic(),
        stale=False,
        active_vlm=vlm_ref,
        active_vlm_body=vlm_body,
        active_vlm_ack_at=ack_at,
        acked_refs=frozenset(acked_refs),
        **fields,
    )


async def _load_deployment(session: AsyncSession) -> ConfigSnapshot:
    epoch = await repo.get_epoch(session, None)
    docs = await repo.list_configs(session, project_id=None, kind='vlm_endpoint')
    by_id = {d.id: d for d in docs}
    endpoints = {d.name: stored_from_doc(d) for d in docs if d.name != ENV_ENDPOINT_NAME}
    probes: dict[str, dict[str, Any]] = {}
    for probe in await repo.list_probes(session):
        doc = by_id.get(probe.config_id)
        if doc is None:
            continue
        if doc.name == ENV_ENDPOINT_NAME:
            # The built-in has no revisions: its newest stored revision carries its probe.
            if probe.revision == doc.current_revision:
                probes[probe_key(doc.name, None)] = dict(probe.result)
        else:
            probes[probe_key(doc.name, probe.revision)] = dict(probe.result)
    desired_row = await repo.get_deployment_setting(session, LOCAL_VLM_DESIRED_KEY)
    desired = dict(desired_row.value) if desired_row is not None and desired_row.value else None
    return ConfigSnapshot(
        config_revision=epoch,
        loaded_at=time.monotonic(),
        stale=False,
        vlm_endpoints=endpoints,
        vlm_probes=probes,
        local_vlm_desired=desired,
    )
