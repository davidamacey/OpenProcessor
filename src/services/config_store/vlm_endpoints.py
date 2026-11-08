"""The VLM endpoint registry's storage operations.

Endpoints are global (deployment-wide): each is a ``vlm_endpoint`` document with immutable
revisions, its last probe per revision is a ``vlm_endpoint_probes`` row, and the desired local
model is the ``local_vlm_desired`` deployment setting. Revision numbers, conflicts and the global
config epoch behave exactly like packs and profiles. Which endpoint a *project* runs is that
project's own activation (:mod:`src.services.config_store.vlm_activation`).

The ``env`` built-in (``OP_VLM_URL`` / ``OP_VLM_MODEL``) has no stored endpoint until it is
activated or probed; then its body is stored as a system revision of the reserved ``env``
document so an activation or a probe has a document to point at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.db.repos import config as repo
from src.services.config_store import documents
from src.services.config_store.global_store import get_global_config_store
from src.services.config_store.snapshot_load import LOCAL_VLM_DESIRED_KEY
from src.services.labeling.vlm_endpoints import (
    ENV_ENDPOINT_NAME,
    VlmEndpoint,
    probe_fingerprint,
    stored_endpoint,
)
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    import uuid

    from src.services.labeling.vlm_endpoint_body import VlmEndpointBody, VlmProbeRecord

ENV_DESCRIPTION = 'From OP_VLM_URL / OP_VLM_MODEL'
_SETTING_ATTEMPTS = 3


async def _reload(axis: str = 'registry') -> Any:
    """Re-read the registry after a write: the exact state, not this process's guess of it
    (another uvicorn worker may have written too), and tell every open UI (``vlm.changed`` on
    the global stream)."""
    from src.services.events.publish import apublish_global_event

    store = get_global_config_store()
    await store.refresh()
    await apublish_global_event('vlm.changed', axis=axis)
    return store.current


async def save_endpoint(
    *,
    name: str,
    body: VlmEndpointBody,
    expected_revision: int | None,
    description: str = '',
    cloned_from: documents.CloneSource | None = None,
) -> VlmEndpoint:
    """Create (``expected_revision=None``) or save a new revision. Raises
    :class:`~src.db.repos.config.RevisionConflictError` on a stale ``expected_revision``."""
    await documents.save_config(
        kind='vlm_endpoint',
        name=name,
        body=body.normalized().model_dump(),
        expected_revision=expected_revision,
        description=description,
        cloned_from=cloned_from,
    )
    snapshot = await _reload()
    return stored_endpoint(snapshot.vlm_endpoints[name], snapshot.vlm_probes)


async def list_revisions(name: str) -> list[repo.RevisionInfo]:
    """Every saved revision of ``name``, newest first."""
    return await documents.list_revisions(kind='vlm_endpoint', name=name) or []


async def delete_endpoint(*, name: str, expected_revision: int) -> None:
    """Delete the endpoint with its revisions and probes. A project's activation still pointing
    at it refuses the delete (:class:`~src.db.repos.config.ConfigInUseError`)."""
    await documents.delete_config(
        kind='vlm_endpoint', name=name, expected_revision=expected_revision
    )
    await _reload()


async def record_probe(
    *, name: str, revision: int | None, body: VlmEndpointBody, record: VlmProbeRecord
) -> None:
    """Persist ``record`` as the last probe of ``name@revision`` (``revision`` is ``None`` only
    for the ``env`` built-in, whose current body is stored first); another revision's probe is
    never touched. Never bumps the endpoint revision; DOES bump the global epoch, so every
    process (and every worker's quiesce-and-swap) sees the new ``json_mode`` /
    ``max_model_len`` / model root. ``body`` fingerprints what was probed."""
    payload = {'fingerprint': probe_fingerprint(body), 'record': record.model_dump()}
    async with documents.tx('record vlm probe') as session:
        if revision is None:
            doc = await repo.ensure_config(
                session,
                project_id=None,
                kind='vlm_endpoint',
                name=ENV_ENDPOINT_NAME,
                body=body.model_dump(),
                description=ENV_DESCRIPTION,
            )
        else:
            found = await repo.get_config(
                session, project_id=None, kind='vlm_endpoint', name=name, revision=revision
            )
            if found is None:
                raise KeyError(f'{name}@{revision}')
            doc = found
        await repo.record_probe(
            session, config_id=doc.id, revision=doc.revision, ok=record.ok, result=payload
        )
    await _reload()


async def endpoint_config_id(name: str) -> uuid.UUID | None:
    """The stored document id of endpoint ``name`` (every revision shares it), or ``None``.

    Usage rows (``vlm_usage_hourly``) point at the document; the endpoint registry is
    deployment-wide, so the read runs in the deployment scope."""
    async with documents.tx('look up a vlm endpoint document for usage') as session:
        doc = await repo.get_config(session, project_id=None, kind='vlm_endpoint', name=name)
    return None if doc is None else doc.id


async def _write_desired(value: dict[str, Any]) -> None:
    for _ in range(_SETTING_ATTEMPTS):
        async with documents.tx('write desired local vlm') as session:
            current = await repo.get_deployment_setting(session, LOCAL_VLM_DESIRED_KEY)
            try:
                await repo.put_deployment_setting(
                    session,
                    key=LOCAL_VLM_DESIRED_KEY,
                    value=value,
                    expected_revision=current.revision if current else None,
                )
            except repo.RevisionConflictError:
                continue
            return
    raise repo.RevisionConflictError(None)


async def set_local_desired(*, catalog_id: str) -> dict[str, Any]:
    doc = {'catalog_id': catalog_id, 'requested_at': now_iso()}
    await _write_desired(doc)
    await _reload('local_vlm')
    return doc


async def clear_local_desired() -> None:
    """An empty value is "no desired local model" (the setting row is never deleted)."""
    await _write_desired({})
    await _reload('local_vlm')


__all__ = [
    'ENV_DESCRIPTION',
    'clear_local_desired',
    'delete_endpoint',
    'endpoint_config_id',
    'list_revisions',
    'record_probe',
    'save_endpoint',
    'set_local_desired',
]
