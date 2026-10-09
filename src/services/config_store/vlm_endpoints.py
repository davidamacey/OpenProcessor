"""The VLM endpoint registry's storage operations (W9.6).

Endpoints are global (deployment-wide): docs live in ``op_global_configs``
(``vlm:<name>`` current, ``vlm:<name>@<rev>`` immutable copies,
``vlm_probe:<name>@<rev>``, ``local_vlm:desired``) and are written through W2's
OCC primitives, so revision numbers, conflicts and the config revision
counter behave exactly like packs and profiles. Which endpoint a *project*
runs is that project's own activation
(:mod:`src.services.config_store.vlm_activation`).

These functions run on the global router, i.e. with NO project bound: the
project guard refuses a bound write to this index.
"""

from __future__ import annotations

import contextlib
import datetime
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import NotFoundError

from src.services.config_store.global_store import get_global_config_store
from src.services.config_store.index import (
    WRITE_REFRESH,
    RevisionConflictError,
    bump_config_revision,
    delete_config as _delete_config,
    save_config as _save_config,
)
from src.services.config_store.vlm_snapshot import LOCAL_DESIRED_DOC_ID, probe_doc_id
from src.services.labeling.vlm_endpoints import (
    VlmEndpoint,
    probe_fingerprint,
    probe_key,
    stored_endpoint,
)


if TYPE_CHECKING:
    from src.services.labeling.vlm_endpoint_body import VlmEndpointBody, VlmProbeRecord


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


async def _reload(client: Any, axis: str = 'registry') -> Any:
    """Re-read the registry after a write: the exact state, not this
    process's guess of it (another uvicorn worker may have written too), and
    tell every open UI (``vlm.changed`` on the global stream)."""
    from src.services.curation.event_hub import publish_global_event

    store = get_global_config_store()
    await store.refresh(client)
    publish_global_event('vlm.changed', axis=axis)
    return store.current


async def save_endpoint(
    client: Any,
    *,
    name: str,
    body: VlmEndpointBody,
    expected_revision: int | None,
    description: str = '',
    cloned_from: str | None = None,
) -> VlmEndpoint:
    """Create (``expected_revision=None``) or save a new revision. Raises
    :class:`RevisionConflictError` on a stale ``expected_revision``."""
    store = get_global_config_store()
    await _save_config(
        client,
        store.index,
        kind='vlm_endpoint',
        name=name,
        body=body.normalized().model_dump(),
        expected_revision=expected_revision,
        description=description,
        cloned_from=cloned_from,
    )
    snapshot = await _reload(client)
    return stored_endpoint(snapshot.vlm_endpoints[name], snapshot.vlm_probes)


async def list_revisions(client: Any, name: str) -> list[dict[str, Any]]:
    """Every saved revision of ``name``, newest first (the immutable copies,
    which survive a delete of the current doc)."""
    store = get_global_config_store()
    resp = await client.search(
        index=store.index,
        body={
            'size': 1000,
            'query': {
                'bool': {
                    'filter': [
                        {'term': {'doc_type': 'revision'}},
                        {'term': {'kind': 'vlm_endpoint'}},
                        {'term': {'name': name}},
                    ]
                }
            },
            'sort': [{'revision': {'order': 'desc'}}],
        },
    )
    return [hit['_source'] for hit in resp['hits']['hits']]


async def delete_endpoint(client: Any, *, name: str, expected_revision: int) -> None:
    """Delete the current doc (revision copies stay, so revision numbers
    are never reused and a project's pinned activation still resolves)."""
    store = get_global_config_store()
    await _delete_config(
        client, store.index, kind='vlm_endpoint', name=name, expected_revision=expected_revision
    )
    for doc in await list_revisions(client, name):
        with contextlib.suppress(NotFoundError):
            await client.delete(
                index=store.index,
                id=probe_doc_id(probe_key(name, int(doc['revision']))),
                refresh=WRITE_REFRESH,
            )
    await bump_config_revision(client, store.index)
    await _reload(client)


async def record_probe(
    client: Any, *, name: str, revision: int | None, body: VlmEndpointBody, record: VlmProbeRecord
) -> None:
    """Persist ``record`` as the last probe of ``name@revision`` (``revision``
    is ``None`` only for the ``env`` built-in); another revision's probe is
    never touched. Never bumps the endpoint revision; DOES bump the config revision, so every process (and every
    worker's quiesce-and-swap) sees the new ``json_mode`` /
    ``max_model_len`` / model root. ``body`` fingerprints what was probed."""
    store = get_global_config_store()
    payload = {'fingerprint': probe_fingerprint(body), 'record': record.model_dump()}
    key = probe_key(name, revision)
    await client.index(
        index=store.index,
        id=probe_doc_id(key),
        body={
            'doc_type': 'vlm_probe',
            'name': name,
            'probe_key': key,
            'probed_at': record.probed_at,
            'body': payload,
        },
        refresh=WRITE_REFRESH,
    )
    await bump_config_revision(client, store.index)
    await _reload(client)


async def set_local_desired(client: Any, *, catalog_id: str) -> dict[str, Any]:
    store = get_global_config_store()
    doc = {'doc_type': 'local_vlm', 'catalog_id': catalog_id, 'requested_at': _now()}
    await client.index(index=store.index, id=LOCAL_DESIRED_DOC_ID, body=doc, refresh=WRITE_REFRESH)
    await bump_config_revision(client, store.index)
    await _reload(client, 'local_vlm')
    return doc


async def clear_local_desired(client: Any) -> None:
    store = get_global_config_store()
    with contextlib.suppress(NotFoundError):
        await client.delete(index=store.index, id=LOCAL_DESIRED_DOC_ID, refresh=WRITE_REFRESH)
    await bump_config_revision(client, store.index)
    await _reload(client, 'local_vlm')


__all__ = [
    'RevisionConflictError',
    'clear_local_desired',
    'delete_endpoint',
    'list_revisions',
    'record_probe',
    'save_endpoint',
    'set_local_desired',
]
