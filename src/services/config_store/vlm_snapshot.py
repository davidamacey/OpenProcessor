"""The VLM axis's part of a :class:`~src.services.config_store.store.ConfigSnapshot`
(W9.2). Kept out of ``store.py`` (size ratchet, and it is one concern).

Two indexes, two halves:

- the **global** store (``op_global_configs``) holds the endpoint registry:
  ``vlm:<name>`` current docs, ``vlm:<name>@<rev>`` immutable copies,
  ``vlm_probe:<name>`` last probes and ``local_vlm:desired``;
- each **project** store holds only ``activation:vlm`` (which endpoint that
  project runs, and which ``name@revision`` external endpoints it has
  acknowledged). The activated revision's body is resolved from the global
  registry's immutable copy and PINNED into that project's snapshot, so a
  later PUT to the endpoint never changes what a running project uses, and
  a cold worker process reconstructs exactly the same thing.

Reads of the global index while a project is bound go through
:func:`~src.services.projects.guard.global_configs_read`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import NotFoundError

from src.core.logging import get_logger


if TYPE_CHECKING:
    from src.services.config_store.store import AxisRef, StoredConfig

logger = get_logger(__name__)

LOCAL_DESIRED_DOC_ID = 'local_vlm:desired'


def probe_doc_id(name: str) -> str:
    return f'vlm_probe:{name}'


#: Immutable ``(name, revision) -> StoredConfig`` copies, process-wide.
#: Immutable by construction (revision numbers are never reused), so it is
#: safe to share across projects and to keep forever; bounded anyway.
_REVISION_CACHE: dict[tuple[str, int], Any] = {}
_REVISION_CACHE_MAX = 512


def reset_vlm_revision_cache() -> None:
    _REVISION_CACHE.clear()


def _stored(src: dict[str, Any], name: str) -> StoredConfig:
    from src.services.config_store.store import StoredConfig

    return StoredConfig(
        kind='vlm_endpoint',
        name=name,
        revision=int(src['revision']),
        body=src.get('body') or {},
        description=src.get('description') or '',
        created_at=src.get('created_at'),
        updated_at=src.get('updated_at'),
        cloned_from=src.get('cloned_from'),
    )


async def fetch_revision(client: Any, name: str, revision: int) -> StoredConfig | None:
    """``vlm:<name>@<revision>`` from the global registry (immutable, so
    cached). ``None`` only for a genuine 404; any other error propagates
    (never fail-open, §3.6)."""
    from src.services.config_store.global_store import global_configs_index
    from src.services.config_store.index import config_doc_id
    from src.services.projects.guard import global_configs_read

    key = (name, revision)
    cached = _REVISION_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        with global_configs_read():
            doc = await client.get(
                index=global_configs_index(), id=config_doc_id('vlm_endpoint', name, revision)
            )
    except NotFoundError:
        return None
    item = _stored(doc['_source'], name)
    if len(_REVISION_CACHE) >= _REVISION_CACHE_MAX:
        _REVISION_CACHE.pop(next(iter(_REVISION_CACHE)))
    _REVISION_CACHE[key] = item
    return item


async def load_global_fields(client: Any, index: str) -> dict[str, Any]:
    """Probes and the desired local model (endpoint docs arrive with the
    generic ``doc_type: config`` search in ``ConfigStore._load_snapshot``)."""
    from src.services.projects.guard import global_configs_read

    with global_configs_read():
        resp = await client.search(
            index=index,
            body={
                'size': 1000,
                'query': {'bool': {'filter': [{'term': {'doc_type': 'vlm_probe'}}]}},
            },
        )
        probes = {
            hit['_source']['name']: dict(hit['_source'].get('body') or {})
            for hit in resp['hits']['hits']
        }
        try:
            desired_doc = await client.get(index=index, id=LOCAL_DESIRED_DOC_ID)
            desired = dict(desired_doc['_source'])
        except NotFoundError:
            desired = None
    return {'vlm_probes': probes, 'local_vlm_desired': desired}


def axis_ref(activation_doc: dict[str, Any] | None) -> AxisRef:
    from src.services.config_store.store import _axis_ref

    return _axis_ref(activation_doc)


async def load_project_fields(client: Any, activation: dict[str, Any] | None) -> dict[str, Any]:
    """``activation:vlm`` of one project, with the activated revision's
    body pinned from the global registry."""
    ref = axis_ref(activation)
    body: StoredConfig | None = None
    if isinstance(ref, tuple) and ref[1] is not None:
        body = await fetch_revision(client, ref[0], ref[1])
        if body is None:
            logger.warning('vlm_active_revision_missing', name=ref[0], revision=ref[1])
    return {
        'active_vlm': ref,
        'active_vlm_body': body,
        'active_vlm_ack_at': (activation or {}).get('external_ack_at'),
        'acked_refs': frozenset((activation or {}).get('acked_refs') or ()),
    }


__all__ = [
    'LOCAL_DESIRED_DOC_ID',
    'axis_ref',
    'fetch_revision',
    'load_global_fields',
    'load_project_fields',
    'probe_doc_id',
    'reset_vlm_revision_cache',
]
