"""Low-level OpenSearch doc-id, OCC and revision-counter primitives for
the config store (W2, any_domain_plan.md §3.1/§3.6).

This module is deliberately dependency-light: it knows about doc shapes
and OCC, nothing about prompt packs, region profiles or FastAPI. W3/W4's
CRUD routes and W2's own settings bridge both call these.

Doc ids (see any_domain_plan.md §3.1):
- ``pack:<name>`` / ``pack:<name>@<rev>`` (current / immutable revision copy)
- ``profile:<name>`` / ``profile:<name>@<rev>``
- ``activation:<axis>`` (``axis`` is ``prompt_pack`` | ``detection_profile``)
- ``activation_event:<uuid4>``
- ``meta:config_revision``
- ``runtime:<process>:<hostname>``
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Literal

from opensearchpy.exceptions import ConflictError, NotFoundError

from src.core.logging import get_logger


logger = get_logger(__name__)

ConfigKind = Literal['prompt_pack', 'region_profile']
ConfigAxis = Literal['prompt_pack', 'detection_profile']

KIND_TO_PREFIX: dict[str, str] = {'prompt_pack': 'pack', 'region_profile': 'profile'}

META_CONFIG_REVISION_DOC_ID = 'meta:config_revision'


def config_doc_id(kind: ConfigKind, name: str, revision: int | None = None) -> str:
    prefix = KIND_TO_PREFIX[kind]
    return f'{prefix}:{name}' if revision is None else f'{prefix}:{name}@{revision}'


def activation_doc_id(axis: ConfigAxis) -> str:
    return f'activation:{axis}'


def activation_event_doc_id() -> str:
    return f'activation_event:{uuid.uuid4()}'


def runtime_doc_id(process: str, hostname: str) -> str:
    return f'runtime:{process}:{hostname}'


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


class RevisionConflictError(Exception):
    """OCC mismatch on a stored config doc's ``expected_revision``."""

    def __init__(self, current_revision: int | None) -> None:
        self.current_revision = current_revision
        super().__init__(f'revision conflict (current_revision={current_revision})')


class ActiveConflictError(Exception):
    """OCC mismatch on an activation's ``expected_active``."""

    def __init__(self, current: dict[str, Any] | None) -> None:
        self.current = current
        super().__init__(f'active conflict (current={current})')


async def bump_config_revision(client: Any, index: str) -> int:
    """Atomically increment ``meta:config_revision`` (upsert-on-first-use)
    and return the new value. Every config mutation ends by calling this
    exactly once."""
    await client.update(
        index=index,
        id=META_CONFIG_REVISION_DOC_ID,
        body={
            'script': {'source': 'ctx._source.config_revision += 1', 'lang': 'painless'},
            'upsert': {'config_revision': 1, 'doc_type': 'meta'},
        },
        retry_on_conflict=5,
    )
    doc = await client.get(index=index, id=META_CONFIG_REVISION_DOC_ID)
    return int(doc['_source']['config_revision'])


async def get_config_revision(client: Any, index: str) -> int:
    try:
        doc = await client.get(index=index, id=META_CONFIG_REVISION_DOC_ID)
    except NotFoundError:
        return 0
    return int(doc['_source'].get('config_revision', 0))


async def _next_revision(
    client: Any, index: str, kind: ConfigKind, name: str, current_revision: int
) -> int:
    """The next revision id: current+1, or one past the highest existing
    immutable revision copy if the name was previously deleted (a
    revision id is never reused)."""
    if current_revision:
        return current_revision + 1
    resp = await client.search(
        index=index,
        body={
            'size': 1,
            'query': {
                'bool': {
                    'filter': [
                        {'term': {'doc_type': 'revision'}},
                        {'term': {'kind': kind}},
                        {'term': {'name': name}},
                    ]
                }
            },
            'sort': [{'revision': {'order': 'desc'}}],
        },
    )
    hits = resp['hits']['hits']
    if hits:
        return int(hits[0]['_source']['revision']) + 1
    return 1


async def save_config(
    client: Any,
    index: str,
    *,
    kind: ConfigKind,
    name: str,
    body: dict[str, Any],
    expected_revision: int | None,
    description: str = '',
    cloned_from: str | None = None,
) -> dict[str, Any]:
    """Create (``expected_revision=None``) or save a new revision
    (``expected_revision=<current>``) of a stored config doc.

    Writes the current doc (OCC on ``pack:<name>``/``profile:<name>``)
    **and** an immutable ``<kind>:<name>@<rev>`` copy, then bumps the
    global config revision. Raises :class:`RevisionConflictError` on a
    mismatch.
    """
    doc_id = config_doc_id(kind, name)
    now = _now_iso()
    seq_no = primary_term = None
    current_revision = 0
    created_at = now
    try:
        current = await client.get(index=index, id=doc_id)
        current_revision = int(current['_source'].get('revision', 0))
        seq_no = current['_seq_no']
        primary_term = current['_primary_term']
        created_at = current['_source'].get('created_at', now)
    except NotFoundError:
        pass

    if (expected_revision or 0) != current_revision:
        raise RevisionConflictError(current_revision)

    next_revision = await _next_revision(client, index, kind, name, current_revision)
    doc = {
        'doc_type': 'config',
        'kind': kind,
        'name': name,
        'revision': next_revision,
        'body': body,
        'description': description,
        'created_at': created_at,
        'updated_at': now,
        'updated_by': None,
        'cloned_from': cloned_from,
    }
    index_kwargs: dict[str, Any] = {'index': index, 'id': doc_id, 'body': doc}
    if seq_no is not None:
        index_kwargs['if_seq_no'] = seq_no
        index_kwargs['if_primary_term'] = primary_term
    try:
        await client.index(**index_kwargs)
    except ConflictError as exc:
        raise RevisionConflictError(None) from exc
    await client.index(
        index=index,
        id=config_doc_id(kind, name, next_revision),
        body={**doc, 'doc_type': 'revision'},
    )
    await bump_config_revision(client, index)
    return doc


async def delete_config(
    client: Any, index: str, *, kind: ConfigKind, name: str, expected_revision: int
) -> None:
    """Delete ``pack:<name>``/``profile:<name>`` (revision copies are
    kept). Raises :class:`RevisionConflictError` on an OCC mismatch,
    ``KeyError`` if the name doesn't exist."""
    doc_id = config_doc_id(kind, name)
    try:
        current = await client.get(index=index, id=doc_id)
    except NotFoundError as exc:
        raise KeyError(name) from exc
    current_revision = int(current['_source'].get('revision', 0))
    if expected_revision != current_revision:
        raise RevisionConflictError(current_revision)
    try:
        await client.delete(
            index=index,
            id=doc_id,
            if_seq_no=current['_seq_no'],
            if_primary_term=current['_primary_term'],
        )
    except ConflictError as exc:
        raise RevisionConflictError(None) from exc
    await bump_config_revision(client, index)


async def get_activation(client: Any, index: str, axis: ConfigAxis) -> dict[str, Any] | None:
    try:
        doc = await client.get(index=index, id=activation_doc_id(axis))
    except NotFoundError:
        return None
    return doc['_source']


async def activate(
    client: Any,
    index: str,
    *,
    axis: ConfigAxis,
    name: str | None,
    revision: int | None,
    expected_active: dict[str, Any] | None,
) -> dict[str, Any]:
    """Write ``activation:<axis>`` plus an ``activation_event`` and bump
    the global revision. ``name=None`` deactivates the axis.
    ``expected_active`` (``{"name": ..., "revision": ...}`` or ``None``
    for "currently off") must equal the current activation, else
    :class:`ActiveConflictError`.
    """
    doc_id = activation_doc_id(axis)
    seq_no = primary_term = None
    previous: dict[str, Any] | None = None
    try:
        current = await client.get(index=index, id=doc_id)
        previous = {
            'name': current['_source'].get('name'),
            'revision': current['_source'].get('revision'),
        }
        seq_no = current['_seq_no']
        primary_term = current['_primary_term']
    except NotFoundError:
        previous = None

    if expected_active != previous:
        raise ActiveConflictError(previous)

    now = _now_iso()
    doc = {
        'doc_type': 'activation',
        'axis': axis,
        'name': name,
        'revision': revision,
        'activated_at': now,
        'previous': previous,
    }
    index_kwargs: dict[str, Any] = {'index': index, 'id': doc_id, 'body': doc}
    if seq_no is not None:
        index_kwargs['if_seq_no'] = seq_no
        index_kwargs['if_primary_term'] = primary_term
    try:
        await client.index(**index_kwargs)
    except ConflictError as exc:
        raise ActiveConflictError(None) from exc

    await client.index(
        index=index,
        id=activation_event_doc_id(),
        body={
            'doc_type': 'activation_event',
            'axis': axis,
            'name': name,
            'revision': revision,
            'previous': previous,
            'activated_at': now,
        },
    )
    new_revision = await bump_config_revision(client, index)
    return {**doc, 'config_revision': new_revision}


async def rollback(
    client: Any, index: str, *, axis: ConfigAxis, expected_active: dict[str, Any] | None
) -> dict[str, Any]:
    """Re-activate ``previous``. Raises ``LookupError('no_previous')`` if
    there is none."""
    current = await get_activation(client, index, axis)
    previous = (current or {}).get('previous')
    if not previous:
        msg = 'no_previous'
        raise LookupError(msg)
    return await activate(
        client,
        index,
        axis=axis,
        name=previous.get('name'),
        revision=previous.get('revision'),
        expected_active=expected_active,
    )


async def upsert_runtime_doc(
    client: Any, index: str, *, process: str, hostname: str, fields: dict[str, Any]
) -> None:
    """Upsert ``runtime:<process>:<hostname>`` -- the worker's "what did I
    actually apply" record (§4.5), refreshed at swap and every 60s."""
    doc_id = runtime_doc_id(process, hostname)
    body = {'doc_type': 'runtime', 'process': process, **fields}
    await client.index(index=index, id=doc_id, body=body)


async def get_runtime_docs(client: Any, index: str, *, process: str) -> list[dict[str, Any]]:
    resp = await client.search(
        index=index,
        body={
            'size': 100,
            'query': {'bool': {'filter': [{'term': {'doc_type': 'runtime'}}]}},
        },
    )
    return [
        hit['_source']
        for hit in resp['hits']['hits']
        if hit['_id'].startswith(f'runtime:{process}:')
    ]
