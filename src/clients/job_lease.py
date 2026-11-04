"""Job leases: one OpenSearch document per running singleton job.

A lease is the cross-process claim "this job is running": a document in the bound project's
SETTINGS-role index, written create-only and with optimistic concurrency, so two API workers
cannot both start one job and a sibling's live job is never taken over. Official behaviour
relied on (OpenSearch index API,
https://docs.opensearch.org/latest/api-reference/document-apis/index-document/):
``op_type=create`` indexes only when the id does not exist, and ``if_seq_no`` /
``if_primary_term`` reject a write whose version moved (409). ``get`` is realtime, so a read
by id needs no refresh (https://docs.opensearch.org/latest/api-reference/document-apis/get-documents/);
a lease is never read with ``search``.

Only the three ``_get`` / ``_create`` / ``_replace`` functions touch the client. A 409 on
acquire is an answer (:class:`Busy`), never a retry case; any other OpenSearch error becomes
an :class:`~src.utils.upstream_errors.UpstreamError` (HTTP 503), so a start whose OpenSearch
is down refuses instead of proceeding without a lease.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import NotFoundError, OpenSearchException

from src.config.curation_accessors import idx
from src.config.curation_roles import IndexRole
from src.core.logging import get_logger
from src.domain.job_lease import (
    LEASE_TTL_S,
    STATE_HELD,
    STATE_RELEASED,
    Acquired,
    Busy,
    HeldLease,
    LeaseKind,
    Lost,
    Renewed,
    doc_id,
    is_expired,
    is_takeable,
    next_generation,
    owner_id,
)
from src.utils.upstream_errors import UpstreamError


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

_Doc = tuple[dict[str, Any], int, int]


def is_conflict(exc: Exception) -> bool:
    """A version conflict (HTTP 409), by class name as ``clients/occ.py`` tests it."""
    return 'Conflict' in type(exc).__name__ or getattr(exc, 'status_code', None) == 409


async def _get(client: AsyncOpenSearch, did: str) -> _Doc | None:
    """The lease document ``(lease, seq_no, primary_term)``; ``None`` when absent."""
    try:
        resp = await client.get(index=idx(IndexRole.SETTINGS), id=did)
    except NotFoundError:
        return None
    lease = (resp.get('_source') or {}).get('lease')
    if not isinstance(lease, dict):
        return None
    return lease, int(resp['_seq_no']), int(resp['_primary_term'])


async def _create(client: AsyncOpenSearch, did: str, lease: dict[str, Any]) -> tuple[int, int]:
    resp = await client.index(
        index=idx(IndexRole.SETTINGS),
        id=did,
        body={'lease': lease},
        op_type='create',
        refresh=False,
    )
    return int(resp['_seq_no']), int(resp['_primary_term'])


async def _replace(
    client: AsyncOpenSearch, did: str, lease: dict[str, Any], seq_no: int, primary_term: int
) -> tuple[int, int]:
    resp = await client.index(
        index=idx(IndexRole.SETTINGS),
        id=did,
        body={'lease': lease},
        if_seq_no=seq_no,
        if_primary_term=primary_term,
        refresh=False,
    )
    return int(resp['_seq_no']), int(resp['_primary_term'])


def _document(
    kind: str,
    scope: str | None,
    job_id: str,
    owner: str,
    generation: int,
    now: float,
    ttl_s: float,
    acquired_at: float | None = None,
) -> dict[str, Any]:
    return {
        'kind': kind,
        'scope': scope,
        'job_id': job_id,
        'owner': owner,
        'generation': generation,
        'acquired_at': now if acquired_at is None else acquired_at,
        'heartbeat_at': now,
        'ttl_s': ttl_s,
        'state': STATE_HELD,
    }


async def _claim(
    client: AsyncOpenSearch,
    kind: LeaseKind | str,
    scope: str | None,
    job_id: str,
    owner: str | None,
    ttl_s: float,
    *,
    allow_released: bool,
) -> Acquired | Busy:
    did = doc_id(kind, scope)
    who = owner or owner_id()
    now = time.time()
    try:
        found = await _get(client, did)
    except (OpenSearchException, OSError) as exc:
        raise UpstreamError('opensearch', 'opensearch_unavailable') from exc
    try:
        if found is None:
            doc = _document(str(kind), scope, job_id, who, 1, now, ttl_s)
            seq_no, term = await _create(client, did, doc)
            generation = 1
        else:
            current, cur_seq, cur_term = found
            takeable = is_takeable(current, now) if allow_released else is_expired(current, now)
            if not takeable:
                return Busy(current.get('job_id'), float(current.get('heartbeat_at', 0.0)))
            generation = next_generation(current)
            doc = _document(str(kind), scope, job_id, who, generation, now, ttl_s)
            seq_no, term = await _replace(client, did, doc, cur_seq, cur_term)
    except (OpenSearchException, OSError) as exc:
        if not is_conflict(exc):
            raise UpstreamError('opensearch', 'opensearch_unavailable') from exc
        return Busy(None, None)
    return Acquired(HeldLease(str(kind), scope, job_id, who, generation, seq_no, term, now, ttl_s))


async def acquire(
    client: AsyncOpenSearch,
    kind: LeaseKind | str,
    scope: str | None,
    job_id: str,
    owner: str | None = None,
    *,
    ttl_s: float = LEASE_TTL_S,
) -> Acquired | Busy:
    """Take the lease when it is absent, released or expired; otherwise :class:`Busy`."""
    return await _claim(client, kind, scope, job_id, owner, ttl_s, allow_released=True)


async def takeover_expired(
    client: AsyncOpenSearch,
    kind: LeaseKind | str,
    scope: str | None,
    job_id: str,
    owner: str | None = None,
) -> Acquired | Busy:
    """Take the lease only when it is absent or expired (a released lease means its
    holder finished cleanly, so a repair has nothing to take over)."""
    return await _claim(client, kind, scope, job_id, owner, LEASE_TTL_S, allow_released=False)


async def _rewrite(
    client: AsyncOpenSearch,
    lease: HeldLease,
    *,
    owner: str | None = None,
    state: str = STATE_HELD,
) -> HeldLease | None:
    """Rewrite the held document with a new ``heartbeat_at`` under the lease's own version
    (one round trip); ``None`` when the version moved (taken over) or the document is
    gone, both of which OpenSearch answers with a 409."""
    who = owner or lease.owner
    doc = _document(
        lease.kind,
        lease.scope,
        lease.job_id,
        who,
        lease.generation,
        time.time(),
        lease.ttl_s,
        lease.acquired_at,
    )
    doc['state'] = state
    try:
        seq_no, term = await _replace(client, lease.doc_id, doc, lease.seq_no, lease.primary_term)
    except Exception as exc:
        if is_conflict(exc):
            return None
        raise
    return replace(lease, owner=who, seq_no=seq_no, primary_term=term)


async def renew(client: AsyncOpenSearch, lease: HeldLease) -> Renewed | Lost:
    """Push ``heartbeat_at`` forward; :class:`Lost` when someone took the lease over."""
    done = await _rewrite(client, lease)
    return Lost() if done is None else Renewed(done)


async def transfer(client: AsyncOpenSearch, lease: HeldLease, new_owner: str) -> Renewed | Lost:
    """Hand the lease to another process (the queue worker that runs the job)."""
    done = await _rewrite(client, lease, owner=new_owner)
    return Lost() if done is None else Renewed(done)


async def release(client: AsyncOpenSearch, lease: HeldLease) -> bool:
    """Mark the lease released. ``False`` (logged, nothing overwritten) when it was taken
    over meanwhile."""
    done = await _rewrite(client, lease, state=STATE_RELEASED)
    if done is None:
        logger.warning('job_lease_release_lost', lease=lease.doc_id, generation=lease.generation)
    return done is not None


async def is_live(client: AsyncOpenSearch, kind: LeaseKind | str, scope: str | None = None) -> bool:
    """Whether a live holder has the lease right now: a held, unexpired document. An
    absent, released or expired lease means whoever wrote the record it guards is gone."""
    found = await _get(client, doc_id(kind, scope))
    if found is None:
        return False
    return not is_takeable(found[0], time.time())
