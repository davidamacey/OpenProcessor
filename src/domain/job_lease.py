"""Rules of the OpenSearch job lease (J-LEASE): kinds, document ids, expiry, results.

A lease is one document in the bound project's leases index, written create-only
and with ``if_seq_no`` / ``if_primary_term`` (see ``src/clients/job_lease.py`` for the
operations). This module holds only the pure rules both the client and the callers share.
"""

from __future__ import annotations

import enum
import os
import socket
import threading
from dataclasses import dataclass


LEASE_TTL_S = 120.0
"""A lease whose last renew is older than this is expired and may be taken over."""

HEARTBEAT_TICK_S = 15.0
"""Seconds between renews by the heartbeat thread (eight ticks fit in one TTL)."""

FENCE_MARGIN_S = 30.0
"""A holder stops writing when its last good renew is older than ``TTL - margin``."""

STATE_HELD = 'held'
STATE_RELEASED = 'released'


class LeaseKind(enum.StrEnum):
    """Every job family that holds a lease; the value is the document-id prefix."""

    REPROCESS = 'reprocess'
    ASSIST_RUN = 'assist_run'
    ASSIST_LOOP = 'assist_loop'
    DATASET_IMPORT = 'dataset_import'
    ITEM_SCORES = 'item_scores'
    PROBE = 'probe'
    COMBINE = 'combine'
    BULK_INGEST = 'bulk_ingest'
    SOURCE_SCAN = 'source_scan'
    LABEL_OP = 'label_op'
    LABEL_UNDO = 'label_undo'
    SELECTION = 'selection'
    EMBEDDING_VIZ = 'embedding_viz'
    REGION_CLUSTER = 'region_cluster'
    REGION_FP_CENTROIDS = 'region_fp_centroids'
    AUTOLABEL = 'autolabel'


def doc_id(kind: LeaseKind | str, scope: str | None = None) -> str:
    """``lease:<kind>`` or ``lease:<kind>:<scope>``."""
    base = f'lease:{kind}'
    return f'{base}:{scope}' if scope else base


def owner_id() -> str:
    """``<hostname>:<pid>:<thread>``: who holds a lease, for the trail (no secrets)."""
    return f'{socket.gethostname()}:{os.getpid()}:{threading.current_thread().name}'


def is_expired(lease: dict[str, float | str], now: float) -> bool:
    """True once ``heartbeat_at + ttl_s`` is in the past."""
    return float(lease['heartbeat_at']) + float(lease.get('ttl_s', LEASE_TTL_S)) < now


def is_takeable(lease: dict[str, float | str], now: float) -> bool:
    """A released lease, or an expired one, may be acquired by anyone."""
    return lease.get('state') == STATE_RELEASED or is_expired(lease, now)


def next_generation(lease: dict[str, float | str] | None) -> int:
    """The generation a new holder takes: 1 for a new lease, else previous + 1."""
    return 1 if lease is None else int(lease.get('generation', 0)) + 1


@dataclass(frozen=True)
class HeldLease:
    """A lease this process holds and the version token its next write must present."""

    kind: str
    scope: str | None
    job_id: str
    owner: str
    generation: int
    seq_no: int
    primary_term: int
    acquired_at: float
    ttl_s: float = LEASE_TTL_S

    @property
    def doc_id(self) -> str:
        return doc_id(self.kind, self.scope)


@dataclass(frozen=True)
class Acquired:
    lease: HeldLease


@dataclass(frozen=True)
class Busy:
    """A live holder exists (or a concurrent writer won the race)."""

    job_id: str | None
    heartbeat_at: float | None


@dataclass(frozen=True)
class Renewed:
    lease: HeldLease


@dataclass(frozen=True)
class Lost:
    """The lease was taken over or removed; the holder must stop."""
