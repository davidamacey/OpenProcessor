"""Postgres advisory-lock classes: the first key of ``pg_advisory_lock(class, obj)``.

One place so two sections never pick the same number. ``BEAT`` and ``MIGRATE`` are session
locks on a direct connection; the others are ``pg_try_advisory_xact_lock(class, obj)`` taken
by a periodic task body, ``obj`` being the project's ``lock_key`` or 0.
"""

from __future__ import annotations


BEAT = 1
MIGRATE = 2
INGEST_DISPATCH = 10
OPEN_VOCAB_SWEEP = 11
CACHE_PRUNE = 12
GPU_ARBITER = 13
CLUSTER_REFRESH = 14
REAPER = 16
NEAR_DUP_RESCORE = 17
UPLOAD_SWEEP = 18
DB_BACKUP = 19
