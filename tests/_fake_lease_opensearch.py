"""An OpenSearch stand-in for the lease documents: ``get`` by id, ``index`` with
``op_type='create'`` and ``if_seq_no`` / ``if_primary_term``, as the server enforces them.

Test scaffolding emulating the server's per-document atomicity, not production locking. The
store is any mapping (a ``multiprocessing.Manager`` dict for cross-process races) and the
lock any context manager (a ``multiprocessing`` lock for the same).
"""

# ruff: noqa: A002, ARG002 - mirrors the opensearch-py keyword names (id, refresh)
from __future__ import annotations

import copy
import json
from typing import Any

from opensearchpy.exceptions import ConflictError, NotFoundError


class FakeLeaseOpenSearch:
    def __init__(self, store: Any, lock: Any) -> None:
        self.store = store
        self.lock = lock
        self.fail_get: Exception | None = None
        self.closed = False

    async def get(self, *, index: str, id: str, **_: Any) -> dict[str, Any]:
        if self.fail_get is not None:
            raise self.fail_get
        with self.lock:
            entry = self.store.get(f'{index}/{id}')
            if entry is None:
                raise NotFoundError(404, 'not_found', {'found': False})
            raw, seq_no = entry
            return {'_source': json.loads(raw), '_seq_no': seq_no, '_primary_term': 1}

    async def index(
        self,
        *,
        index: str,
        id: str,
        body: dict[str, Any],
        op_type: str | None = None,
        if_seq_no: int | None = None,
        if_primary_term: int | None = None,
        refresh: Any = None,
    ) -> dict[str, Any]:
        key = f'{index}/{id}'
        with self.lock:
            entry = self.store.get(key)
            if op_type == 'create' and entry is not None:
                raise ConflictError(409, 'version_conflict_engine_exception', {})
            if if_seq_no is not None and (
                entry is None or entry[1] != if_seq_no or if_primary_term != 1
            ):
                raise ConflictError(409, 'version_conflict_engine_exception', {})
            seq_no = int(self.store.get('__seq__', 0)) + 1
            self.store['__seq__'] = seq_no
            self.store[key] = (json.dumps(copy.deepcopy(body)), seq_no)
            return {'_seq_no': seq_no, '_primary_term': 1}

    async def close(self) -> None:
        self.closed = True
