"""An in-memory index and a counting encoder for the re-index job tests (no OpenSearch, no Triton)."""

from __future__ import annotations

import uuid
from collections import Counter
from typing import Any

import psycopg
from sqlalchemy.orm import Session

from src.db.repos import embedding_spaces as repo
from src.domain.embedding_spaces import Target, index_role_for, physical_names
from src.services.embedding_spaces.catalog import sync_catalog
from src.services.embedding_spaces.reindex import DocRef, Outcome, ReindexMode, VectorWrite


KEY = 'mobileclip2_s2'


def seed_space(
    pg: Any, project: uuid.UUID, *, status: str = 'pending', target: Target = Target.ITEM
) -> uuid.UUID:
    """A candidate space of the shipped catalog's first image model, inserted past the service."""
    with Session(pg.engine('app')) as session:
        sync_catalog(session)
    names = physical_names(KEY, target)
    space = uuid.uuid4()
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        version = conn.execute(
            'SELECT id, native_dim FROM embedding_model_versions WHERE model_key = %s AND version = 1',
            (KEY,),
        ).fetchone()
        assert version is not None
        conn.execute(
            'INSERT INTO embedding_spaces (id, project_id, space_key, model_version_id, target, dim, '
            'role, status, index_role, vector_field, state_field, nested_path, knn_method, created_by) '
            "VALUES (%s, %s, %s, %s, %s, %s, 'candidate', %s, %s, %s, %s, %s, "
            '\'{"engine": "lucene"}\'::jsonb, \'test\')',
            (
                space,
                project,
                KEY,
                version[0],
                target.value,
                version[1],
                status,
                index_role_for(target),
                names.vector_field,
                names.state_field,
                names.nested_path,
            ),
        )
    return space


def vector_for(doc_id: str) -> list[float]:
    seed = sum(map(ord, doc_id))
    return [1.0, (seed % 7) / 7, (seed % 11) / 11]


class FakeIndex:
    """Documents ``d000``.. in id order; a box ``moved`` after it was read is skipped."""

    def __init__(self, count: int, *, boxes: bool = False) -> None:
        self.bbox: dict[str, tuple[float, ...] | None] = {
            f'd{i:03d}': ((0.0, 0.0, 0.5, 0.5) if boxes else None) for i in range(count)
        }
        self.vectors: dict[str, list[float]] = {}
        self.state: dict[str, str] = {}
        self.mapped: list[str] = []
        self.on_write: Any = None

    async def put_mapping(self, index_role: str, _properties: dict[str, Any]) -> None:
        self.mapped.append(index_role)

    def _ids(self, mode: ReindexMode) -> list[str]:
        ids = sorted(self.bbox)
        if mode is ReindexMode.BACKFILL:
            ids = [i for i in ids if i not in self.vectors]
        return ids

    async def count(self, _space: Any, mode: ReindexMode, _scope: Any) -> int:
        return len(self._ids(mode))

    async def page(
        self, _space: Any, mode: ReindexMode, _scope: Any, after: str | None, size: int
    ) -> list[DocRef]:
        ids = [i for i in self._ids(mode) if after is None or i > after][:size]
        return [DocRef(i, self.bbox[i]) for i in ids]

    async def write(self, _space: Any, writes: Any) -> dict[str, Outcome]:
        if self.on_write is not None:
            self.on_write(self)
        out: dict[str, Outcome] = {}
        for w in writes:
            if w.doc_id not in self.bbox:
                out[w.doc_id] = 'missing'
            elif self.bbox[w.doc_id] != w.expected_bbox:
                out[w.doc_id] = 'moved'
            else:
                self.vectors[w.doc_id] = list(w.vector)
                self.state[w.doc_id] = 'ready'
                out[w.doc_id] = 'written'
        return out

    async def mark(self, _space: Any, doc_ids: Any, state: str) -> None:
        for i in doc_ids:
            self.state[i] = state

    async def count_with_vector(self, _space: Any) -> int:
        return len(self.vectors)

    async def sample(self, _space: Any, _size: int) -> list[DocRef]:
        return [DocRef(i, self.bbox[i]) for i in sorted(self.vectors)[-5:]]

    async def read_vectors(self, _space: Any, doc_ids: Any) -> dict[str, list[float]]:
        return {i: self.vectors[i] for i in doc_ids if i in self.vectors}

    async def clear(self, _space: Any) -> None:
        self.vectors.clear()
        self.state.clear()


class CountingEncoder:
    """Counts encodes per document; ``die_at`` raises SystemExit (a killed worker) at that call."""

    def __init__(self, *, die_at: int | None = None, skip: frozenset[str] = frozenset()) -> None:
        self.calls: Counter[str] = Counter()
        self.die_at = die_at
        self.skip = skip
        self._n = 0

    async def encode(self, _space: Any, docs: Any) -> dict[str, list[float]]:
        self._n += 1
        if self.die_at is not None and self._n == self.die_at:
            raise SystemExit(1)
        for d in docs:
            self.calls[d.doc_id] += 1
        return {d.doc_id: vector_for(d.doc_id) for d in docs if d.doc_id not in self.skip}


__all__ = ['KEY', 'CountingEncoder', 'FakeIndex', 'VectorWrite', 'repo', 'seed_space', 'vector_for']
