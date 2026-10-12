"""Reference arm: exact flat search on the CPU (FAISS for the unfiltered case, a masked
matrix product otherwise); an upper bound, not a datastore."""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.bench.vectordb.adapters.base import Adapter, Meta
from scripts.bench.vectordb.vdb_lib import FilterSpec, filter_mask


class FaissFlatAdapter(Adapter):
    engine = 'faiss_flat'

    def create(self, dim: int) -> None:
        self.dim = dim
        self.vectors = np.zeros((0, dim), np.float32)
        self.meta: Meta = {}

    def encode(self, vectors: np.ndarray, meta: Meta, ids: np.ndarray) -> Any:  # noqa: ARG002
        return vectors, meta

    def send(self, payload: Any) -> None:
        vectors, meta = payload
        self.vectors = np.ascontiguousarray(np.concatenate([self.vectors, vectors]))
        for k, v in meta.items():
            self.meta[k] = (
                np.concatenate([self.meta[k], v]) if k in self.meta else np.asarray(v).copy()
            )

    def load(self, threads: int) -> None:  # noqa: ARG002
        for p in self.payloads:
            self.send(p)
        self.payloads = []

    def finish_load(self) -> None:
        import faiss

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.vectors)
        self.masks: dict[str, np.ndarray] = {}

    def search(
        self,
        vector: np.ndarray,
        k: int,
        flt: FilterSpec,
        params: dict[str, Any],  # noqa: ARG002
    ) -> list[int]:
        if flt.name == 'none':
            _, ids = self.index.search(vector[None, :], k)
            return [int(i) for i in ids[0] if i >= 0]
        if flt.name not in self.masks:
            self.masks[flt.name] = np.flatnonzero(filter_mask(flt, self.meta)).astype(np.int64)
        rows = self.masks[flt.name]
        scores = self.vectors[rows] @ vector
        return [int(i) for i in rows[np.argsort(-scores)[:k]]]

    def update_field(self, ids: np.ndarray, field: str, values: np.ndarray, threads: int) -> None:  # noqa: ARG002
        self.meta[field][ids] = values
        self.masks = {}

    def delete_where(self, flt: FilterSpec) -> None:
        raise NotImplementedError

    def settle(self) -> None:
        return

    def bytes_on_disk(self) -> int:
        return int(self.vectors.nbytes)
