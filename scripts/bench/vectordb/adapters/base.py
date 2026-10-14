"""The one interface every engine adapter implements (plan section 5).

Engine client libraries are imported only inside the adapter that needs them, so this
module and ``vdb_lib`` stay importable (and testable) without any of them installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from scripts.bench.vectordb.vdb_lib import FilterSpec


Meta = dict[str, np.ndarray]


def row_batches(n: int, batch: int) -> Iterator[tuple[int, int]]:
    for start in range(0, n, batch):
        yield start, min(n, start + batch)


def run_threads(fn: Callable[[int, int], None], n: int, batch: int, threads: int) -> None:
    """Call ``fn(start, stop)`` for every batch of ``n`` rows on ``threads`` client threads."""
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for fut in [pool.submit(fn, a, b) for a, b in row_batches(n, batch)]:
            fut.result()


class Adapter(ABC):
    """One engine in one configuration (an "arm").

    ``arm`` is the plain-dict configuration from ``arms.py``; ``endpoint`` is the host:port
    (or a directory for the embedded engine). Row ids are the row positions of the dataset.
    """

    engine: str = ''
    container: str | None = None  # compose service name, used for docker stats and du
    data_path: str | None = None  # directory inside the container that holds the data

    def __init__(self, arm: dict[str, Any], endpoint: str) -> None:
        self.arm = arm
        self.endpoint = endpoint

    @abstractmethod
    def create(self, dim: int) -> None:
        """Drop and create the collection / index / table for this arm."""

    @abstractmethod
    def encode(self, vectors: np.ndarray, meta: Meta, ids: np.ndarray) -> Any:
        """Turn rows into the engine's wire payload (client CPU, kept outside the timed load)."""

    @abstractmethod
    def send(self, payload: Any) -> None:
        """Write one encoded payload to the engine."""

    def prepare(self, vectors: np.ndarray, meta: Meta, batch: int) -> None:
        """Encode every batch before the clock starts, as the write-ceiling probe does."""
        self.payloads = [
            self.encode(vectors[a:b], {f: v[a:b] for f, v in meta.items()}, np.arange(a, b))
            for a, b in row_batches(len(vectors), batch)
        ]

    def load(self, threads: int) -> None:
        """Send the prepared batches from ``threads`` client threads."""
        with ThreadPoolExecutor(max_workers=threads) as pool:
            for fut in [pool.submit(self.send, p) for p in self.payloads]:
                fut.result()
        self.payloads = []

    @abstractmethod
    def finish_load(self) -> None:
        """Make the data searchable: refresh, flush, index build, merge, wait until idle."""

    @abstractmethod
    def search(
        self, vector: np.ndarray, k: int, flt: FilterSpec, params: dict[str, Any]
    ) -> list[int]:
        """Ids of the top ``k`` neighbours under the filter (thread safe)."""

    @abstractmethod
    def update_field(self, ids: np.ndarray, field: str, values: np.ndarray, threads: int) -> None:
        """Rewrite one scalar field of the given rows (the cluster-pass write-back)."""

    @abstractmethod
    def delete_where(self, flt: FilterSpec) -> None:
        """Delete every row the filter keeps."""

    @abstractmethod
    def settle(self) -> None:
        """Reclaim space after deletes (merge, compaction, vacuum) and wait."""

    @abstractmethod
    def bytes_on_disk(self) -> int:
        """Storage of the collection as the engine reports or the data directory shows."""

    def insert_rows(self, vectors: np.ndarray, meta: Meta, ids: np.ndarray) -> None:
        """Streaming insert of extra rows with explicit ids."""
        self.send(self.encode(vectors, meta, ids))

    def search_grid(self) -> list[dict[str, Any]]:
        """Query-time parameter settings to sweep, cheapest first."""
        return [dict(p) for p in self.arm.get('grid', [{}])]

    def engine_stats(self) -> dict[str, Any]:
        """Engine-internal memory, segment and merge figures worth recording (default: none)."""
        return {}

    def count(self) -> int:
        return -1

    def drop(self) -> None:  # noqa: B027  # optional hook, empty by design
        """Remove the collection (the container is torn down by compose)."""

    def close(self) -> None:  # noqa: B027  # optional hook, empty by design
        """Release client resources."""
