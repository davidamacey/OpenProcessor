"""The benchmark dataset: nested prefixes of one real embedding set, queries and exact ground truth.

Built from the output of ``export_embeddings.py`` (one or more export directories). Held-out images (never inserted) supply the
queries and the streaming-insert pool; the rest supplies the index set. Prefix sizes keep whole
images together in a seeded image order, so the 2k set is contained in the 10k set, and so on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.bench.vectordb.vdb_lib import (
    CLUSTER_COUNT,
    FilterSpec,
    exact_topk,
    filter_mask,
    filter_matrix,
    make_filter_fields,
)


if TYPE_CHECKING:
    from pathlib import Path


TRUTH_K = 100
HELD_OUT_QUERIES = 1000
IN_SET_QUERIES = 1000


@dataclass
class Dataset:
    size: int
    vectors: np.ndarray
    meta: dict[str, np.ndarray]
    held_queries: np.ndarray
    in_queries: np.ndarray
    in_query_rows: np.ndarray
    stream_vectors: np.ndarray
    stream_meta: dict[str, np.ndarray]
    filters: list[FilterSpec]
    root: Path

    def truth(self, qset: str, flt: FilterSpec) -> np.ndarray:
        """Exact top-100 row ids per query of ``qset`` ('held' or 'in') under ``flt``, cached."""
        cache = self.root / f'truth_{self.size}.npz'
        key = f'{qset}__{flt.name}'
        store = dict(np.load(cache)) if cache.exists() else {}
        if key not in store:
            queries = self.held_queries if qset == 'held' else self.in_queries
            mask = filter_mask(flt, self.meta)
            store[key] = exact_topk(self.vectors, queries, mask, TRUTH_K)
            np.savez(cache, **store)
        return store[key]


def kmeans_clusters(vectors: np.ndarray, seed: int) -> np.ndarray:
    import faiss

    km = faiss.Kmeans(vectors.shape[1], CLUSTER_COUNT, niter=15, seed=seed, spherical=True)
    km.train(vectors)
    _, assign = km.index.search(vectors, 1)
    return assign[:, 0].astype(np.int64)


def build(export_dirs: list[Path], out_dir: Path, seed: int = 20261011) -> None:
    """Write ``base.npz`` (index-set order, all fields) and ``queries.npz`` under ``out_dir``."""
    import pandas as pd

    vectors = np.concatenate([np.load(d / 'vectors.f32.npy') for d in export_dirs])
    meta = pd.concat([pd.read_parquet(d / 'meta.parquet') for d in export_dirs], ignore_index=True)
    meta['item_id'] = np.arange(len(meta), dtype=np.int64)
    held = meta['held_out'].to_numpy()
    rng = np.random.default_rng(seed)

    index_meta = meta[~held]
    images = index_meta['image_id'].unique()
    order = {int(i): n for n, i in enumerate(rng.permutation(images))}
    rank = index_meta['image_id'].map(order).to_numpy()
    rows = np.flatnonzero(~held)[np.argsort(rank, kind='stable')]
    base_vecs = np.ascontiguousarray(vectors[rows])
    base = meta.iloc[rows]
    fields = make_filter_fields(base['image_id'].to_numpy(), base['item_id'].to_numpy(), seed)
    fields['cluster_id'] = kmeans_clusters(base_vecs, seed)
    fields['class_name'] = base['class_name'].to_numpy().astype(str)
    fields['image_id'] = base['image_id'].to_numpy()
    fields['kind'] = base['kind'].to_numpy().astype(str)

    held_rows = np.flatnonzero(held)
    q_pick = rng.permutation(held_rows)
    n_q = min(HELD_OUT_QUERIES, len(held_rows) // 2)
    held_q = q_pick[:n_q]
    stream = q_pick[n_q:]
    in_rows = rng.choice(len(base_vecs), size=min(IN_SET_QUERIES, len(base_vecs)), replace=False)
    held_meta = meta.iloc[stream]
    stream_fields = make_filter_fields(
        held_meta['image_id'].to_numpy(), held_meta['item_id'].to_numpy(), seed + 1
    )
    stream_fields['cluster_id'] = rng.integers(0, CLUSTER_COUNT, len(stream)).astype(np.int64)
    stream_fields['class_name'] = held_meta['class_name'].to_numpy().astype(str)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'base.npz', vectors=base_vecs, **fields)  # type: ignore[arg-type]  # mypy reads **dict as the allow_pickle flag
    np.savez(
        out_dir / 'queries.npz',
        held=vectors[held_q],
        held_kind=meta.iloc[held_q]['kind'].to_numpy().astype(str),
        in_rows=in_rows,
        stream_vectors=vectors[stream],
        **{f'stream_{k}': v for k, v in stream_fields.items()},  # type: ignore[arg-type]  # mypy reads **dict as the allow_pickle flag
    )


def load(root: Path, size: int) -> Dataset:
    """The first ``size`` index vectors (whole images first), queries, and the filter matrix."""
    base = np.load(root / 'base.npz')
    queries = np.load(root / 'queries.npz')
    n = min(size, len(base['vectors']))
    meta = {k: base[k][:n] for k in base.files if k != 'vectors'}
    vectors = np.ascontiguousarray(base['vectors'][:n])
    in_rows = queries['in_rows'] % n
    project = 0
    in_proj = meta['project'] == project
    top_class = _most_common(meta['class_name'][in_proj & (meta['class_name'] != '_frame')])
    cluster = _most_common(meta['cluster_id'][in_proj])
    stream_meta = {
        k[7:]: queries[k]
        for k in queries.files
        if k.startswith('stream_') and k != 'stream_vectors'
    }
    return Dataset(
        size=n,
        vectors=vectors,
        meta=meta,
        held_queries=queries['held'],
        in_queries=vectors[in_rows],
        in_query_rows=in_rows,
        stream_vectors=queries['stream_vectors'],
        stream_meta=stream_meta,
        filters=filter_matrix(project, str(top_class), int(cluster)),
        root=root,
    )


def _most_common(values: np.ndarray) -> Any:
    uniq, counts = np.unique(values, return_counts=True)
    return uniq[np.argmax(counts)]
