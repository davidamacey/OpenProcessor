"""Whole-frame near-duplicate grouping over whole-frame embeddings.

Canonical, pure-numpy core shared by offline analysis tooling and any
training-data export, so both group near-duplicates identically.
Callers fetch the images index's whole-frame embedding vectors
themselves (sync or async OpenSearch client); this module only does
the math.

Method: the whole-frame embeddings are unit-norm, so a dot product is
cosine similarity. We form an undirected graph linking any pair with cosine
``>= threshold`` and take connected components (union-find) as near-dup groups.
The cosine pass is exact (full ``X·Xᵀ`` in row blocks) — deterministic, which
matters for a reproducible frozen dataset.

Threshold convention: **0.98** is the training cut. At/above 0.98 frames
are true duplicates (same subject, same angle, same moment) and collapse to
one representative; below that (0.96-0.97 = same subject, different angle)
they are KEPT as useful viewpoint variation, not redundancy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from src.config import CurationConfig, get_curation_config


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


DEFAULT_FRAME_DEDUP_THRESHOLD = 0.98
DEFAULT_EMBEDDING_FIELD = 'pe_embedding'


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._p = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self._p
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[ra] = rb


def near_dup_groups(
    mat: np.ndarray, threshold: float = DEFAULT_FRAME_DEDUP_THRESHOLD, block: int = 2048
) -> list[list[int]]:
    """Return connected-component groups (size >= 2) of near-duplicate rows.

    Args:
        mat: ``(n, d)`` float32 array of unit-norm embeddings (dot == cosine).
        threshold: cosine cutoff; pairs ``>= threshold`` are linked.
        block: row-block size for the chunked ``X·Xᵀ`` pass (memory knob).

    Returns:
        Groups of row indices that are mutually near-duplicate (transitively),
        each of length >= 2. Singletons are omitted.
    """
    n = mat.shape[0]
    if n < 2:
        return []
    uf = _UnionFind(n)
    mat_t = mat.T
    for r0 in range(0, n, block):
        r1 = min(r0 + block, n)
        sims = mat[r0:r1] @ mat_t  # (b, n) cosine
        rows, cols = np.where(sims >= threshold)
        gi = rows + r0
        keep = cols > gi  # upper triangle only; drops self-pairs + double counts
        for a, b in zip(gi[keep], cols[keep], strict=True):
            uf.union(int(a), int(b))
    roots: dict[int, list[int]] = {}
    for i in range(n):
        roots.setdefault(uf.find(i), []).append(i)
    return [g for g in roots.values() if len(g) > 1]


def most_central(mat: np.ndarray, group: list[int]) -> int:
    """Index (into ``group``) of the most central member (max summed cosine).

    The representative kept when a near-dup group is collapsed — the frame most
    typical of the burst, rather than an arbitrary one.
    """
    sub = mat[group]
    centrality = (sub @ sub.T).sum(axis=1)
    return int(np.argmax(centrality))


def collapse_keep_indices(
    mat: np.ndarray,
    threshold: float = DEFAULT_FRAME_DEDUP_THRESHOLD,
    *,
    prefer: np.ndarray | None = None,
) -> tuple[list[int], int, int]:
    """Pick which row indices to KEEP after collapsing near-dup groups.

    Every singleton is kept. Each near-dup group collapses to one survivor: if
    ``prefer`` is given, a preferred member wins (e.g. a held-out test frame, so
    its near-twins can't leak into train); otherwise the most-central frame.

    Args:
        mat: ``(n, d)`` unit-norm embeddings.
        threshold: cosine cutoff.
        prefer: optional bool array length ``n`` — members flagged True are
            preferred as the surviving representative.

    Returns:
        ``(keep_indices_sorted, n_groups, n_dropped)``.
    """
    n = mat.shape[0]
    groups = near_dup_groups(mat, threshold)
    grouped: set[int] = set()
    keep: list[int] = []
    n_dropped = 0
    for g in groups:
        grouped.update(g)
        rep = None
        if prefer is not None:
            preferred = [i for i in g if prefer[i]]
            if preferred:
                # Keep the most central among preferred members.
                rep = preferred[most_central(mat, preferred)]
        if rep is None:
            rep = g[most_central(mat, g)]
        keep.append(rep)
        n_dropped += len(g) - 1
    keep.extend(i for i in range(n) if i not in grouped)
    return sorted(keep), len(groups), n_dropped


# --------------------------------------------------------------------------
# Async orchestration shared by export pipelines
# --------------------------------------------------------------------------
class _DedupRow(Protocol):
    """Minimal row shape the export dedup needs: an id and a holdout flag."""

    image_id: str
    has_test_crop: bool


async def _fetch_pe_embeddings(
    opensearch: AsyncOpenSearch, image_ids: list[str], *, index: str, field: str
) -> dict[str, list[float]]:
    """mget the whole-frame embedding for each id, keyed by image_id."""
    out: dict[str, list[float]] = {}
    for i in range(0, len(image_ids), 1000):
        chunk = image_ids[i : i + 1000]
        resp = await opensearch.mget(index=index, body={'ids': chunk}, _source=['image_id', field])
        for d in resp.get('docs', []):
            s = d.get('_source') or {}
            emb = s.get(field)
            if emb is not None:
                out[str(s.get('image_id') or d.get('_id'))] = emb
    return out


async def dedup_rows_by_embedding[R: _DedupRow](
    opensearch: AsyncOpenSearch,
    rows: list[R],
    *,
    threshold: float = DEFAULT_FRAME_DEDUP_THRESHOLD,
    index: str | None = None,
    field: str = DEFAULT_EMBEDDING_FIELD,
    config: CurationConfig | None = None,
) -> tuple[list[R], dict[str, Any]]:
    """De-duplicate ``rows`` at the *source-frame* level by near-dup embedding.

    Each row must expose ``image_id`` (join to ``index``'s whole-frame ``field``
    embedding) and ``has_test_crop`` (held-out flag). Rows are grouped by
    ``image_id``; near-duplicate *frames* (cosine ``>= threshold``) collapse to a
    single representative frame, and **every** row of a surviving frame is kept.
    This is what makes the function correct for both export modes: whole-frame
    (one row per frame -> ordinary frame dedup) and item-crop (many crop rows
    per frame -> all crops of a kept frame survive, distinct items in one
    frame are never mistaken for duplicates of each other). A held-out frame is
    preferred as its group's survivor, so near-twins are dropped from train
    rather than the held-out frame — preventing train/val/test leakage. Rows
    with no embedding (or no ``image_id``) are kept untouched. Applied across the
    whole pool, so every cluster/dataset is deduped on the same footing.

    ``index`` defaults to the configured :class:`CurationConfig.images_index`
    when omitted, so callers never need to hardcode a deployment's index name.

    Returns ``(kept_rows, stats)``.
    """
    resolved_index = index if index is not None else (config or get_curation_config()).images_index

    without_id = [r for r in rows if not r.image_id]
    rows_by_frame: dict[str, list[R]] = {}
    for r in rows:
        if r.image_id:
            rows_by_frame.setdefault(r.image_id, []).append(r)

    frame_ids = list(rows_by_frame)
    id_to_vec = await _fetch_pe_embeddings(opensearch, frame_ids, index=resolved_index, field=field)
    embedded_ids = [fid for fid in frame_ids if fid in id_to_vec]
    missing_ids = [fid for fid in frame_ids if fid not in id_to_vec]

    # Rows we never touch: no image_id, or frame had no embedding.
    kept: list[R] = list(without_id)
    for fid in missing_ids:
        kept.extend(rows_by_frame[fid])

    base_stats: dict[str, Any] = {
        'enabled': True,
        'threshold': threshold,
        'n_input_rows': len(rows),
        'n_frames': len(frame_ids),
        'n_frames_embedded': len(embedded_ids),
        'n_frames_missing_embedding': len(missing_ids),
    }
    if len(embedded_ids) < 2:
        for fid in embedded_ids:
            kept.extend(rows_by_frame[fid])
        return rows, {
            **base_stats,
            'near_dup_groups': 0,
            'frames_dropped': 0,
            'rows_dropped': 0,
            'n_output_rows': len(rows),
        }

    mat = np.asarray([id_to_vec[fid] for fid in embedded_ids], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = np.where(norms > 0, mat / norms, mat).astype(np.float32)  # dot == cosine
    prefer = np.fromiter(
        (any(r.has_test_crop for r in rows_by_frame[fid]) for fid in embedded_ids),
        dtype=bool,
        count=len(embedded_ids),
    )
    keep_idx, n_groups, n_frames_dropped = collapse_keep_indices(mat, threshold, prefer=prefer)

    for i in keep_idx:
        kept.extend(rows_by_frame[embedded_ids[i]])
    return kept, {
        **base_stats,
        'near_dup_groups': n_groups,
        'frames_dropped': n_frames_dropped,
        'rows_dropped': len(rows) - len(kept),
        'n_output_rows': len(kept),
    }
