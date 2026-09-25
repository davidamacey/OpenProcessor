"""Near-duplicate scorer — crop-level reuse of the validated whole-frame
primitive (curation-strategy plan §2.5).

:func:`src.services.detection.frame_dedup.near_dup_groups` (exact blocked
cosine + union-find connected components, already validated at threshold
0.98 for whole-frame image ``pe_embedding``) is reused verbatim here,
just applied to crop-level embeddings instead of frame-level ones. No new
math. New fields only: ``dup_group_id`` / ``dup_group_size`` /
``dup_is_representative`` (plan §2.5/§4).

**Threshold is a hypothesis, not a finding** (plan §2.5/§10.3): crops likely
need a tighter cut than the whole-frame 0.98 — a near-dup *photo* burst still
contains distinct crops (a truck and its trailer), so 0.98 was never
validated at crop granularity. ``OP_CROP_DUP_THRESHOLD`` (default 0.98,
matching :data:`frame_dedup.DEFAULT_FRAME_DEDUP_THRESHOLD` until the sweep
in plan §6 says otherwise) makes this a knob, not a hardcoded assumption.

**Bucket-scoped, not global** (plan §3 compute budget): an O(n²) pass over
the full residual pool is 1.2e11 pairs at 350k crops — a requirement to
avoid, not an optimization. When ``centroids`` are supplied (production
path), embeddings are first assigned to their nearest IVF centroid and
``near_dup_groups`` runs independently per bucket (≈3.2e7 pairs total across
512 buckets of ~700 crops each). Cross-bucket near-dups are consequently
never found — acceptable, since near-duplicate photos of the same subject
already land in the same or adjacent IVF bucket. Without centroids (small
pools — e.g. tests) it degrades to one global pass.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from src.services.curation.item_scores.base import ScoreResult
from src.services.detection.frame_dedup import (
    DEFAULT_FRAME_DEDUP_THRESHOLD,
    most_central,
    near_dup_groups,
)


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


NEAR_DUP_VERSION = 'v1'


def _threshold() -> float:
    try:
        return float(os.environ.get('OP_CROP_DUP_THRESHOLD', str(DEFAULT_FRAME_DEDUP_THRESHOLD)))
    except ValueError:
        return DEFAULT_FRAME_DEDUP_THRESHOLD


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _assign_buckets(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest-centroid assignment (int64 label per row), matching
    ``IVFCentroidStore.assign_batch``'s IndexFlatL2 convention but without
    needing a persisted store on disk — used to bucket-scope near-dup."""
    import faiss

    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    quantizer.add(np.ascontiguousarray(centroids, dtype=np.float32))
    _, labels = quantizer.search(np.ascontiguousarray(embeddings, dtype=np.float32), 1)
    return labels.reshape(-1).astype(np.int64)


def compute_near_dup_groups(
    embeddings: np.ndarray,
    *,
    threshold: float = DEFAULT_FRAME_DEDUP_THRESHOLD,
    centroids: np.ndarray | None = None,
) -> list[list[int]]:
    """Pure math core — row-index groups (each length >= 2), global row
    indices into ``embeddings`` (not bucket-local). Testable directly."""
    n = embeddings.shape[0]
    if n < 2:
        return []
    if centroids is None or centroids.shape[0] >= n:
        return near_dup_groups(embeddings, threshold=threshold)

    bucket_labels = _assign_buckets(embeddings, centroids)
    groups: list[list[int]] = []
    for bucket_id in np.unique(bucket_labels):
        member_idx = np.flatnonzero(bucket_labels == bucket_id)
        if member_idx.size < 2:
            continue
        sub_groups = near_dup_groups(embeddings[member_idx], threshold=threshold)
        groups.extend([int(member_idx[i]) for i in g] for g in sub_groups)
    return groups


class NearDupScorer:
    """``crop_scores`` registry entry wrapping :func:`compute_near_dup_groups`."""

    name: ClassVar[str] = 'near_dup'
    writes: ClassVar[tuple[str, ...]] = (
        'dup_group_id',
        'dup_group_size',
        'dup_is_representative',
        'dup_threshold',
        'dup_method',
        'dup_scored_at',
    )
    version: ClassVar[str] = NEAR_DUP_VERSION

    def __init__(self, threshold: float | None = None, centroids: np.ndarray | None = None) -> None:
        self.threshold = threshold if threshold is not None else _threshold()
        self._centroids_override = centroids

    def _load_centroids(self) -> np.ndarray:
        if self._centroids_override is not None:
            return self._centroids_override
        from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore

        store = IVFCentroidStore()
        if not store.load():
            # Fail loudly rather than silently degrading to an unscoped
            # global O(n^2) pass (plan §3: 1.2e11 pairs at 350k crops —
            # exactly the cost the bucket-scoping exists to avoid). A
            # missing store is a setup problem the caller needs to know
            # about, not a shape the scorer should quietly paper over.
            raise RuntimeError(
                'near_dup scorer requires a trained IVF centroid store '
                '(none found — run the clustering pipeline at least once first)'
            )
        index = store._index
        return np.asarray(index.reconstruct_n(0, index.ntotal), dtype=np.float32)

    async def score(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        *,
        opensearch: AsyncOpenSearch | None = None,  # noqa: ARG002 - protocol uniformity
        progress: Any = None,  # noqa: ARG002 - protocol uniformity
    ) -> ScoreResult:
        centroids = self._load_centroids()
        groups = compute_near_dup_groups(embeddings, threshold=self.threshold, centroids=centroids)
        now = _now_iso()
        fields: dict[str, dict[str, Any]] = {}
        for gi, group in enumerate(groups):
            rep_local = most_central(embeddings, group)
            group_id = f'dup_{gi}'
            for local_i, row_idx in enumerate(group):
                fields[ids[row_idx]] = {
                    'dup_group_id': group_id,
                    'dup_group_size': len(group),
                    'dup_is_representative': local_i == rep_local,
                    'dup_threshold': self.threshold,
                    'dup_method': 'crop_pe_cosine_unionfind',
                    'dup_scored_at': now,
                }
        return ScoreResult(
            scorer=self.name,
            version=self.version,
            scored_at=now,
            fields=fields,
            n_scored=len(fields),
            extra={'n_groups': len(groups), 'threshold': self.threshold},
        )


__all__ = ['NEAR_DUP_VERSION', 'NearDupScorer', 'compute_near_dup_groups']
