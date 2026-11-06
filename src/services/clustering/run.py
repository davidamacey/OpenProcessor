"""The ``cluster_run`` body: fit, score, carry identities over, record one partition.

The pool arrives sorted by item key (the snapshot is canonical, so the same vectors in any arrival
order give the same fit), is fitted by the profile's method (FAISS k-means for ``ivf``,
scikit-learn MiniBatchKMeans for ``minibatch_kmeans``), scored against the cluster prototypes,
matched to the parent partition's identities and written with ``record_partition``. Activation is
a separate step (``activation``); the outcome says whether the run is safe to auto-activate.
References: https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
and https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import text

from src.services.clustering.assign import AssignBands, band_state
from src.services.clustering.carry_over import carry_over
from src.services.clustering.partitions import (
    PartitionMember,
    PartitionRecord,
    PartitionSpace,
    record_partition,
)


if TYPE_CHECKING:
    import uuid
    from collections.abc import Collection, Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

METHOD_VERSION = '1'
DEFAULT_SEED = 42  # today's IVF trains with it, so a first run reproduces today's centroids
MIN_POOL = 2
IVF_DEFAULT_N_CLUSTERS = 512
IVF_MIN_VECTORS_PER_CLUSTER = 6
IVF_MAX_TRAIN_SAMPLE = 50_000
IVF_NITER = 50
REGION_TARGET_BUCKET_SIZE = 200
SIM_CHUNK = 20_000
METHODS = ('ivf', 'minibatch_kmeans')


class RunRefused(ValueError):  # noqa: N818 - a refusal, not a fault
    """The inputs cannot make a partition (too few vectors, an unknown method)."""


@dataclass(frozen=True)
class Fit:
    labels: np.ndarray
    """int64, dense ``0..k-1`` (this module never emits noise)."""
    centroids: np.ndarray
    backend: str


@dataclass(frozen=True)
class RunOutcome:
    partition_id: uuid.UUID
    n_items: int
    n_clusters: int
    diff: dict[str, Any]
    stability: float
    anchored_moves: int
    validated_moves: int
    metrics: dict[str, Any] = field(default_factory=dict)


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _k_for(method: str, n: int, params: Mapping[str, Any]) -> int:
    if method == 'ivf':
        wanted = int(params.get('n_clusters') or IVF_DEFAULT_N_CLUSTERS)
        return max(1, min(wanted, n // IVF_MIN_VECTORS_PER_CLUSTER))
    bucket = int(params.get('target_bucket_size') or REGION_TARGET_BUCKET_SIZE)
    return min(max(8, round(n / bucket)), n)


def _fit_ivf(x: np.ndarray, k: int, params: Mapping[str, Any], seed: int) -> Fit:
    import faiss

    cap = int(params.get('max_train_sample') or IVF_MAX_TRAIN_SAMPLE)
    train = x
    if len(x) > cap:
        picked = np.random.default_rng(seed).choice(len(x), size=cap, replace=False)
        train = np.ascontiguousarray(x[np.sort(picked)])
    kmeans = faiss.Kmeans(
        x.shape[1],
        k,
        niter=int(params.get('niter') or IVF_NITER),
        nredo=int(params.get('nredo') or 1),
        seed=seed,
        spherical=True,
        verbose=False,
        max_points_per_centroid=max(256, len(train) // max(k, 1) + 1),
    )
    kmeans.train(train)
    centroids = np.asarray(kmeans.centroids, dtype=np.float32)
    _, nearest = kmeans.index.search(x, 1)
    return Fit(nearest[:, 0].astype(np.int64), centroids, 'faiss_cpu')


def _fit_minibatch(x: np.ndarray, k: int, seed: int) -> Fit:
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096)
    labels = model.fit_predict(x)
    return Fit(
        labels.astype(np.int64), np.asarray(model.cluster_centers_, dtype=np.float32), 'sklearn'
    )


def fit_pool(method: str, x: np.ndarray, params: Mapping[str, Any], seed: int) -> Fit:
    """Fit ``method`` on unit-norm rows; deterministic for equal rows, params and seed."""
    if method not in METHODS:
        raise RunRefused(f'unknown clustering method {method!r}')
    if len(x) < MIN_POOL:
        raise RunRefused(f'{len(x)} vectors cannot make a partition')
    k = _k_for(method, len(x), params)
    return _fit_ivf(x, k, params, seed) if method == 'ivf' else _fit_minibatch(x, k, seed)


def score_rows(
    x: np.ndarray, labels: np.ndarray, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine to the own cluster's unit centroid, and the margin to the best other centroid."""
    unit = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    scores = np.empty(len(x), dtype=np.float32)
    margins = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), SIM_CHUNK):
        sims = x[start : start + SIM_CHUNK] @ unit.T
        rows = np.arange(len(sims))
        own = sims[rows, labels[start : start + SIM_CHUNK]]
        sims[rows, labels[start : start + SIM_CHUNK]] = -np.inf
        other = sims.max(axis=1) if sims.shape[1] > 1 else own - 1.0
        scores[start : start + SIM_CHUNK] = own
        margins[start : start + SIM_CHUNK] = own - other
    return scores, margins


def input_digest(keys: Sequence[str], method: str, params: Mapping[str, Any], seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(f'{method}|{sorted(params.items())}|{seed}|{len(keys)}\n'.encode())
    for key in keys:
        digest.update(key.encode())
        digest.update(b'\n')
    return digest.hexdigest()


def _prototypes(
    keys: Sequence[str], labels: np.ndarray, scores: np.ndarray, fit: Fit
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, float]]:
    """One prototype per cluster: the unit centroid, its size, and the nearest member's key."""
    protos: dict[int, list[dict[str, Any]]] = {}
    dispersion: dict[int, float] = {}
    order = np.argsort(labels, kind='stable')
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    for chunk in np.split(order, bounds):
        label = int(labels[chunk[0]])
        centroid = fit.centroids[label] / max(float(np.linalg.norm(fit.centroids[label])), 1e-12)
        best = chunk[int(scores[chunk].argmax())]
        protos[label] = [
            {
                'vector': [round(float(v), 6) for v in centroid],
                'count': len(chunk),
                'item_key': keys[best],
            }
        ]
        dispersion[label] = float(1.0 - scores[chunk].mean())
    return protos, dispersion


def _diff(
    parent_labels: Mapping[str, uuid.UUID],
    keys: Sequence[str],
    uid_of_key: Mapping[str, uuid.UUID | None],
    counts: Mapping[str, int],
    protected: Collection[str],
    pinned: Mapping[str, Any],
) -> dict[str, Any]:
    moved = [k for k in keys if k in parent_labels and parent_labels[k] != uid_of_key.get(k)]
    return {
        **counts,
        'moved_items': len(moved),
        'new_items': sum(1 for k in keys if k not in parent_labels),
        'removed_items': sum(1 for k in parent_labels if k not in uid_of_key),
        'anchored_moves': len(pinned),
        'validated_moves': sum(1 for k in moved if k in protected),
    }


async def run_partition(
    session: AsyncSession,
    *,
    project: uuid.UUID | None,
    target: str,
    level: str,
    space: PartitionSpace,
    method: str,
    params: Mapping[str, Any],
    keys: Sequence[str],
    vectors: np.ndarray,
    parent_partition_id: uuid.UUID | None = None,
    bands: tuple[uuid.UUID, AssignBands] | None = None,
    profile: tuple[uuid.UUID, int] | None = None,
    protected_keys: Collection[str] = frozenset(),
    seed: int = DEFAULT_SEED,
    job_id: uuid.UUID | None = None,
    created_by: str = 'system',
) -> RunOutcome:
    """Fit the pool and write the partition (status ``ready``) in the caller's transaction."""
    if len(keys) != len(vectors):
        raise RunRefused('keys and vectors differ in length')
    started = time.perf_counter()
    order = sorted(range(len(keys)), key=keys.__getitem__)
    sorted_keys = [keys[i] for i in order]
    x = unit_rows(np.asarray(vectors)[order])
    fit = await asyncio.to_thread(fit_pool, method, x, params, seed)
    fitted = time.perf_counter()
    scores, margins = score_rows(x, fit.labels, fit.centroids)
    carry = await carry_over(
        session,
        keys=sorted_keys,
        labels=fit.labels.tolist(),
        parent_partition_id=parent_partition_id,
    )
    protos, dispersion = await asyncio.to_thread(_prototypes, sorted_keys, fit.labels, scores, fit)
    band_body = bands[1] if bands else None
    members: list[PartitionMember] = []
    uid_of_key: dict[str, uuid.UUID | None] = {}
    for i, key in enumerate(sorted_keys):
        uid = carry.pinned.get(key) or carry.uid_of_label[int(fit.labels[i])]
        state = (
            'assigned'
            if key in carry.pinned
            else band_state(float(scores[i]), float(margins[i]), band_body)
        )
        if state == 'unassigned':
            members.append(
                PartitionMember(key, None, 'unassigned', float(scores[i]), float(margins[i]))
            )
            uid_of_key[key] = None
            continue
        members.append(PartitionMember(key, uid, state, float(scores[i]), float(margins[i])))
        uid_of_key[key] = uid
    parent_labels = await _parent_labels(session, parent_partition_id)
    diff = _diff(parent_labels, sorted_keys, uid_of_key, carry.counts, protected_keys, carry.pinned)
    diff['stability'] = carry.stability
    record = PartitionRecord(
        method_version=METHOD_VERSION,
        profile=profile,
        calibration_id=bands[0] if bands else None,
        input_digest=input_digest(sorted_keys, method, params, seed),
        seed=seed,
        backend='cpu',
        backend_detail={'library': fit.backend},
        parent_partition_id=parent_partition_id,
        job_id=job_id,
        created_by=created_by,
        metrics={
            'mean_score': float(scores.mean()),
            'largest_cluster': max((p[0]['count'] for p in protos.values()), default=0),
        },
        diff=diff,
        stage_durations={'fit_s': round(fitted - started, 3)},
        lineage=carry.lineage,
        prototypes={carry.uid_of_label[label]: p for label, p in protos.items()},
        dispersion={carry.uid_of_label[label]: d for label, d in dispersion.items()},
    )
    partition_id = await record_partition(
        session,
        project=project,
        target=target,
        level=level,
        space=space,
        method=method,
        params=params,
        members=members,
        record=record,
    )
    return RunOutcome(
        partition_id=partition_id,
        n_items=len(members),
        n_clusters=len(carry.uid_of_label),
        diff=diff,
        stability=carry.stability,
        anchored_moves=int(diff['anchored_moves']),
        validated_moves=int(diff['validated_moves']),
        metrics=dict(record.metrics),
    )


async def _parent_labels(
    session: AsyncSession, parent_partition_id: uuid.UUID | None
) -> dict[str, uuid.UUID]:
    if parent_partition_id is None:
        return {}
    rows = await session.execute(
        text(
            'SELECT item_key, cluster_uid FROM cluster_memberships '
            'WHERE partition_id = :p AND cluster_uid IS NOT NULL'
        ),
        {'p': parent_partition_id},
    )
    return dict(rows.all())


__all__ = ['METHODS', 'Fit', 'RunOutcome', 'RunRefused', 'fit_pool', 'run_partition', 'score_rows']
