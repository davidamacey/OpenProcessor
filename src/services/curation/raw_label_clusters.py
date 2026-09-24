"""Semantic clustering of free-text VLM class labels into candidate sub-classes.

The VLM labeler stores its raw open-vocabulary answer on every item it
labels, whether or not the answer resolved to a registry class. Across a
large corpus the unresolved long tail holds real class signal: "pickup
truck", "ford pickup" and "pick-up" are one missing class, not three.
This module groups the distinct raw strings by text-embedding similarity
(average-linkage agglomerative clustering on cosine distance), names each
group after its most frequent member, and ranks the groups by item volume
so a curator sees the biggest candidate sub-classes first.

The output contract is the set of item fields that
``GET {api_prefix}/review/raw_label_clusters`` aggregates on — the
``*_FIELD`` constants below are the single definition both that endpoint
and the offline writer (``scripts/curation/cluster_raw_labels.py``) use.
The field *values* predate the generic port and are part of the live
items-index mapping (see
:func:`src.clients.curation_opensearch.ensure_items_label_cluster_fields`),
so they are not renamed here.

Cluster ids are ``blake2b(cluster_name)`` truncated to a positive int32:
re-running on grown data keeps a cluster's id stable as long as its
dominant term is unchanged, which is what the labeler UI keys on.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


# Pre-existing items-index field names (see module docstring) — named
# after the VLM integration that first wrote them. Referenced only through
# these constants so a future schema rename is a one-line change.
RAW_LABEL_FIELD = 'gemma_raw_label'
CLUSTER_ID_FIELD = 'gemma_label_cluster_id'
CLUSTER_NAME_FIELD = 'gemma_label_cluster_name'
CLUSTER_DISTANCE_FIELD = 'gemma_label_cluster_distance'
UNMATCHED_CLASS_SOURCE = 'gemma_unmatched'


@dataclass(frozen=True)
class ClusterAssignment:
    cluster_id: int
    cluster_name: str
    distance: float


def stable_cluster_id(cluster_name: str) -> int:
    """Deterministic positive int32 id from a cluster name (fits an ``integer`` mapping)."""
    digest = hashlib.blake2b(cluster_name.encode('utf-8'), digest_size=4).digest()
    return int.from_bytes(digest, 'big') & 0x7FFFFFFF


_TOKEN_RE = re.compile(r'[a-z0-9]+')


def hash_embed(terms: Sequence[str], dim: int = 512) -> np.ndarray:
    """Dependency-free fallback embedding: hashed word tokens + character trigrams.

    Not semantic — "pickup" and "truck" stay unrelated — but spelling and
    punctuation variants ("f-150" / "f150", "pick up" / "pickup") land
    close, which already collapses much of a VLM's phrasing noise. Rows
    are L2-normalized so cosine distance is well defined.
    """
    out = np.zeros((len(terms), dim), dtype=np.float32)
    for i, term in enumerate(terms):
        text = term.lower()
        tokens = _TOKEN_RE.findall(text)
        squashed = ''.join(tokens)
        features = [f'w:{t}' for t in tokens]
        padded = f'#{squashed}#'
        features.extend(f'c:{padded[j : j + 3]}' for j in range(max(0, len(padded) - 2)))
        for feat in features:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=4).digest(), 'big')
            out[i, h % dim] += 2.0 if feat.startswith('w:') else 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def _normalize_rows(vecs: np.ndarray) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def cluster_terms(
    terms: Sequence[str],
    counts: Sequence[int],
    embed: Callable[[list[str]], np.ndarray],
    *,
    distance_threshold: float = 0.30,
    min_count: int = 3,
) -> dict[str, ClusterAssignment]:
    """Cluster distinct raw labels; return ``{term: ClusterAssignment}``.

    Terms seen on fewer than ``min_count`` items are not clustered (their
    embeddings are the noisiest and they would otherwise glue unrelated
    groups together through single weak links); each becomes its own
    singleton cluster so the long tail stays visible in the UI.

    Args:
        terms: Distinct raw labels.
        counts: Item count per term, index-aligned with ``terms``.
        embed: Maps a list of strings to an ``(N, D)`` array.
        distance_threshold: Cosine-distance cut for average linkage.
        min_count: Minimum item count for a term to be clustered.
    """
    if len(terms) != len(counts):
        raise ValueError('terms and counts must be the same length')
    out: dict[str, ClusterAssignment] = {}
    common = [i for i, c in enumerate(counts) if c >= min_count]
    rare = [i for i, c in enumerate(counts) if c < min_count]

    if len(common) == 1:
        rare = common + rare
        common = []
    if common:
        from sklearn.cluster import AgglomerativeClustering

        vecs = _normalize_rows(embed([terms[i] for i in common]))
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage='average',
            distance_threshold=distance_threshold,
        ).fit_predict(vecs)
        groups: dict[int, list[int]] = {}
        for local, label in enumerate(labels):
            groups.setdefault(int(label), []).append(local)
        for members in groups.values():
            # Ties broken by term text so the name (and so the id) is deterministic.
            best = max(members, key=lambda m: (counts[common[m]], terms[common[m]]))
            name = terms[common[best]]
            cid = stable_cluster_id(name)
            centroid = vecs[members].mean(axis=0)
            centroid /= float(np.linalg.norm(centroid)) or 1.0
            for m in members:
                out[terms[common[m]]] = ClusterAssignment(
                    cluster_id=cid,
                    cluster_name=name,
                    distance=float(1.0 - float(np.dot(vecs[m], centroid))),
                )

    for i in rare:
        out[terms[i]] = ClusterAssignment(
            cluster_id=stable_cluster_id(terms[i]), cluster_name=terms[i], distance=0.0
        )
    return out


def rank_clusters(
    assignments: dict[str, ClusterAssignment],
    counts: dict[str, int],
    *,
    samples: int = 5,
) -> list[dict[str, Any]]:
    """Group assignments into candidate sub-classes ranked by item volume."""
    by_id: dict[int, dict[str, Any]] = {}
    for term, a in assignments.items():
        bucket = by_id.setdefault(
            a.cluster_id,
            {'cluster_id': a.cluster_id, 'cluster_name': a.cluster_name, 'n_items': 0, 'terms': []},
        )
        n = int(counts.get(term, 0))
        bucket['n_items'] += n
        bucket['terms'].append((term, n))
    ranked = sorted(by_id.values(), key=lambda b: (-b['n_items'], b['cluster_name']))
    for b in ranked:
        b['n_terms'] = len(b['terms'])
        b['sample_terms'] = [
            {'label': t, 'count': n} for t, n in sorted(b.pop('terms'), key=lambda x: (-x[1], x[0]))
        ][:samples]
    return ranked


def cluster_update_doc(assignment: ClusterAssignment, now: str) -> dict[str, Any]:
    """Partial item-document update written for one clustered item."""
    return {
        CLUSTER_ID_FIELD: int(assignment.cluster_id),
        CLUSTER_NAME_FIELD: str(assignment.cluster_name),
        CLUSTER_DISTANCE_FIELD: float(assignment.distance),
        'updated_at': now,
    }


__all__ = [
    'CLUSTER_DISTANCE_FIELD',
    'CLUSTER_ID_FIELD',
    'CLUSTER_NAME_FIELD',
    'RAW_LABEL_FIELD',
    'UNMATCHED_CLASS_SOURCE',
    'ClusterAssignment',
    'cluster_terms',
    'cluster_update_doc',
    'hash_embed',
    'rank_clusters',
    'stable_cluster_id',
]
