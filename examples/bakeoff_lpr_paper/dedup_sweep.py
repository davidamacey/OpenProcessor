#!/usr/bin/env python3
"""Recompute the whole-frame near-duplicate threshold sweep for an LPR paper.

Example / write-up tooling, not part of the bake-off harness: it assumes the
license-plate example domain (frames carrying a confirmed region).

Pulls the images index's ``pe_embedding`` vector for every confirmed-plate
frame (the region-status field set to 'detected') and reports, per cosine
threshold, the number of near-duplicate groups, frames dropped (each group
collapses to one representative), the drop fraction, and the largest group.
This is the data behind a paper's dedup-threshold table and the justification
for the adopted tau=0.98 cut. Re-run whenever the confirmed-plate pool changes
(from the repo root)::

    .venv/bin/python -m examples.bakeoff_lpr_paper.dedup_sweep

Reuses the production near-dup core (``frame_dedup.near_dup_groups``) so the
sweep groups frames identically to the exporter.
"""

from __future__ import annotations

import logging

import numpy as np
import requests

from src.config import get_curation_config, get_region_fields
from src.config.region_state import RegionStatus
from src.services.detection.frame_dedup import near_dup_groups


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('dedup_sweep')

OS_URL = 'http://localhost:4607'
_cfg = get_curation_config()
CROPS_INDEX = _cfg.items_index
IMAGES_INDEX = _cfg.images_index
THRESHOLDS = (0.960, 0.970, 0.980, 0.990, 0.995)


def detected_frame_ids() -> list[str]:
    """Distinct ``image_id`` of every frame carrying a confirmed plate."""
    body = {
        'size': 1000,
        '_source': ['image_id'],
        'query': {'term': {f'{get_region_fields().status}.keyword': RegionStatus.DETECTED}},
    }
    resp = requests.post(f'{OS_URL}/{CROPS_INDEX}/_search?scroll=3m', json=body, timeout=60).json()
    ids: set[str] = set()
    scroll_id = resp.get('_scroll_id')
    hits = resp.get('hits', {}).get('hits', [])
    while hits:
        for h in hits:
            iid = (h.get('_source') or {}).get('image_id')
            if iid:
                ids.add(str(iid))
        resp = requests.post(
            f'{OS_URL}/_search/scroll',
            json={'scroll': '3m', 'scroll_id': scroll_id},
            timeout=60,
        ).json()
        scroll_id = resp.get('_scroll_id')
        hits = resp.get('hits', {}).get('hits', [])
    if scroll_id:
        requests.delete(f'{OS_URL}/_search/scroll', json={'scroll_id': [scroll_id]}, timeout=30)
    return sorted(ids)


def fetch_embeddings(ids: list[str]) -> np.ndarray:
    """mget the whole-frame PE embedding for each id; return unit-norm matrix."""
    vecs: list[list[float]] = []
    for i in range(0, len(ids), 1000):
        chunk = ids[i : i + 1000]
        docs = [{'_id': d, '_source': ['pe_embedding']} for d in chunk]
        resp = requests.post(
            f'{OS_URL}/{IMAGES_INDEX}/_mget', json={'docs': docs}, timeout=120
        ).json()
        for d in resp.get('docs', []):
            emb = (d.get('_source') or {}).get('pe_embedding')
            if emb is not None:
                vecs.append(emb)
    mat = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return np.where(norms > 0, mat / norms, mat).astype(np.float32)


def main() -> None:
    ids = detected_frame_ids()
    logger.info('confirmed-plate frames: %d', len(ids))
    mat = fetch_embeddings(ids)
    logger.info('frames with an embedding: %d', mat.shape[0])
    logger.info('\n tau    groups  dropped   %%set  largest')
    for tau in THRESHOLDS:
        groups = near_dup_groups(mat, tau)
        dropped = sum(len(g) - 1 for g in groups)
        largest = max((len(g) for g in groups), default=0)
        pct = 100.0 * dropped / mat.shape[0] if mat.shape[0] else 0.0
        logger.info('%.3f  %6d  %7d  %5.1f  %7d', tau, len(groups), dropped, pct, largest)


if __name__ == '__main__':
    main()
