"""Seed the live write-path verification harness with deterministic data.

Run against the throwaway stack in ``docker/test/compose.yml`` (never a real
deployment — every index it touches must carry the ``verify_`` prefix, and the
script refuses to run otherwise).

What it creates
---------------
1. Every curation index, through the application's own
   :func:`src.routers.curation._common._ensure_indexes` — that chains
   ``create_curation_indexes`` plus the eleven ``ensure_*`` migrations in the
   right order, with the one intentionally-unwired migration correctly
   skipped. Hand-rolling the mappings here would mean testing a schema the
   product never produces.
2. Eight classes through :meth:`ClassRegistry.add_class`, synced into the
   classes index — before any item doc, since the label endpoints reject an
   unknown class id.
3. ~490 item documents with ``_id == crop_id`` (every OCC path addresses by
   doc id), spread across all three cluster-id bands: negative (noise /
   excluded), the class band, and the residual/candidate band.
4. Non-degenerate embeddings: each intended sub-cluster gets a random unit
   centroid, and its members are ``centroid + N(0, sigma)`` re-normalized to
   unit length. ``sigma`` is scaled as ``0.15 / sqrt(dim)`` so the *noise
   vector* has norm ~0.15 against a unit centroid regardless of
   dimensionality — with a flat 0.15 per component a 1024-d noise vector
   would have norm ~4.8 and swamp the centroid, collapsing every group into
   mutual near-orthogonality and making the refine assertions pass
   vacuously for the opposite reason.
5. A small solid-colour JPEG per source image, so the thumbnail/region-crop
   routes (and the VLM label path, which crops real pixels) have something
   real to read.

Determinism: one fixed RNG seed, one fixed id scheme. Re-running with
``--wipe`` reproduces byte-identical cohorts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.region_state import RegionStatus  # noqa: E402 - needs the sys.path fix above


RANDOM_SEED = 1337
NOISE_NORM = 0.15

# Class names are deliberately generic shipping-domain slugs matching the
# built-in prompt pack's vocabulary, so the fake VLM's fixed answer resolves.
CLASS_NAMES = (
    'box',
    'envelope',
    'tube',
    'crate',
    'pallet',
    'drum',
    'sack',
    'canister',
)

N_IMAGES = 60
HDD_SOURCE = 'verify_src'


@dataclass(frozen=True)
class Cohort:
    """One seeded group of item docs sharing a cluster id and label shape."""

    prefix: str
    cluster_id: int | None
    size: int
    subgroups: int
    # ``class_plan`` maps a member index onto (class_id, class_source,
    # class_validated). None means "unlabeled candidate".
    class_id: int | None = None
    class_source: str | None = None
    class_validated: bool = False
    # Secondary class mixed in to drive purity below the auto-promote gate.
    minority_class_id: int | None = None
    minority_count: int = 0
    excluded: bool = False
    # '' for no region, else a RegionStatus member (see region_state.py).
    region: str = ''


# The cohort table IS the test fixture contract — every scenario in
# tests/live/ addresses one of these prefixes. Keep the comments accurate.
COHORTS: tuple[Cohort, ...] = (
    # Class cluster 0: pure, half human-validated and half unvalidated
    # v6-model rows -> the auto-promote positive case.
    Cohort('cls0', 0, 40, 3, class_id=0, class_source='item_model', class_validated=False),
    # Class cluster 1: 60/40 class mix -> purity 0.6, below the 0.85 gate,
    # so auto-promote must leave it alone. Also the labels-scenario cohort
    # (v6-sourced rows are never part of the human holdout cohort).
    Cohort(
        'cls1',
        1,
        40,
        2,
        class_id=1,
        class_source='item_model',
        class_validated=False,
        minority_class_id=2,
        minority_count=16,
    ),
    # Human-validated class clusters -> the holdout-freeze cohort.
    Cohort('cls2', 2, 40, 2, class_id=2, class_source='human', class_validated=True),
    Cohort('cls3', 3, 40, 2, class_id=3, class_source='human', class_validated=True),
    Cohort('cls4', 4, 40, 2, class_id=4, class_source='human', class_validated=True),
    Cohort('cls5', 5, 40, 2, class_id=5, class_source='human', class_validated=True),
    Cohort('cls6', 6, 20, 1, class_id=6, class_source='human', class_validated=True),
    # Merge-target cohort: never human-validated, so it never acquires a
    # frozen holdout row and stays mergeable.
    Cohort('cls7', 7, 20, 1, class_id=7, class_source='item_model', class_validated=False),
    # Residual / candidate band.
    Cohort('cnd10000', 10000, 30, 3),
    Cohort('cnd10001', 10001, 20, 2),
    Cohort('cnd10002', 10002, 15, 1),
    # Region-of-interest cohorts (own candidate bucket so they never skew a
    # class cluster's purity).
    Cohort('rgn', 10003, 40, 2, region=RegionStatus.DETECTED),
    Cohort('rgnfp', 10003, 12, 1, region=RegionStatus.FALSE_POSITIVE),
    # Negative band.
    Cohort('noise', -1, 80, 8),
    Cohort('excl', -2, 15, 1, excluded=True),
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _fail(message: str) -> None:
    print(f'FATAL: {message}')
    raise SystemExit(2)


def _assert_verify_scoped(cfg: Any) -> None:
    """Refuse to touch anything that is not a `verify_`-prefixed index."""
    names = {
        'images_index': cfg.images_index,
        'items_index': cfg.items_index,
        'labels_confirmed_index': cfg.labels_confirmed_index,
        'classes_index': cfg.classes_index,
        'clusters_index': cfg.clusters_index,
        'settings_index': cfg.settings_index,
    }
    bad = {k: v for k, v in names.items() if not v.startswith('verify_')}
    if bad:
        _fail(
            'refusing to seed: these configured index names lack the '
            f'`verify_` prefix: {bad}. Set the OP_*_INDEX env vars first.'
        )


def _unit_vectors(rng: Any, count: int, dim: int) -> Any:
    import numpy as np

    vecs = rng.normal(size=(count, dim))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.astype('float32')


def _members_around(rng: Any, centroid: Any, count: int, dim: int) -> Any:
    """``count`` unit vectors scattered tightly around ``centroid``."""
    import numpy as np

    sigma = NOISE_NORM / math.sqrt(dim)
    members = centroid[None, :] + rng.normal(scale=sigma, size=(count, dim))
    members /= np.linalg.norm(members, axis=1, keepdims=True)
    return members.astype('float32')


def _round_vec(vec: Any) -> list[float]:
    return [round(float(v), 5) for v in vec]


def _write_images(source_root: Path, count: int) -> list[dict[str, Any]]:
    """Write ``count`` tiny solid-colour JPEGs; return their image docs."""
    from PIL import Image

    source_root.mkdir(parents=True, exist_ok=True)
    docs: list[dict[str, Any]] = []
    for i in range(count):
        name = f'img_{i:03d}.jpg'
        path = source_root / name
        colour = (37 * i % 256, 91 * i % 256, 151 * i % 256)
        Image.new('RGB', (96, 96), colour).save(path, format='JPEG', quality=80)
        docs.append(
            {
                'image_id': f'img_{i:03d}',
                'image_path': str(path),
                'hdd_source': HDD_SOURCE,
                'width': 96,
                'height': 96,
                'imohash': f'imo{i:06d}',
                'indexed_at': _utc_now().isoformat(),
            }
        )
    return docs


def _bbox_for(index: int) -> list[float]:
    """A small, always-valid normalized bbox that varies per item."""
    x1 = 0.05 + (index % 5) * 0.12
    y1 = 0.05 + (index % 7) * 0.09
    return [round(x1, 4), round(y1, 4), round(x1 + 0.22, 4), round(y1 + 0.18, 4)]


def _region_bbox_for(index: int) -> list[float]:
    x1 = 0.30 + (index % 4) * 0.08
    y1 = 0.40 + (index % 3) * 0.07
    return [round(x1, 4), round(y1, 4), round(x1 + 0.10, 4), round(y1 + 0.06, 4)]


def _build_items(
    cfg: Any,
    fields: Any,
    image_docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize every item doc, deterministically."""
    import numpy as np

    rng = np.random.default_rng(RANDOM_SEED)
    item_dim = cfg.encoder_embedding_dim
    region_dim = cfg.encoder_embedding_dim

    now = _utc_now()
    docs: list[dict[str, Any]] = []
    global_index = 0

    for cohort in COHORTS:
        centroids = _unit_vectors(rng, cohort.subgroups, item_dim)
        region_centroids = _unit_vectors(rng, max(cohort.subgroups, 1), region_dim)
        per_group = [cohort.size // cohort.subgroups] * cohort.subgroups
        for i in range(cohort.size - sum(per_group)):
            per_group[i] += 1

        member_index = 0
        for group_idx, group_size in enumerate(per_group):
            vectors = _members_around(rng, centroids[group_idx], group_size, item_dim)
            region_vectors = _members_around(
                rng, region_centroids[group_idx], group_size, region_dim
            )
            for local_idx in range(group_size):
                crop_id = f'{cohort.prefix}_{member_index:04d}'
                image = image_docs[global_index % len(image_docs)]
                is_minority = (
                    cohort.minority_class_id is not None and member_index < cohort.minority_count
                )
                class_id = cohort.minority_class_id if is_minority else cohort.class_id
                class_name = CLASS_NAMES[class_id] if class_id is not None else None
                # Half of the pure class cluster is a settled human label;
                # the rest stays v6-sourced so auto-promote has candidates.
                validated = cohort.class_validated
                class_source = cohort.class_source
                if cohort.prefix == 'cls0' and member_index % 5 < 2:
                    validated = True
                    class_source = 'human'

                doc: dict[str, Any] = {
                    'crop_id': crop_id,
                    'image_id': image['image_id'],
                    'image_path': image['image_path'],
                    'hdd_source': HDD_SOURCE,
                    'request_id': 'seed',
                    'bbox_norm': _bbox_for(global_index),
                    'class_id': class_id,
                    'class_name': class_name,
                    'class_source': class_source,
                    'class_validated': bool(validated),
                    'label_source': 'human' if class_source == 'human' else class_source,
                    'confidence': round(0.45 + (global_index % 50) / 100.0, 3),
                    'cluster_id': cohort.cluster_id,
                    'cluster_distance': round(0.02 + (local_idx % 10) / 100.0, 4),
                    'cluster_subid': None,
                    'crop_area_norm': round(0.04 + (global_index % 17) / 100.0, 4),
                    'crop_rank_in_image': 1 + (global_index % 3),
                    'blur_lap_var': round(20.0 + (global_index % 40), 3),
                    'blur_lap_ratio': round(0.3 + (global_index % 60) / 100.0, 3),
                    'test_holdout': False,
                    'class_excluded': cohort.excluded,
                    'classifier_raw_confidence': round(0.3 + (global_index % 60) / 100.0, 3),
                    # Stored probe predictions so the review-queue sorting and
                    # the mistakenness scorer need no model at all.
                    'probe_pred_class': class_name or CLASS_NAMES[global_index % 8],
                    'probe_pred_entropy': round(0.05 + (global_index % 90) / 100.0, 4),
                    'probe_pred_confidence': round(0.99 - (global_index % 90) / 100.0, 4),
                    'probe_pred_margin': round(0.02 + (global_index % 50) / 100.0, 4),
                    'probe_disagreement': global_index % 11 == 0,
                    'probe_model_version': 'seed-probe-v1',
                    'probe_scored_at': now.isoformat(),
                    'pe_embedding': _round_vec(vectors[local_idx]),
                    'created_at': (now - timedelta(minutes=global_index)).isoformat(),
                    'updated_at': (now - timedelta(minutes=global_index)).isoformat(),
                }
                if cohort.excluded:
                    doc['excluded_at'] = now.isoformat()
                    doc['excluded_by'] = 'seed'
                    doc['excluded_reason'] = 'ignore'
                if cohort.region:
                    doc[fields.bbox_norm] = _region_bbox_for(global_index)
                    doc[fields.score] = round(0.35 + (global_index % 60) / 100.0, 3)
                    doc[fields.detector] = 'fake_detector'
                    doc[fields.detector_version] = '1'
                    doc[fields.detector_chain] = ['fake_detector:hit']
                    doc[fields.bbox_frame] = 'source'
                    doc[fields.detected_at] = now.isoformat()
                    doc[fields.embedding] = _round_vec(region_vectors[local_idx])
                    doc[fields.validated] = False
                    if cohort.region == RegionStatus.FALSE_POSITIVE:
                        doc[fields.status] = RegionStatus.FALSE_POSITIVE
                        doc[fields.verified] = False
                        doc[fields.cluster_id] = -100
                        doc[fields.cluster_distance] = 0.0
                    else:
                        doc[fields.status] = RegionStatus.DETECTED
                        doc[fields.verified] = True
                        doc[fields.cluster_id] = 1
                        doc[fields.cluster_distance] = round(0.05 + (local_idx % 10) / 100.0, 4)
                docs.append(doc)
                member_index += 1
                global_index += 1
    return docs


def _confirmed_label_docs(item_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A confirmed-label row per merge-cohort item, so a class merge has
    something to relabel in the labels-confirmed index too."""
    out: list[dict[str, Any]] = []
    for doc in item_docs:
        if not str(doc['crop_id']).startswith('cls7_'):
            continue
        out.append(
            {
                'label_id': f'lbl_{doc["crop_id"]}',
                'crop_id': doc['crop_id'],
                'image_path': doc['image_path'],
                'bbox_norm': doc['bbox_norm'],
                'class_id': doc['class_id'],
                'class_name': doc['class_name'],
                'label_source': 'seed',
                'confirmed_at': _utc_now().isoformat(),
            }
        )
    return out


async def _bulk(client: Any, index: str, docs: list[dict[str, Any]], id_key: str) -> None:
    chunk = 200
    for start in range(0, len(docs), chunk):
        body: list[dict[str, Any]] = []
        for doc in docs[start : start + chunk]:
            body.append({'index': {'_index': index, '_id': doc[id_key]}})
            body.append(doc)
        resp = await client.bulk(body=body, refresh=False)
        if resp.get('errors'):
            first = next(
                (
                    item
                    for item in resp.get('items', [])
                    if any('error' in v for v in item.values())
                ),
                None,
            )
            _fail(f'bulk index into {index} reported errors, first: {json.dumps(first)[:600]}')
    await client.indices.refresh(index=index)


async def _seed(args: argparse.Namespace) -> int:
    from opensearchpy import AsyncOpenSearch

    from src.clients.curation_opensearch import ClassRegistry
    from src.config import get_curation_config, get_region_fields
    from src.routers.curation._common import _ensure_indexes

    cfg = get_curation_config()
    fields = get_region_fields()
    _assert_verify_scoped(cfg)

    client = AsyncOpenSearch(hosts=[args.opensearch_url], timeout=60, max_retries=3)
    try:
        if args.wipe:
            for name in (
                cfg.images_index,
                cfg.items_index,
                cfg.labels_confirmed_index,
                cfg.classes_index,
                cfg.clusters_index,
                cfg.settings_index,
            ):
                try:
                    await client.indices.delete(index=name, ignore=[404])
                except Exception as exc:
                    print(f'  (wipe) {name}: {exc}')
            registry_path = Path(cfg.class_registry_path)
            if registry_path.parent.is_dir():
                for stale in registry_path.parent.glob(f'{registry_path.stem}*.json'):
                    stale.unlink()
            for sub in ('exports', 'state', 'crop_cache'):
                target = Path(args.data_root) / sub
                if target.is_dir():
                    shutil.rmtree(target)
            jobs_dir = Path(args.data_root) / 'jobs'
            if jobs_dir.is_dir():
                for stale_job in jobs_dir.iterdir():
                    if stale_job.is_file():
                        stale_job.unlink()
            print('wiped previous harness state')

        # 1. Indexes, through the application's own bootstrap chain.
        await _ensure_indexes(client)
        print(f'indexes ensured: {cfg.items_index}, {cfg.images_index}, ...')

        # 2. Class registry — before any item or label write.
        registry = ClassRegistry(cfg.class_registry_path)
        existing = {c.class_name for c in registry.load().classes}
        for name in CLASS_NAMES:
            if name not in existing:
                registry.add_class(name, group='shipping')
        synced = await registry.sync_to_opensearch(client)
        print(f'class registry: {len(CLASS_NAMES)} classes, synced={synced}')

        # 3. Source images + image docs.
        image_docs = _write_images(Path(args.source_root), N_IMAGES)
        await _bulk(client, cfg.images_index, image_docs, 'image_id')
        print(f'images: {len(image_docs)} jpegs + docs -> {cfg.images_index}')

        # 4. Item docs.
        item_docs = _build_items(cfg, fields, image_docs)
        await _bulk(client, cfg.items_index, item_docs, 'crop_id')
        print(f'items: {len(item_docs)} docs -> {cfg.items_index}')

        # 5. Confirmed labels for the merge cohort.
        label_docs = _confirmed_label_docs(item_docs)
        await _bulk(client, cfg.labels_confirmed_index, label_docs, 'label_id')
        print(f'confirmed labels: {len(label_docs)} docs -> {cfg.labels_confirmed_index}')

        count = await client.count(index=cfg.items_index)
        print(f'done. {cfg.items_index} now holds {count.get("count")} documents')
    finally:
        await client.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--opensearch-url',
        default=os.environ.get('OPENSEARCH_URL', 'http://opensearch:9200'),
        help='OpenSearch REST endpoint of the verification harness.',
    )
    parser.add_argument(
        '--data-root',
        default='/verify-data',
        help='Root of the harness data mount (its exports/state/jobs are wiped with --wipe).',
    )
    parser.add_argument(
        '--source-root',
        default=None,
        help=(
            'Where to WRITE the generated JPEGs. Defaults to the configured '
            'OP_SOURCE_ROOT. Set this when the writer (host) and the reader '
            '(container) see the mount at different paths.'
        ),
    )
    parser.add_argument(
        '--wipe',
        action='store_true',
        help='Delete the verify_* indexes, registry and job files first.',
    )
    args = parser.parse_args()
    if args.source_root is None:
        from src.config import get_curation_config

        args.source_root = str(get_curation_config().source_root)
    return asyncio.run(_seed(args))


if __name__ == '__main__':
    raise SystemExit(main())
