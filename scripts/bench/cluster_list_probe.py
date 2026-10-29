"""Latency probe for ``GET /clusters`` on a seeded throw-away project.

``seed`` creates a project through the API and bulk-indexes N synthetic items
into it (public COCO class names, no images and no vectors: the cluster list
reads only item fields); ``probe`` times the list call; ``delete`` removes the
project again. Only the project named on the command line is ever touched.

    cluster_list_probe.py seed   --api URL --opensearch URL --project SLUG --items 100000
    cluster_list_probe.py probe  --api URL --project SLUG --reps 3 --calls 30 --out run.json
    cluster_list_probe.py delete --api URL --project SLUG

``probe`` records the 1-minute load average before and after each repetition
(a repetition above load 16 is marked noisy, as in the plan's protocol) and
reports p50 and p95 per repetition and over all calls.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx


PREFIX = os.environ.get('OP_API_PREFIX', '/curation')
RESIDUAL_CLUSTER_ID_OFFSET = 10000
CLASS_CLUSTERS = 80
CANDIDATE_CLUSTERS = 512
NOISY_LOAD = 16.0
BULK_CHUNK = 5000

COCO = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic_light',
    'fire_hydrant',
    'stop_sign',
    'parking_meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports_ball',
    'kite',
    'baseball_bat',
    'baseball_glove',
    'skateboard',
    'surfboard',
    'tennis_racket',
    'bottle',
    'wine_glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot_dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted_plant',
    'bed',
    'dining_table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell_phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy_bear',
    'hair_drier',
    'toothbrush',
]


_SOURCES = (
    {'class_source': 'human'},
    {'class_source': 'vlm'},
    {'class_source': 'vlm'},
    {'class_source': 'vlm_unmatched'},
)


def _load1() -> float:
    return float(Path('/proc/loadavg').read_text().split()[0])


def _doc(i: int, rng: random.Random, weights: list[float], now: datetime) -> dict[str, Any]:
    if rng.random() < 0.4:
        cid = rng.randrange(CLASS_CLUSTERS)
        base = cid
    else:
        slot = rng.choices(range(CANDIDATE_CLUSTERS), weights)[0]
        cid = RESIDUAL_CLUSTER_ID_OFFSET + slot
        base = slot % len(COCO)
    detector = base if rng.random() < 0.85 else rng.randrange(len(COCO))
    conf = round(rng.uniform(0.25, 1.0), 4)
    stamp = (now - timedelta(seconds=rng.randrange(30 * 86400))).isoformat()
    doc: dict[str, Any] = {
        'crop_id': f'syn-{i:08d}',
        'image_id': f'syn-img-{i // 4:08d}',
        'cluster_id': cid,
        'cluster_distance': round(rng.uniform(0.0, 0.6), 4),
        'cluster_distance_cluster_id': cid,
        'cluster_nearest_id': cid if rng.random() < 0.88 else cid + 1,
        'detector_class_name': COCO[detector],
        'detector_confidence': conf,
        'confidence': conf,
        'class_validated': rng.random() < 0.1,
        'test_holdout': rng.random() < 0.04,
        'crop_rank_in_image': rng.randrange(1, 5),
        'created_at': stamp,
        'updated_at': stamp,
    }
    if cid < RESIDUAL_CLUSTER_ID_OFFSET or rng.random() < 0.3:
        labelled = base if rng.random() < 0.9 else rng.randrange(len(COCO))
        doc['class_name'] = COCO[labelled]
        doc['class_id'] = labelled
    doc.update(rng.choice(_SOURCES))
    if rng.random() < 0.25:
        doc['vlm_confidence'] = rng.choice(['high', 'medium', 'low'])
    if rng.random() < 0.01:
        doc['class_excluded'] = True
    return doc


def seed(api: str, opensearch: str, project: str, items: int, seed_value: int) -> None:
    rng = random.Random(seed_value)
    weights = [1.0 / (r + 1) ** 0.7 for r in range(CANDIDATE_CLUSTERS)]
    created = httpx.post(
        f'{api}{PREFIX}/projects',
        json={'slug': project, 'display_name': f'{project} (latency probe, synthetic)'},
        timeout=120,
    )
    if created.status_code != 409:  # an interrupted earlier seed left the project
        created.raise_for_status()
    summary = httpx.get(f'{api}{PREFIX}/projects/{project}', timeout=60)
    summary.raise_for_status()
    index = summary.json()['resources']['indexes']['items']
    now = datetime.now(UTC)
    started = time.monotonic()
    with httpx.Client(timeout=300) as client:
        for start in range(0, items, BULK_CHUNK):
            lines: list[str] = []
            for i in range(start, min(start + BULK_CHUNK, items)):
                lines.append(json.dumps({'index': {'_index': index, '_id': f'syn-{i:08d}'}}))
                lines.append(json.dumps(_doc(i, rng, weights, now)))
            resp = client.post(
                f'{opensearch}/_bulk',
                content='\n'.join(lines) + '\n',
                headers={'content-type': 'application/x-ndjson'},
            )
            resp.raise_for_status()
            if resp.json().get('errors'):
                raise SystemExit(f'bulk errors near item {start}: {resp.text[:400]}')
        client.post(f'{opensearch}/{index}/_refresh').raise_for_status()
    print(f'seeded {items} items into {index} in {time.monotonic() - started:.1f}s')


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def probe(
    api: str, project: str, reps: int, calls: int, params: dict[str, str], endpoint: str
) -> dict[str, Any]:
    url = f'{api}{PREFIX}/projects/{project}/{endpoint}'
    runs: list[dict[str, Any]] = []
    with httpx.Client(timeout=300) as client:
        client.get(url, params=params).raise_for_status()  # warm caches
        for _ in range(reps):
            load_start = _load1()
            times: list[float] = []
            body: dict[str, Any] = {}
            for _ in range(calls):
                t0 = time.perf_counter()
                resp = client.get(url, params=params)
                times.append((time.perf_counter() - t0) * 1000)
                resp.raise_for_status()
                body = resp.json()
            load_end = _load1()
            runs.append(
                {
                    'p50_ms': round(statistics.median(times), 1),
                    'p95_ms': round(_pct(times, 0.95), 1),
                    'max_ms': round(max(times), 1),
                    'load1_start': load_start,
                    'load1_end': load_end,
                    'noisy': max(load_start, load_end) > NOISY_LOAD,
                    'times_ms': [round(t, 1) for t in times],
                }
            )
    return {
        'url': url,
        'params': params,
        'cards': len(body.get('items', [])),
        'truncated': body.get('truncated'),
        'response_bytes': len(json.dumps(body)),
        'runs': runs,
        'p95_ms_median_of_runs': statistics.median(r['p95_ms'] for r in runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in ('seed', 'probe', 'delete'):
        p = sub.add_parser(name)
        p.add_argument('--api', required=True)
        p.add_argument('--project', required=True)
        if name == 'seed':
            p.add_argument('--opensearch', required=True)
            p.add_argument('--items', type=int, default=100_000)
            p.add_argument('--seed', type=int, default=7)
        if name == 'probe':
            p.add_argument('--reps', type=int, default=3)
            p.add_argument('--calls', type=int, default=30)
            p.add_argument('--param', action='append', default=[], metavar='K=V')
            p.add_argument(
                '--endpoint',
                default='clusters',
                help='path under the project, e.g. clusters/10001/facets (no default query)',
            )
            p.add_argument('--out')
    args = parser.parse_args()
    if args.cmd == 'seed':
        seed(args.api, args.opensearch, args.project, args.items, args.seed)
    elif args.cmd == 'probe':
        params = (
            {'limit': '50', 'per_cluster': '4', 'max_clusters': '1000'}
            if args.endpoint == 'clusters'
            else {}
        )
        params.update(dict(kv.split('=', 1) for kv in args.param))
        result = probe(args.api, args.project, args.reps, args.calls, params, args.endpoint)
        text = json.dumps(result, indent=2)
        if args.out:
            Path(args.out).write_text(text)
        print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, indent=2))
        for run in result['runs']:
            print({k: v for k, v in run.items() if k != 'times_ms'})
    else:
        resp = httpx.delete(
            f'{args.api}{PREFIX}/projects/{args.project}',
            params={'confirm': args.project},
            timeout=300,
        )
        print(resp.status_code, resp.text[:300])


if __name__ == '__main__':
    main()
