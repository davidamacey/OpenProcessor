"""Retrieval-quality baseline for OpenProcessor search on public COCO val2017 data.

Subcommands (each writes JSON under ``--out``; defaults keep raw output in ``artifacts_local/``):

    prep              seeded sample of COCO val2017 images + 5 captions + category sets
    ingest-curation   upload the sample into a throw-away curation project, wait until embedded
    ingest-core       ingest the sample through the core ``POST /ingest`` (MobileCLIP global index)
    text-curation     text-to-image through ``GET /curation/projects/{project}/search/text``
    text-core         text-to-image through ``POST /search/text``
    image-core        leave-one-out image-to-image through ``POST /search/image``
    filters           class_name / cluster_id filtered curation search: latency and correctness
    dedupe            near-duplicate variants through the core ingest and ``GET /query/duplicates``
    core-cleanup      delete this sample's images from the core indexes (``DELETE /query/image``)

Ground truth is the human COCO captions (text-to-image) and COCO instance-category set overlap
(image-to-image, Jaccard at least 0.5; a proxy, not a human relevance judgement). Metrics: recall@k,
MRR, nDCG and precision@k; results are written up in ``docs/research/retrieval_baseline.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench import retrieval_eval_data as data, retrieval_eval_runs as runs


def host_load() -> dict[str, Any]:
    l1, l5, l15 = os.getloadavg()
    return {'load1': l1, 'load5': l5, 'load15': l15, 'nproc': os.cpu_count()}


def cmd_prep(a: argparse.Namespace) -> None:
    work = Path(a.work)
    caps, inst = data.load_annotations(Path(a.cache_dir))
    rows = data.build_sample(caps, inst, a.n, a.seed)
    data.download_sample(rows, work / 'images')
    (work / 'sample.json').write_text(json.dumps({'seed': a.seed, 'rows': rows}, indent=1))
    print(f'{len(rows)} images in {work / "images"}')


def cmd_ingest_curation(a: argparse.Namespace) -> None:
    sample = runs.load_sample(a.work)
    started = time.monotonic()
    mapping = runs.ingest_curation(a.api, a.project, Path(a.work), sample, a.batch)
    (Path(a.work) / f'curation_map_{a.project}.json').write_text(json.dumps(mapping))
    print(f'ingested {len(mapping)} images in {time.monotonic() - started:.0f}s')


def cmd_ingest_core(a: argparse.Namespace) -> None:
    sample = runs.load_sample(a.work)
    started = time.monotonic()
    summary = runs.ingest_core(a.api, Path(a.work), sample, a.workers)
    print(json.dumps(summary), f'{time.monotonic() - started:.0f}s')


def cmd_core_cleanup(a: argparse.Namespace) -> None:
    sample = runs.load_sample(a.work)
    print(json.dumps(runs.core_cleanup(a.api, sample, include_variants=True)))


def emit(a: argparse.Namespace, name: str, payload: dict[str, Any]) -> None:
    payload = {
        'schema_version': 1,
        'started_utc': payload.pop('_started'),
        'finished_utc': datetime.now(UTC).isoformat(timespec='seconds'),
        'api': a.api,
        'sample_seed': data.SEED,
        **payload,
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{name}.json'
    path.write_text(json.dumps(payload, indent=1))
    print(f'wrote {path}')


def cmd_eval(a: argparse.Namespace) -> None:
    sample = runs.load_sample(a.work)
    started = datetime.now(UTC).isoformat(timespec='seconds')
    load0 = host_load()
    with httpx.Client(timeout=120.0) as client:
        if a.cmd == 'text-curation':
            res = runs.eval_text_curation(client, a, sample)
        elif a.cmd == 'text-core':
            res = runs.eval_text_core(client, a, sample)
        elif a.cmd == 'image-core':
            res = runs.eval_image_core(client, a, sample)
        elif a.cmd == 'filters':
            res = runs.eval_filters(client, a, sample)
        else:
            res = runs.eval_dedupe(client, a, sample)
    emit(
        a,
        f'{a.cmd}_{a.tag}',
        {'_started': started, 'host_load_start': load0, 'host_load_end': host_load(), **res},
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        'cmd',
        choices=[
            'prep',
            'ingest-curation',
            'ingest-core',
            'text-curation',
            'text-core',
            'image-core',
            'filters',
            'dedupe',
            'core-cleanup',
        ],
    )
    p.add_argument('--api', default='http://127.0.0.1:5603')
    p.add_argument('--project', default='')
    p.add_argument(
        '--work', required=True, help='scratch dir outside the repo (images, sample.json)'
    )
    p.add_argument('--out', default='artifacts_local/retrieval_baseline')
    p.add_argument('--tag', default='run1', help='suffix of the result file name')
    p.add_argument('--cache-dir', default='cache/datasets')
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--seed', type=int, default=data.SEED)
    p.add_argument('--batch', type=int, default=20)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument(
        '--captions', type=int, choices=(1, 5), default=1, help='captions per image used as queries'
    )
    p.add_argument('--limit', type=int, default=0, help='use only the first N queries (0 = all)')
    p.add_argument('--min-score', type=float, default=0.0, help='core text search min_score')
    p.add_argument('--depth', type=int, default=200, help='curation hits fetched per page')
    p.add_argument(
        '--variants', type=int, default=100, help='images turned into near-duplicate variants'
    )
    a = p.parse_args(argv)
    if a.cmd == 'prep':
        cmd_prep(a)
    elif a.cmd == 'ingest-curation':
        cmd_ingest_curation(a)
    elif a.cmd == 'ingest-core':
        cmd_ingest_core(a)
    elif a.cmd == 'core-cleanup':
        cmd_core_cleanup(a)
    else:
        cmd_eval(a)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
