"""One entry point for the baseline: environment, ingest, Triton, endpoints, VLM.

Writes (and merges into) one JSON document with a fixed schema, so a phase can be
re-run alone. Phases (``--phases``, comma list; default all but ``vlm``):

* ``env``: GPU model/driver/CUDA, CPU, RAM, git sha, container image digests,
  Triton and OpenSearch versions, dataset pin hash, foreign GPU load before the runs.
* ``ingest``: ``--reps`` interleaved repetitions of ingest through
  ``POST .../ingest/upload`` and ``POST .../ingest/batch`` on a fresh project each,
  with the stage timers, Triton statistics, wire bytes, CPU seconds, GPU samples,
  cluster training wall time and settled OpenSearch storage per image and per vector.
* ``triton``: perf_analyzer points per model at batch 1/8/16/32 (SDK container).
* ``endpoints``: /detect, /embed/image, /faces/detect, /ocr/predict, sequential and concurrent.
* ``vlm``: ``POST .../vlm/label_batch`` on crops of the last ingest project.

    baseline_suite.py --manifest scripts/datasets/manifests/coco_bench_2000.json \\
        --images data/samples/coco_bench_2000/images --out artifacts_local/bench/v050.json \\
        --project base045 --api-url http://127.0.0.1:4903 --triton-url http://127.0.0.1:4900 \\
        --opensearch-url http://127.0.0.1:4907 --source-map HOST_DIR=/data/source/coco_bench_2000

Nothing here reads image content into the report; paths and names are not recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.suite_ingest import phase_ingest
from scripts.bench.suite_lib import SCHEMA
from scripts.bench.suite_phases import (
    PERF_BATCHES,
    PERF_MODELS,
    phase_endpoints,
    phase_triton,
    phase_vlm,
)
from scripts.bench.suite_probes import phase_env


PHASES = ('env', 'ingest', 'triton', 'endpoints', 'vlm')
DEFAULT_PHASES = ('env', 'ingest', 'triton', 'endpoints')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        '--manifest', type=Path, required=True, help='bench_set pin (scripts/datasets/manifests)'
    )
    p.add_argument('--images', type=Path, required=True, help='directory holding the pinned images')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--project', required=True, help='compose project name (container prefix)')
    p.add_argument('--api-url', required=True)
    p.add_argument('--triton-url', required=True)
    p.add_argument('--opensearch-url', required=True)
    p.add_argument(
        '--phases', default=','.join(DEFAULT_PHASES), help=f'comma list from {",".join(PHASES)}'
    )
    p.add_argument('--modes', default='upload,batch')
    p.add_argument(
        '--source-map',
        action='append',
        default=[],
        metavar='HOST=CONTAINER',
        help='rewrite image paths for /ingest/batch',
    )
    p.add_argument('--prefix', default='bl')
    p.add_argument('--gpu', default='0')
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument(
        '--limit', type=int, default=0, help='use only the first N pinned images (smoke runs)'
    )
    p.add_argument('--idle-samples', type=int, default=30)
    p.add_argument('--skip-cluster', action='store_true', help='no cluster training after ingest')
    p.add_argument('--skip-settle', action='store_true', help='no merge/storage accounting')
    p.add_argument(
        '--embedding-mode',
        choices=('all', 'lazy'),
        default='all',
        help='ingest embedding policy of each run project: all (detect + embed) or lazy (detect only)',
    )
    p.add_argument('--perf-dir', type=Path, default=Path('artifacts_local/bench/perf_analyzer'))
    p.add_argument('--perf-models', default=','.join(PERF_MODELS))
    p.add_argument('--perf-batches', default=','.join(str(b) for b in PERF_BATCHES))
    p.add_argument('--endpoint-images', type=int, default=200)
    p.add_argument('--vlm-crops', type=int, default=200)
    p.add_argument('--vlm-reps', type=int, default=2)
    p.add_argument('--vlm-batch', type=int, default=32)
    p.add_argument('--vlm-threads', type=int, default=8)
    p.add_argument('--vlm-gpu')
    p.add_argument(
        '--vlm-project', help='project whose crops are labeled (default: last upload rep)'
    )
    p.add_argument(
        '--vlm-param',
        action='append',
        default=[],
        metavar='K=V',
        help='query param for label_batch',
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.vlm_params = dict(kv.split('=', 1) for kv in args.vlm_param)
    args.perf_dir = args.perf_dir.resolve()
    args.images = args.images.resolve()
    phases = args.phases.split(',')
    if set(phases) - set(PHASES):
        sys.exit(f'--phases must be from {PHASES}')
    doc: dict[str, Any] = (
        json.loads(args.out.read_text()) if args.out.is_file() else {'schema': SCHEMA}
    )
    runners: dict[str, Callable[[], dict[str, Any]]] = {
        'ingest': lambda: phase_ingest(args),
        'triton': lambda: phase_triton(args),
        'endpoints': lambda: phase_endpoints(args),
        'vlm': lambda: phase_vlm(args, doc),
    }
    with httpx.Client(timeout=30.0) as client:
        for phase in phases:
            started = time.monotonic()
            result = phase_env(args, client) if phase == 'env' else runners[phase]()
            doc[phase] = {**doc.get(phase, {}), **result} if phase == 'triton' else result
            doc.setdefault('phase_seconds', {})[phase] = time.monotonic() - started
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
            print(f'phase {phase} done in {time.monotonic() - started:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
