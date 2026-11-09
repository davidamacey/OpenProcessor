"""#439 / 4D-0: where does a project create spend its time? Run N creates, print the stage table.

``POST /curation/projects`` now answers 202 and a ``project_create`` job builds the project; the
job's result carries the seconds of each stage (``stages``). This script creates ``--runs``
throwaway projects one after another (public data not needed: the projects stay empty), follows
each job to its end, deletes the project, and prints a markdown table of the median, p95 and max
seconds per stage plus the 202 latency and the total from POST to ``active``.

    .venv/bin/python scripts/bench/project_create_stages.py --api http://127.0.0.1:4603 --runs 10

The coordinator posts the table to #439. It needs the stack (API, a ``jobs`` worker, OpenSearch,
Postgres) and nothing else; no GPU.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.curation._project_worker_utils import curation_api_prefix


POLL_S = 0.25
JOB_TIMEOUT_S = 600.0
DONE = ('completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted')


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def render_table(runs: list[dict[str, Any]]) -> str:
    """Markdown rows: the stages in the order the first run reported them, then the wall times."""
    series: dict[str, list[float]] = {}
    for run in runs:
        for name, seconds in run['stages'].items():
            series.setdefault(name, []).append(float(seconds))
    series['route_202 (POST until 202)'] = [r['post_s'] for r in runs]
    series['total (POST until job done)'] = [r['total_s'] for r in runs]
    lines = [
        f'| stage | median s | p95 s | max s |  (n={len(runs)})',
        '|---|---:|---:|---:|',
    ]
    for name, values in series.items():
        lines.append(
            f'| {name} | {statistics.median(values):.3f} | {_p95(values):.3f} | {max(values):.3f} |'
        )
    return '\n'.join(lines)


def one_create(http: httpx.Client, base: str, slug: str) -> dict[str, Any]:
    start = time.perf_counter()
    posted = http.post(f'{base}/projects', json={'slug': slug, 'display_name': slug}, timeout=60)
    post_s = time.perf_counter() - start
    posted.raise_for_status()
    if posted.status_code != 202:
        raise RuntimeError(f'expected 202, got {posted.status_code}')
    job_id = posted.json()['job_id']
    job: dict[str, Any] = {}
    while time.perf_counter() - start < JOB_TIMEOUT_S:
        job = http.get(f'{base}/jobs/{job_id}', timeout=30).json()
        if job['status'] in DONE:
            break
        time.sleep(POLL_S)
    total_s = time.perf_counter() - start
    if job.get('status') != 'completed':
        raise RuntimeError(f'{slug}: job ended {job.get("status")}: {job.get("error_message")}')
    return {'post_s': post_s, 'total_s': total_s, 'stages': (job.get('result') or {})['stages']}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--api', default='http://127.0.0.1:4603')
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args(argv)
    base = f'{args.api}{curation_api_prefix()}'
    prefix = f'stage-{uuid.uuid4().hex[:6]}'
    runs: list[dict[str, Any]] = []
    with httpx.Client() as http:
        for i in range(args.runs):
            slug = f'{prefix}-{i}'
            try:
                runs.append(one_create(http, base, slug))
            finally:
                http.delete(
                    f'{base}/projects/{slug}',
                    params={'confirm': slug},
                    timeout=120,
                )
    print(render_table(runs))
    return 0


if __name__ == '__main__':
    sys.exit(main())
