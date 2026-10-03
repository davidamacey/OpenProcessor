"""One-command baseline run: ingest a manifest into a throwaway project and report.

Creates a project ``<slug-prefix>-<utc stamp>``, sets its ingest policy, ingests
the manifest through ``POST /curation/projects/{slug}/ingest/batch``, waits for
the requested later stages, and writes ``<out>.json`` and ``<out>.md``: images/s,
items/s, bytes moved, per-stage timers (the ``op_pipeline_stage_*`` series plus
any other ``op_*`` series scraped from the API, worker and Triton metrics
endpoints), GPU utilization and memory, and storage per image after a
refresh and force-merge. The report holds aggregate numbers only: no image
content and no image paths.

The batch route reads files by path inside the API container, so the manifest
paths must be visible there; ``--path-map HOST=CONTAINER`` rewrites a prefix.
The first ``--warmup`` images are ingested but excluded from rates and deltas.

    run_baseline.py MANIFEST --api-url URL --slug-prefix NAME --policy all|selected|lazy \\
        --out PATH_NO_SUFFIX [--stages ingest,embed,region,vlm,cluster] \\
        [--opensearch-url URL] [--worker-metrics-url URL] [--triton-metrics-url URL]
    run_baseline.py --compare before.json after.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.baseline_report import (
    SCHEMA,
    BatchOutcome,
    compare_reports,
    ingest_summary,
    metrics_delta,
    parse_nvidia_smi,
    parse_prometheus,
    storage_summary,
    summarize_gpu,
    to_markdown,
)
from scripts.bench.select_baseline_set import read_manifest


STAGE_ORDER = ('ingest', 'region', 'embed', 'vlm', 'cluster')
API_METRIC_PREFIXES = ('op_', 'http_request_duration_seconds')
TRITON_METRIC_PREFIXES = (
    'nv_inference_compute_input_duration_us',
    'nv_inference_compute_infer_duration_us',
    'nv_inference_compute_output_duration_us',
    'nv_inference_queue_duration_us',
    'nv_inference_exec_count',
    'nv_inference_count',
)
NVIDIA_SMI = [
    'nvidia-smi',
    '--query-gpu=index,utilization.gpu,memory.used',
    '--format=csv,noheader,nounits',
]


def read_nvidia_smi(
    runner: Callable[..., str] | None = None,
) -> dict[str, tuple[float, float]] | None:
    """One nvidia-smi reading, or ``None`` when the tool is absent or fails."""
    run = runner or (
        lambda cmd: subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=10
        ).stdout
    )
    try:
        return parse_nvidia_smi(run(NVIDIA_SMI))
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, OSError):
        return None


class GpuSampler:
    """Background GPU sampler; a reader returning ``None`` means "no GPU data"."""

    def __init__(
        self, reader: Callable[[], dict[str, tuple[float, float]] | None], interval: float = 1.0
    ) -> None:
        self._reader = reader
        self._interval = interval
        self._samples: list[dict[str, tuple[float, float]]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> None:
        reading = self._reader()
        if reading:
            self._samples.append(reading)

    def start(self) -> None:
        if self._reader() is None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict[str, dict[str, float]]:
        return summarize_gpu(self._samples)


def map_path(path: str, path_maps: Sequence[str]) -> str:
    for spec in path_maps:
        host, _, container = spec.partition('=')
        if path.startswith(host):
            return container + path[len(host) :]
    return path


def _scrape(client: httpx.Client, url: str | None) -> dict[str, float]:
    if not url:
        return {}
    resp = client.get(url)
    resp.raise_for_status()
    return parse_prometheus(resp.text)


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


def _scoped(slug: str, suffix: str) -> str:
    return f'/projects/{slug}{suffix}'


def _set_policy(client: httpx.Client, slug: str, policy: str, classes: list[str]) -> None:
    revision = client.get(_scoped(slug, '/ingest/policy')).raise_for_status().json()['revision']
    body = {'embedding': {'mode': policy, 'classes': classes}, 'expected_revision': revision}
    client.put(_scoped(slug, '/ingest/policy'), json=body).raise_for_status()


def _ingest_batch(
    client: httpx.Client, slug: str, paths: list[str], clock: Callable[[], float]
) -> BatchOutcome:
    started = clock()
    resp = client.post(
        _scoped(slug, '/ingest/batch'),
        json={'items': [{'path': p, 'source': 'baseline'} for p in paths]},
    )
    resp.raise_for_status()
    elapsed = clock() - started
    summary = resp.json()['summary']
    return BatchOutcome(
        images=len(paths),
        ok=summary['successful'],
        duplicates=summary['duplicates'],
        failed=summary['failed'],
        items=summary['crops_indexed'],
        embedded=summary['n_embedded'],
        seconds=elapsed,
    )


def _wait_region_drain(
    client: httpx.Client,
    slug: str,
    *,
    poll: float,
    timeout: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> float:
    started = clock()
    while True:
        state = client.get(_scoped(slug, '/ingest/region_drain')).raise_for_status().json()
        if state['drained']:
            return clock() - started
        if clock() - started > timeout:
            raise TimeoutError(f'region stage not drained after {timeout:.0f}s')
        sleep(poll)


def _run_autolabel(
    client: httpx.Client,
    slug: str,
    params: dict[str, str],
    *,
    poll: float,
    timeout: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> float:
    """Start one auto-label job, wait for it, return its wall seconds."""
    started = clock()
    job = (
        client.post(_scoped(slug, '/pipeline/auto_label/start'), params=params)
        .raise_for_status()
        .json()
    )
    while True:
        state = (
            client.get(_scoped(slug, f'/pipeline/auto_label/status/{job["job_id"]}'))
            .raise_for_status()
            .json()
        )
        if state['status'] in ('completed', 'failed', 'cancelled'):
            if state['status'] != 'completed':
                raise RuntimeError(f'auto_label job ended {state["status"]}')
            durations = state.get('stage_durations') or {}
            return sum(durations.values()) if durations else clock() - started
        if clock() - started > timeout:
            raise TimeoutError(f'auto_label job not finished after {timeout:.0f}s')
        sleep(poll)


def _settle_store_bytes(client: httpx.Client, os_url: str, slug: str) -> int:
    names = client.get(_scoped(slug, '')).raise_for_status().json()['resources']['indexes'].values()
    target = ','.join(sorted(set(names)))
    client.post(f'{os_url}/{target}/_refresh').raise_for_status()
    client.post(
        f'{os_url}/{target}/_forcemerge', params={'max_num_segments': '1'}
    ).raise_for_status()
    stats = client.get(f'{os_url}/{target}/_stats/store').raise_for_status().json()
    return int(stats['_all']['total']['store']['size_in_bytes'])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('manifest', type=Path)
    p.add_argument('--api-url', required=True)
    p.add_argument('--api-prefix', default=os.environ.get('OP_API_PREFIX', '/curation'))
    p.add_argument('--slug-prefix', required=True)
    p.add_argument('--policy', choices=('all', 'selected', 'lazy'), required=True)
    p.add_argument(
        '--selected-classes', nargs='*', default=[], help='classes for --policy selected'
    )
    p.add_argument(
        '--out',
        type=Path,
        required=True,
        help='report path without suffix (.json and .md are written)',
    )
    p.add_argument('--stages', default='ingest', help=f'comma list from {",".join(STAGE_ORDER)}')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument(
        '--warmup', type=int, default=100, help='leading images excluded from rates and deltas'
    )
    p.add_argument('--path-map', action='append', default=[], metavar='HOST=CONTAINER')
    p.add_argument('--opensearch-url', help='enables the settle step and storage per image')
    p.add_argument('--worker-metrics-url')
    p.add_argument('--triton-metrics-url')
    p.add_argument(
        '--crop-cache-dir', type=Path, help='host directory of the crop cache, sized at the end'
    )
    p.add_argument('--gpu-interval', type=float, default=1.0)
    p.add_argument('--poll-s', type=float, default=5.0)
    p.add_argument('--timeout-s', type=float, default=7200.0)
    return p


def run(
    args: argparse.Namespace,
    *,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    gpu_reader: Callable[[], dict[str, tuple[float, float]] | None] = read_nvidia_smi,
) -> dict[str, Any]:
    stages = set(args.stages.split(','))
    unknown = stages - set(STAGE_ORDER)
    if unknown or 'ingest' not in stages:
        sys.exit(f'--stages must include ingest and only {STAGE_ORDER}; got {args.stages}')
    if args.policy == 'selected' and not args.selected_classes:
        sys.exit('--policy selected needs --selected-classes')
    paths = read_manifest(args.manifest)
    sizes = [Path(p).stat().st_size for p in paths]
    sent = [map_path(p, args.path_map) for p in paths]
    slug = f'{args.slug_prefix}-{datetime.now(UTC).strftime("%Y%m%d%H%M%S")}'
    api_base = args.api_url.rstrip('/') + args.api_prefix.rstrip('/')
    wait = {'poll': args.poll_s, 'timeout': args.timeout_s, 'sleep': sleep, 'clock': clock}

    with httpx.Client(
        timeout=httpx.Timeout(3600.0), transport=transport, base_url=api_base
    ) as client:
        client.post('/projects', json={'slug': slug, 'display_name': slug}).raise_for_status()
        _set_policy(client, slug, args.policy, args.selected_classes)

        warm = min(args.warmup, len(sent))
        batches = [
            _ingest_batch(client, slug, sent[start : min(start + args.batch_size, warm)], clock)
            for start in range(0, warm, args.batch_size)
        ]
        sources = {
            'api': (f'{args.api_url}/metrics', API_METRIC_PREFIXES),
            'worker': (args.worker_metrics_url, API_METRIC_PREFIXES),
            'triton': (args.triton_metrics_url, TRITON_METRIC_PREFIXES),
        }
        before = {name: _scrape(client, url) for name, (url, _) in sources.items()}
        sampler = GpuSampler(gpu_reader, args.gpu_interval)
        sampler.start()
        started = clock()
        measured = [
            _ingest_batch(client, slug, sent[start : start + args.batch_size], clock)
            for start in range(warm, len(sent), args.batch_size)
        ]
        ingest_wall = clock() - started
        batches.extend(measured)

        stage_wall: dict[str, dict[str, float]] = {}
        if 'region' in stages:
            stage_wall['region_drain'] = {'wall_s': _wait_region_drain(client, slug, **wait)}
        label_runs = {
            'embed': {'embed_missing': 'true', 'train_clusters': 'false', 'run_vlm': 'false'},
            'vlm': {'run_vlm': 'true', 'train_clusters': 'false'},
            'cluster': {'train_clusters': 'true', 'run_vlm': 'false'},
        }
        for name, params in label_runs.items():
            if name in stages and (name != 'embed' or args.policy == 'lazy'):
                stage_wall[name] = {'wall_s': _run_autolabel(client, slug, params, **wait)}
        sampler.stop()

        after = {name: _scrape(client, url) for name, (url, _) in sources.items()}
        total_ok = sum(b.ok for b in batches)
        storage: dict[str, float] = {}
        if args.opensearch_url:
            cache = _dir_bytes(args.crop_cache_dir) if args.crop_cache_dir else 0
            storage = storage_summary(
                images=total_ok,
                store_bytes=_settle_store_bytes(client, args.opensearch_url.rstrip('/'), slug),
                crop_cache_bytes=cache,
            )

    measured_bytes = sum(sizes[warm:])
    manifest_lines = args.manifest.read_text(encoding='utf-8').splitlines()
    sha = next((ln.split(': ', 1)[1] for ln in manifest_lines if ln.startswith('# sha256: ')), '')
    report = {
        'schema': SCHEMA,
        'manifest': {
            'name': args.manifest.name,
            'sha256': sha,
            'count': len(paths),
            'bytes': sum(sizes),
        },
        'config': {
            'policy': args.policy,
            'batch_size': args.batch_size,
            'warmup': warm,
            'stages': sorted(stages),
            'project': slug,
        },
        'ingest': ingest_summary(measured, wall_s=ingest_wall, input_bytes=measured_bytes),
        'stages': stage_wall,
        'storage': storage,
        'gpu': sampler.summary(),
        'metrics': {
            name: metrics_delta(before[name], after[name], prefixes)
            for name, (url, prefixes) in sources.items()
            if url
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    args.out.with_suffix('.md').write_text(to_markdown(report) + '\n', encoding='utf-8')
    return report


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['--compare']:
        files = argparse.ArgumentParser(prog='run_baseline.py --compare')
        files.add_argument('before', type=Path)
        files.add_argument('after', type=Path)
        ns = files.parse_args(argv[1:])
        before, after = (json.loads(p.read_text(encoding='utf-8')) for p in (ns.before, ns.after))
        print(compare_reports(before, after))
        return 0
    report = run(build_parser().parse_args(argv))
    print(to_markdown(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
