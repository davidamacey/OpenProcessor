"""Scale benchmark driver: one bulk ingest with a forced cancel, a resume and resource sampling.

Starts ``POST /curation/projects/{p}/ingest/bulk`` for a folder, polls progress every 15 s
(ingested / processed / img/s recent and cumulative), cancels once at a fraction of the TRUE
final total (the folder preview's file count, not the growing discovery count), resumes by
starting the same source again, and samples GPU, host, container and OpenSearch state every
30 s. It writes ``<out>/<label>.jsonl`` (time series) and ``<out>/<label>.md`` (a table shaped
for docs/PERFORMANCE.md). It only reads the stack: HTTP to the API, ``nvidia-smi``,
``docker stats --no-stream``, ``docker exec <opensearch> curl`` and ``docker logs``.

    .venv/bin/python scripts/bench/scale_run.py --api http://127.0.0.1:4603 \\
        --project coco-100k --folder /data/bench/images --label coco-100k \\
        --stop-at 0.5 --out-dir artifacts_local/scale
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import UTC
from pathlib import Path
from typing import Any


POLL_S = 15.0
SAMPLE_S = 30.0
TERMINAL = ('completed', 'completed_with_errors', 'failed')
STOPPED = ('cancelled', 'interrupted')


def stop_count(total: int, fraction: float) -> int:
    """Absolute processed count at which to cancel, clamped inside the run."""
    if total < 2:
        raise ValueError('need at least 2 files to stop part-way')
    if not 0.0 <= fraction <= 1.0:
        raise ValueError('stop fraction must be within 0..1')
    return min(max(1, math.ceil(total * fraction)), total - 1)


def should_cancel(*, processed: int, stop_at: int, already_cancelled: bool) -> bool:
    return not already_cancelled and processed >= stop_at


def cumulative_rate(count: int, elapsed_s: float) -> float:
    return count / elapsed_s if elapsed_s > 0 else 0.0


def recent_rate(window: list[tuple[float, int]]) -> float:
    """img/s across a window of (time, count) points."""
    if len(window) < 2:
        return 0.0
    (t0, c0), (t1, c1) = window[0], window[-1]
    return cumulative_rate(c1 - c0, t1 - t0)


def _peak(samples: list[dict[str, Any]], key: str) -> float:
    vals: list[float] = []
    for s in samples:
        v = s.get(key)
        if isinstance(v, list):
            vals.extend(float(x) for x in v if x is not None)
        elif v is not None:
            vals.append(float(v))
    return max(vals) if vals else 0.0


def summarize(
    legs: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    *,
    total: int,
    baseline_img_s: float | None,
) -> dict[str, Any]:
    wall = sum(float(leg['wall_s']) for leg in legs)
    rate = cumulative_rate(total, wall)
    return {
        'total': total,
        'wall_s': wall,
        'legs_wall_s': [float(leg['wall_s']) for leg in legs],
        'failed': sum(int(leg['failed']) for leg in legs),
        'images_per_s': rate,
        'baseline_img_s': baseline_img_s,
        'pct_of_baseline': (100.0 * rate / baseline_img_s) if baseline_img_s else None,
        'peak_gpu_util': _peak(samples, 'gpu_util'),
        'peak_gpu_mem_mib': _peak(samples, 'gpu_mem_mib'),
        'peak_host_cpu_pct': _peak(samples, 'host_cpu_pct'),
        'peak_host_mem_gb': _peak(samples, 'host_mem_used_gb'),
        'peak_os_heap_pct': _peak(samples, 'os_heap_pct'),
        'peak_os_segments': _peak(samples, 'os_segments'),
        'os_store_bytes': _peak(samples, 'os_store_bytes'),
    }


def _gib(n: float) -> str:
    return f'{n / 1024**3:.2f} GiB'


def render_markdown(
    label: str, s: dict[str, Any], *, cluster_pass_s: float | None, db_bytes: float | None
) -> str:
    base = (
        f' ({s["pct_of_baseline"]:.0f} % of baseline {s["baseline_img_s"]:.1f} img/s)'
        if s.get('pct_of_baseline') is not None
        else ''
    )
    legs = ', '.join(f'{w:.0f} s' for w in s['legs_wall_s'])
    rows = [
        ('Images', f'{s["total"]:,}'),
        ('Throughput', f'{s["images_per_s"]:.2f} img/s{base}'),
        ('Wall time', f'{s["wall_s"]:.0f} s (legs: {legs})'),
        ('Failures', str(s['failed'])),
        ('Peak GPU util / mem', f'{s["peak_gpu_util"]:.0f} % / {s["peak_gpu_mem_mib"]:.0f} MiB'),
        ('Peak host CPU / mem', f'{s["peak_host_cpu_pct"]:.0f} % / {s["peak_host_mem_gb"]:.1f} GB'),
        (
            'OpenSearch peak heap / segments',
            f'{s["peak_os_heap_pct"]:.0f} % / {s["peak_os_segments"]:.0f}',
        ),
        (
            'Cluster pass',
            f'{cluster_pass_s:.0f} s' if cluster_pass_s is not None else 'not recorded',
        ),
        ('Index size on disk', _gib(db_bytes if db_bytes is not None else s['os_store_bytes'])),
    ]
    out = [f'### {label}', '', '| Measure | Value |', '|---|---|']
    out += [f'| {k} | {v} |' for k, v in rows]
    return '\n'.join(out) + '\n'


def call(url: str, body: dict[str, Any] | None = None, *, post: bool = False) -> Any:
    data = json.dumps(body if body is not None else {}).encode() if post else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={'content-type': 'application/json'},
        method='POST' if post else 'GET',
    )
    with urllib.request.urlopen(req, timeout=120) as r:  # nosec B310  operator-supplied http API base
        return json.load(r)


def _run(cmd: list[str], timeout: float = 60) -> str:
    try:
        return subprocess.run(  # nosec B603 B607  fixed argv list, no shell
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ''


def _os_get(container: str, path: str) -> Any:
    out = _run(['docker', 'exec', container, 'curl', '-s', f'localhost:9200/{path}'])
    try:
        return json.loads(out)
    except ValueError:
        return None


def _cpu_pct(prev: tuple[int, int] | None) -> tuple[float, tuple[int, int]]:
    fields = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    cur = (sum(fields), fields[3] + fields[4])
    if prev is None:
        return 0.0, cur
    dt, di = cur[0] - prev[0], cur[1] - prev[1]
    return (100.0 * (dt - di) / dt if dt > 0 else 0.0), cur


def _mem_used_gb() -> float:
    kv = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k, v = line.split(':')
        kv[k] = int(v.split()[0])
    return (kv['MemTotal'] - kv['MemAvailable']) / 1024**2


def take_sample(
    project: str, os_container: str, prefix: str, cpu_prev: tuple[int, int] | None
) -> tuple[dict[str, Any], tuple[int, int]]:
    gpu = _run(
        [
            'nvidia-smi',
            '--query-gpu=index,utilization.gpu,memory.used',
            '--format=csv,noheader,nounits',
        ]
    )
    rows = [[x.strip() for x in ln.split(',')] for ln in gpu.strip().splitlines() if ln]
    cpu, cpu_now = _cpu_pct(cpu_prev)
    stats = _run(['docker', 'stats', '--no-stream', '--format', '{{json .}}'])
    containers = {}
    for ln in stats.splitlines():
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if d.get('Name', '').startswith(prefix):
            containers[d['Name']] = {'cpu': d.get('CPUPerc'), 'mem': d.get('MemUsage')}
    idx = _os_get(os_container, '_cat/indices?format=json&bytes=b') or []
    mine = [i for i in idx if f'{project}__' in i.get('index', '')]
    seg = _os_get(os_container, '_cat/segments?format=json') or []
    mine_names = {i['index'] for i in mine}
    jvm = _os_get(os_container, '_nodes/stats/jvm,indices') or {}
    node: dict[str, Any] = next(iter(jvm.get('nodes', {}).values()), {})
    merges = node.get('indices', {}).get('merges', {})
    sample = {
        't': time.time(),
        'gpu_util': [int(r[1]) for r in rows],
        'gpu_mem_mib': [int(r[2]) for r in rows],
        'host_cpu_pct': cpu,
        'host_mem_used_gb': _mem_used_gb(),
        'containers': containers,
        'os_store_bytes': sum(int(i.get('store.size') or 0) for i in mine),
        'os_docs': sum(int(i.get('docs.count') or 0) for i in mine),
        'os_segments': sum(1 for s in seg if s.get('index') in mine_names),
        'os_merges_current': merges.get('current'),
        'os_merges_total': merges.get('total'),
        'os_heap_pct': node.get('jvm', {}).get('mem', {}).get('heap_used_percent'),
    }
    return sample, cpu_now


class Sampler(threading.Thread):
    def __init__(self, path: Path, project: str, os_container: str, prefix: str, every: float):
        super().__init__(daemon=True)
        self.path, self.project, self.os_container, self.prefix = (
            path,
            project,
            os_container,
            prefix,
        )
        self.every, self._stop_evt = every, threading.Event()
        self.samples: list[dict[str, Any]] = []

    def run(self) -> None:
        prev = None
        with self.path.open('a') as fh:
            while not self._stop_evt.is_set():
                s, prev = take_sample(self.project, self.os_container, self.prefix, prev)
                self.samples.append(s)
                fh.write(json.dumps({'kind': 'sample', **s}) + '\n')
                fh.flush()
                self._stop_evt.wait(self.every)

    def stop(self) -> None:
        self._stop_evt.set()


def preview_total(base: str, folder: str, expected: int | None) -> int:
    pv = call(f'{base}/ingest/bulk/preview', {'folder': folder}, post=True)
    if pv.get('partial'):
        if expected is None:
            raise SystemExit(
                f'preview is partial ({pv.get("stop_reason")}) at {pv["files"]} files; '
                'pass --expected-total with the real count'
            )
        return expected
    return int(pv['files'])


def cluster_pass_seconds(container: str, project: str, since_epoch: float) -> float | None:
    """Best effort: first non-idle refresh line for the project to the next idle line after it."""
    out = _run(['docker', 'logs', '--timestamps', '--since', str(int(since_epoch)), container], 120)
    start = None
    for ln in out.splitlines():
        if f'project={project} ' not in ln or 'action=' not in ln:
            continue
        ts = re.match(r'(\S+)Z? ', ln)
        if not ts:
            continue
        t = _parse_ts(ts.group(1))
        idle = 'action=none' in ln
        if start is None and not idle:
            start = t
        elif start is not None and idle:
            return t - start
    return None


def _parse_ts(s: str) -> float:
    from datetime import datetime

    s = s.rstrip('Z')
    head, _, frac = s.partition('.')
    dt = datetime.strptime(head, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=UTC)
    return dt.timestamp() + (float('0.' + frac) if frac else 0.0)


def drive(args: argparse.Namespace) -> int:
    base = f'{args.api.rstrip("/")}/curation/projects/{args.project}'
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    series = out / f'{args.label}.jsonl'
    total = preview_total(base, args.folder, args.expected_total)
    stop_at = stop_count(total, args.stop_at) if args.stop_at > 0 else None
    print(f'total={total} stop_at={stop_at}', flush=True)
    sampler = Sampler(
        series, args.project, args.opensearch_container, args.container_prefix, SAMPLE_S
    )
    sampler.start()
    t_begin = time.time()
    legs: list[dict[str, Any]] = []
    body = {'folder': args.folder}
    cancelled = False
    with series.open('a') as log:
        for leg in (1, 2):
            run = call(f'{base}/ingest/bulk', body, post=True)
            rid, t0 = run['run_id'], time.time()
            window: list[tuple[float, int]] = []
            while True:
                time.sleep(POLL_S)
                run = call(f'{base}/ingest/bulk/{rid}')
                prog = run['progress']
                now = time.time()
                window = [*window, (now, int(prog.get('processed') or 0))][-5:]
                log.write(
                    json.dumps(
                        {
                            'kind': 'poll',
                            'leg': leg,
                            't': now,
                            'status': run['status'],
                            'ingested': prog.get('ingested'),
                            'processed': prog.get('processed'),
                            'failed': prog.get('failed'),
                            'skipped': prog.get('skipped'),
                            'discovered': prog.get('discovered'),
                            'img_s_recent': recent_rate(window),
                            'img_s_cum': cumulative_rate(int(prog.get('processed') or 0), now - t0),
                            'server_img_s': prog.get('images_per_s'),
                        }
                    )
                    + '\n'
                )
                log.flush()
                if stop_at is not None and should_cancel(
                    processed=int(prog.get('processed') or 0),
                    stop_at=stop_at,
                    already_cancelled=cancelled,
                ):
                    call(f'{base}/ingest/bulk/{rid}/cancel', {}, post=True)
                    cancelled = True
                    print(f'cancel sent at processed={prog.get("processed")}', flush=True)
                if run['status'] in TERMINAL or (cancelled and run['status'] in STOPPED):
                    break
            legs.append(
                {
                    'leg': leg,
                    'run_id': rid,
                    'status': run['status'],
                    'wall_s': time.time() - t0,
                    'ingested': int(run['progress'].get('ingested') or 0),
                    'failed': int(run['progress'].get('failed') or 0),
                    'skipped': int(run['progress'].get('skipped') or 0),
                }
            )
            if run['status'] in TERMINAL or not cancelled:
                break
    sampler.stop()
    sampler.join(timeout=SAMPLE_S + 30)
    cluster_s = (
        cluster_pass_seconds(args.cluster_container, args.project, t_begin)
        if args.cluster_container
        else None
    )
    final = legs[-1]
    final_failed = final['failed']
    # Second leg reports the whole run (earlier files are skipped), so failures are the last leg's.
    summary = summarize(
        [{**leg, 'failed': 0} for leg in legs[:-1]] + [{**final, 'failed': final_failed}],
        sampler.samples,
        total=total,
        baseline_img_s=args.baseline_img_s,
    )
    md = render_markdown(args.label, summary, cluster_pass_s=cluster_s, db_bytes=None)
    (out / f'{args.label}.md').write_text(md)
    with series.open('a') as log:
        log.write(
            json.dumps({'kind': 'summary', 'legs': legs, 'cluster_pass_s': cluster_s, **summary})
            + '\n'
        )
    print(md)
    return 0 if final['status'] in ('completed',) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--api', required=True, help='API base, e.g. http://127.0.0.1:4603')
    p.add_argument('--project', required=True)
    p.add_argument('--folder', required=True, help='Folder path as the API container sees it')
    p.add_argument('--label', required=True, help='Names the output files and the table heading')
    p.add_argument(
        '--stop-at',
        type=float,
        default=0.5,
        help='Cancel at this fraction of the true total (0 = never)',
    )
    p.add_argument('--out-dir', required=True)
    p.add_argument(
        '--expected-total', type=int, default=None, help='Real file count if the preview is partial'
    )
    p.add_argument('--baseline-img-s', type=float, default=None)
    p.add_argument('--opensearch-container', default='opmono-opensearch')
    p.add_argument('--container-prefix', default='opmono')
    p.add_argument(
        '--cluster-container',
        default=None,
        help='Cluster-refresh container to read pass timing from',
    )
    return drive(p.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
